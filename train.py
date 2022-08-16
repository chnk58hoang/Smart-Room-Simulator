import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from model.deepspeech2 import DeepSpeech2
from data.create_dataframe import create_dataframe
from data.dataset import SpeechDataset, collate_fn
from engine.engine import CustomCallBack
from engine.decoder import BeamSearchDecoder
import sentencepiece as sp
import torch.nn as nn
import torch.nn.functional as F

vocal_model = sp.SentencePieceProcessor()
vocal_model.load('vocab_model/vocab.model')

dataframe = create_dataframe('all_data')
dataframe = dataframe.sample(frac=1)

train_len = int(len(dataframe) * 0.7)
valid_len = int(len(dataframe) * 0.9)

train_dataframe = dataframe.iloc[:train_len, :]
valid_dataframe = dataframe.iloc[train_len:valid_len, :]
test_dataframe = dataframe.iloc[valid_len:, :]

train_dataset = SpeechDataset(dataframe=train_dataframe, phase='train', vocab_model=vocal_model)
valid_dataset = SpeechDataset(dataframe=valid_dataframe, phase='valid', vocab_model=vocal_model)
test_dataset = SpeechDataset(dataframe=test_dataframe, phase='test', vocab_model=vocal_model)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collate_fn, num_workers=8)

model = DeepSpeech2(dropout=0.2,n_feats=128,rnn_dim=128,num_classes=54)


class SpeechModule(pl.LightningModule):
    def __init__(self, model):
        super(SpeechModule, self).__init__()
        self.model = model
        self.ctc_loss = nn.CTCLoss(blank=0)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        return [self.optimizer]

    def training_step(self, batch, batch_idx):
        spec, target, target_lengths = batch
        inputs = self.forward(spec)
        input_lengths = torch.full(size=(inputs.size(0),), fill_value=inputs.size(1), dtype=torch.long)
        inputs = inputs.permute(1, 0, 2)
        inputs = F.log_softmax(inputs, dim=-1)
        loss = self.ctc_loss(inputs, target, input_lengths, target_lengths)
        tensorboard_logs = {'train_loss': loss}

        return {"loss": loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        spec, target, target_lengths = batch
        inputs = self(spec)
        input_lengths = torch.full(size=(inputs.size(0),), fill_value=inputs.size(1))
        inputs = inputs.permute(1, 0, 2)
        loss = self.ctc_loss(inputs, target, input_lengths, target_lengths)

        return {"val_loss": loss}

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.scheduler.step(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


def checkpoint_callback():
    return ModelCheckpoint(
        dirpath='checkpoints',
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )


decoder = BeamSearchDecoder()
call_back = CustomCallBack(test_dataset=test_dataset, decoder=decoder, vocab_model=vocal_model)

module = SpeechModule(model)
trainer = pl.Trainer(max_epochs=10, checkpoint_callback=checkpoint_callback(), callbacks=[call_back, ])
trainer.fit(module)