import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from model.deepspeech2 import DeepSpeech2
from data.create_dataframe import create_dataframe
from data.dataset import SpeechDataset, collate_fn
from engine.engine import CustomCallBack
from engine.decoder import BeamSearchDecoder, GreedySearchDecoder
import sentencepiece as sp
import torch.nn as nn
import torch.nn.functional as F


class SpeechModule(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader, device):
        super(SpeechModule, self).__init__()
        self.model = model.to(device)
        self.ctc_loss = nn.CTCLoss(blank=0)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._device = device

    def forward(self, x):
        x = x.to(self._device)
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
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vocab", default="vocab_model/vocab.model", type=str)
    parser.add_argument("--data", default="all_data", type=str)
    parser.add_argument("--mode", type=str, default='greedy')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    """Load vocab model"""
    vocal_model = sp.SentencePieceProcessor()
    vocal_model.load(args.vocab)

    """Create dataframe,dataset,dataloader"""

    dataframe = create_dataframe(args.data)
    dataframe = dataframe.sample(frac=1)

    train_len = int(len(dataframe) * 0.7)
    valid_len = int(len(dataframe) * 0.9)

    train_dataframe = dataframe.iloc[:train_len, :]
    valid_dataframe = dataframe.iloc[train_len:valid_len, :]
    test_dataframe = dataframe.iloc[valid_len:, :]

    train_dataset = SpeechDataset(dataframe=train_dataframe, phase='train', vocab_model=vocal_model)
    valid_dataset = SpeechDataset(dataframe=valid_dataframe, phase='valid', vocab_model=vocal_model)
    test_dataset = SpeechDataset(dataframe=test_dataframe, phase='test', vocab_model=vocal_model)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)

    """Init model"""

    model = DeepSpeech2(dropout=0.2, n_feats=128, rnn_dim=128, num_classes=54)

    """Create decoder"""
    if args.mode == 'beam':
        decoder = BeamSearchDecoder()
    elif args.mode == 'greedy':
        decoder = GreedySearchDecoder()

    """Create callback and train"""
    call_back = CustomCallBack(test_dataset=test_dataset, decoder=decoder, vocab_model=vocal_model)

    module = SpeechModule(model, train_dataloader, valid_dataloader, device)
    trainer = pl.Trainer(max_epochs=args.epoch, checkpoint_callback=checkpoint_callback(), callbacks=[call_back, ],
                         accelerator='gpu', gpus=0)
    trainer.fit(module)
