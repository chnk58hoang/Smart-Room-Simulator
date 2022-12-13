import editdistance
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import RandomSampler, DataLoader
from data.dataset import collate_fn


class CustomCallBack(pl.Callback):
    def __init__(self,val_dataset,test_dataset, decoder, vocab_model):
        super(CustomCallBack, self).__init__()
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.decoder = decoder
        self.vocab_model = vocab_model

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        subset = RandomSampler(data_source=self.val_dataset, num_samples=3)
        dataloader = DataLoader(self.val_dataset, batch_size=1, sampler=subset,collate_fn=collate_fn)
        all_preds = []
        all_labels = []
        for batch, data in enumerate(dataloader):
            probs = pl_module(data[0])
            labels = data[1]
            probs = F.softmax(probs, dim=-1)
            preds = self.decoder(probs)

            for pred in preds:
                all_preds.append(self.vocab_model.decode_ids(pred))

            for label in labels:
                all_labels.append(self.vocab_model.decode_ids(label.tolist()))

        mean_norm_ed = 0.0

        for i in range(len(all_labels)):
            print("Label: {0:70} Prediction: {1}".format(all_labels[i],
                                                         all_preds[i]))
            mean_norm_ed += editdistance.eval(all_preds[i], all_labels[i]) / (len(all_labels[i]) * len(all_labels))

        print(f"Mean Normalized editdistance:{mean_norm_ed}")

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        subset = RandomSampler(data_source=self.test_dataset, num_samples=3)
        dataloader = DataLoader(self.test_dataset, batch_size=1, sampler=subset, collate_fn=collate_fn)
        all_preds = []
        all_labels = []
        for batch, data in enumerate(dataloader):
            probs = pl_module(data[0])
            labels = data[1]
            probs = F.softmax(probs, dim=-1)
            preds = self.decoder(probs)

            for pred in preds:
                all_preds.append(self.vocab_model.decode_ids(pred))

            for label in labels:
                all_labels.append(self.vocab_model.decode_ids(label.tolist()))

        mean_norm_ed = 0.0

        for i in range(len(all_labels)):
            print("Label: {0:70} Prediction: {1}".format(all_labels[i],
                                                         all_preds[i]))
            mean_norm_ed += editdistance.eval(all_preds[i], all_labels[i]) / (len(all_labels[i]) * len(all_labels))

        print(f"Mean Normalized editdistance:{mean_norm_ed}")

