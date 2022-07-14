from torch.utils.data import RandomSampler, DataLoader
from data.dataset import collate_fn
from tqdm import tqdm
import editdistance
import torch


class Trainer():
    def __init__(self, lr_scheduler, patience=5, save_path='checkpoints/best_model.pth'):
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.counter = 0
        self.min_delta = 1e-3
        self.stop = False

    def __call__(self, epoch, current_val_loss, model, optimizer):
        if self.best_val_loss - current_val_loss > self.min_delta:
            print(f'Validation loss improved from {self.best_val_loss} to {current_val_loss}!')
            self.best_val_loss = current_val_loss
            self.counter = 0
            print('Saving best model ... ')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, self.save_path)

        else:
            self.counter += 1
            print(f'Validation did not improve from {self.best_val_loss} ! Counter {self.counter} of {self.patience}.')
            if self.counter < self.patience:
                self.lr_scheduler.step(current_val_loss)
            else:
                self.stop = True


def train_model(model, dataset, dataloader, optimizer, device):
    model.train()
    train_loss = 0.0
    counter = 0
    for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        counter += 1
        optimizer.zero_grad()

        spectrogram = data[0].to(device)
        label = data[1].to(device)
        label_length = data[2].to(device)

        _, loss = model(spectrogram, label, label_length)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    return train_loss / counter


def valid_model(model, dataset, dataloader, device):
    model.eval()
    valid_loss = 0.0
    counter = 0

    with torch.no_grad():
        for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            spectrogram = data[0].to(device)
            label = data[1].to(device)
            label_length = data[2].to(device)

            probs, loss = model(spectrogram, label, label_length)
            valid_loss += loss.item()
        return valid_loss / counter


def inference(model, device, dataset, decoder, sp_model):
    model.eval()
    sample_set = RandomSampler(data_source=dataset, num_samples=3)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sample_set, collate_fn=collate_fn)

    counter = 0
    wer = 0.0

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            counter += 1
            spectrogram = data[0].to(device)
            labels = data[1].to(device)
            label_lengths = data[2].to(device)

            probs, _ = model(spectrogram)
            results, indices = decoder(probs)

            label = sp_model.decode_ids(labels[0].tolist())
            prediction = results[0]

            wer += editdistance.eval(prediction,label) / len(label)
            print("Label: {0:70} Prediction: {1}".format(label, prediction))

        print(f'Word Error Rate {wer}')
