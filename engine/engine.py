from torch.utils.data import Subset,DataLoader
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


def valid_model(model, dataset, dataloader, device, decoder):
    model.eval()
    valid_loss = 0.0
    wer = 0.0
    counter = 0

    with torch.no_grad():
        for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            spectrogram = data[0].to(device)
            label = data[1].to(device)
            label_length = data[2].to(device)

            probs, loss = model(spectrogram, label, label_length)
            valid_loss += loss.item()

            result, indices = decoder(probs)

            for i in range(dataloader.batch_size):
                wer += editdistance.eval(indices[i], label[i][:label_length[i]]) / label_length[i]

        wer /= dataloader.batch_size
        return valid_loss / counter, wer


def inference(model, device, dataset,decoder):
    model.eval()
    subset_indices = torch.randint(size=(3,), low=0, high=len(dataset))

    subset = Subset(dataset, indices=subset_indices)
    dataloader = DataLoader(subset, batch_size=1)
    with torch.no_grad():
        for data in enumerate(dataloader):
            spectrogram = data[0].to(device)
            labels = data[1].to(device)
            label_lengths = data[2].to(device)

            probs, _ = model(spectrogram)
            results, indices = decoder(probs)

            for i in range(dataloader.batch_size):
                print("Prediction")
                print(results[i])
                print("Label")
                print()
