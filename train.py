from model.deepspeech2 import DeepSpeech2
from data.dataset import SpeechDataset, collate_fn
from data.create_dataframe import create_dataframe
from engine.engine import *
from engine.decoder import *
from torch.utils.data import DataLoader
import sentencepiece as sp
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="path to data directory")
    parser.add_argument('--epoch', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--mode', type=str, default='greedy', help='decode mode (greedy or beam)')

    args = parser.parse_args()

    # Define device

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define vocab model
    vocal_model = sp.SentencePieceProcessor()
    vocal_model.load('vocab_model/vocab.model')

    # Define dataset

    dataframe = create_dataframe(args.data_path)
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

    # Define model

    model = DeepSpeech2(num_classes=54).to(device)

    # Define optimizer, lr_scheduler, trainer,decoder

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=2)
    trainer = Trainer(lr_scheduler)

    if args.mode == 'greedy':
        decoder = GreedySearchDecoder(decoder=vocal_model)

    elif args.mode == 'beam':
        decoder = BeamSearchDecoder(decoder=vocal_model)

    # Training progress

    for epoch in range(args.epoch):
        print(f'Epoch {epoch + 1} / {args.epoch}')
        train_loss = train_model(model, train_dataset, train_dataloader, optimizer, device)
        print(f'Traing loss: {train_loss}')
        valid_loss, wer = valid_model(model, valid_dataset, valid_dataloader, device, decoder)
        print(f'Validation loss: {valid_loss}')
        print(f'Word Error Rate: {wer}')
        trainer(epoch, valid_loss, model, optimizer)
        inference(model, device, test_dataset, decoder)
