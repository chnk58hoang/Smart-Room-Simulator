from torch.utils.data import DataLoader
from model.deepspeech2 import DeepSpeech2
from data.create_dataframe import create_dataframe
from data.dataset import SpeechDataset, collate_fn
import sentencepiece as sp

dataframe = create_dataframe('all_data')
dataframe = dataframe.sample(frac=1)

vocal_model = sp.SentencePieceProcessor()
vocal_model.load('vocab_model/vocab.model')

train_len = int(len(dataframe) * 0.7)
valid_len = int(len(dataframe) * 0.9)

train_dataframe = dataframe.iloc[:train_len, :]
valid_dataframe = dataframe.iloc[train_len:valid_len, :]
test_dataframe = dataframe.iloc[valid_len:, :]

train_dataset = SpeechDataset(dataframe=train_dataframe, phase='train', vocab_model=vocal_model)
valid_dataset = SpeechDataset(dataframe=valid_dataframe, phase='valid', vocab_model=vocal_model)
test_dataset = SpeechDataset(dataframe=test_dataframe, phase='test', vocab_model=vocal_model)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)

model = DeepSpeech2(num_classes=54)

maxx = 0
for data in train_dataloader:
    maxx = max(data[0].size(-1),maxx)

print(maxx)
