from data.create_dataframe import create_dataframe
from data.dataset import SpeechDataset,collate_fn
from engine.decoder import GreedySearchDecoder
from tokenizer.tokenizer import create_corpus,create_vocab_model
from torch.utils.data import DataLoader, Subset,RandomSampler
import sentencepiece as sp
import torch


dataframe = create_dataframe('all_data')
dataframe = dataframe.sample(frac=1)

train_len = int(len(dataframe) * 0.7)
valid_len = int(len(dataframe) * 0.9)

train_dataframe = dataframe.iloc[:train_len, :]
valid_dataframe = dataframe.iloc[train_len:valid_len, :]
test_dataframe = dataframe.iloc[valid_len:, :]

x = torch.tensor([[1,2,3],[2,3,4]])
print(x.size())
print(x.size(0))
print(x.shape[0])
