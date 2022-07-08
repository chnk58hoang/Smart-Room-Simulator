from data.create_dataframe import create_dataframe
from data.dataset import SpeechDataset,collate_fn
from engine.decoder import GreedySearchDecoder
from tokenizer.tokenizer import create_corpus,create_vocab_model
from torch.utils.data import DataLoader
import sentencepiece as sp
import torch

dataframe = create_dataframe('all_data')
create_corpus(corpus_path='vocab_model/corpus.txt',dataframe=dataframe)
create_vocab_model(corpus_path='vocab_model/corpus.txt',vocab_size=54,model_type='bpe',model_prefix='vocab_model/vocab')

vocal_model = sp.SentencePieceProcessor()
vocal_model.load('vocab_model/vocab.model')


dataset = SpeechDataset(dataframe,vocab_model=vocal_model,phase='test')

dataloader = DataLoader(dataset,batch_size = 3,shuffle=True,collate_fn=collate_fn)

decoder = GreedySearchDecoder(decoder=vocal_model,blank=0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
