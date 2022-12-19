from torch.utils.data import Dataset
from torchaudio import transforms
from torch.distributions import uniform
from torch.nn.utils.rnn import pad_sequence
import torchaudio.functional as F
import torch
import torch.nn as nn
import torchaudio
import numpy as np


class SpeechDataset(Dataset):
    def __init__(self, dataframe, phase, vocab_model):
        super(SpeechDataset, self).__init__()
        self.dataframe = dataframe
        self.phase = phase
        self.vocab_model = vocab_model
        self.get_melspectrogram = transforms.MelSpectrogram(sample_rate=16000, n_mels=64, win_length=160,
                                                            hop_length=80)


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        wavpath = self.dataframe.iloc[index]['filepath']
        transcription = self.dataframe.iloc[index]['transcription'].lower()
        label = self.vocab_model.encode_as_ids(transcription)
        label_length = len(label)

        waveform, sample_rate = torchaudio.load(wavpath)
        spectrogram = self.get_melspectrogram(waveform)
        spectrogram = np.log(spectrogram + 1e-14)
        return spectrogram, torch.tensor(label), torch.tensor(label_length)


def collate_fn(batch):
    (specs, labels, label_lengths) = zip(*batch)
    all_label_lengths = torch.tensor(label_lengths)
    all_labels = pad_sequence([torch.tensor(label) for label in labels], batch_first=True, padding_value=0)
    all_specs = []
    for spec in specs:
        spec = nn.ConstantPad1d(padding=(0, 200 - spec.size(-1)), value=0)(spec)
        all_specs.append(spec)

    all_specs = torch.stack(all_specs, dim=0)
    return all_specs, all_labels, all_label_lengths


if __name__ == '__main__':
    distribution = uniform.Uniform(0.9, 1.2)
    x = distribution.sample()
    print(x)