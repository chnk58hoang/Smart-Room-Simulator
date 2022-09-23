from torch.utils.data import Dataset
from torchaudio import transforms
from torch.distributions import uniform
from torch.nn.utils.rnn import pad_sequence
import torchaudio.functional as F
import torch
import torchaudio
import numpy as np


class SpeechDataset(Dataset):
    def __init__(self, dataframe, phase, vocab_model):
        super(SpeechDataset, self).__init__()
        self.dataframe = dataframe
        self.phase = phase
        self.vocab_model = vocab_model
        self.stretcher = transforms.TimeStretch(n_freq=128)
        self.time_mask = transforms.TimeMasking(time_mask_param=10)
        self.freq_mask = transforms.FrequencyMasking(freq_mask_param=27)
        self.pitch_shift = transforms.PitchShift(sample_rate=16000, n_steps=0)
        self.get_melspectrogram = transforms.MelSpectrogram(sample_rate=16000, n_mels=128, win_length=160,
                                                            hop_length=80)
        self.gain_distribution = uniform.Uniform(low=9, high=11)
        self.pitch_distribution = uniform.Uniform(low=-4, high=4)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        wavpath = self.dataframe.iloc[index]['filepath']
        transcription = self.dataframe.iloc[index]['transcription'].lower()
        label = self.vocab_model.encode_as_ids(transcription)
        label_length = len(label)

        waveform, sample_rate = torchaudio.load(wavpath)

        if self.phase == 'train' or self.phase == 'valid':
            gain_db = self.gain_distribution.sample()
            waveform = F.gain(waveform=waveform, gain_db=gain_db)

            n_steps = self.pitch_distribution.sample()
            self.pitch_shift.n_steps = n_steps
            waveform = self.pitch_shift(waveform)

            spectrogram = self.get_melspectrogram(waveform)
            spectrogram = np.log(spectrogram + 1e-14)
            spectrogram = self.time_mask(spectrogram)
            spectrogram = self.freq_mask(spectrogram)

        elif self.phase == 'test':
            gain_db = self.gain_distribution.sample()
            waveform = F.gain(waveform=waveform, gain_db=gain_db)
            spectrogram = self.get_melspectrogram(waveform)

        return spectrogram, torch.tensor(label), torch.tensor(label_length)


def collate_fn(batch):
    all_specs = []
    all_labels = []
    all_label_lengths = []

    for (spectrogram, label, label_length) in batch:
        all_specs.append(spectrogram.squeeze(0).transpose(0, 1))
        all_labels.append(label)
        all_label_lengths.append((label_length))

    all_specs = pad_sequence(all_specs, batch_first=True).transpose(1, 2)
    all_labels = pad_sequence(all_labels, batch_first=True)

    return all_specs, all_labels, torch.tensor(all_label_lengths)


if __name__ == '__main__':
    distribution = uniform.Uniform(0.9, 1.2)
    x = distribution.sample()
    print(x)
