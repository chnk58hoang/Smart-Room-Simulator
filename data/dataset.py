from torch.utils.data import Dataset
from torchaudio import transforms
from torch.distributions import uniform
from torch.nn.utils.rnn import pad_sequence
import torchaudio.functional as F
import torch
import torch.nn as nn
import torchaudio


class SpeechDataset(Dataset):
    def __init__(self, dataframe, phase, vocab_model):
        super(SpeechDataset, self).__init__()
        self.dataframe = dataframe
        self.phase = phase
        self.vocab_model = vocab_model

    def __len__(self):
        return len(self.dataframe)

    def time_stretch(self, spectrogram, low, high):
        distribution = uniform.Uniform(low, high)
        overriding_rate = distribution.sample()
        stretcher = transforms.TimeStretch(n_freq=128)
        spectrogram = stretcher(spectrogram, overriding_rate)

        return spectrogram

    def gain(self, waveform, low, high):
        distribution = uniform.Uniform(low, high)
        gain_db = distribution.sample()
        waveform = F.gain(waveform, gain_db=gain_db)
        return waveform

    def time_mask(self, spectrogram, time_mask_param):
        time_masker = transforms.TimeMasking(time_mask_param)
        spectrogram = time_masker(spectrogram)
        return spectrogram

    def freq_mask(self, spectrogram, freq_mask_param):
        freq_masker = transforms.FrequencyMasking(freq_mask_param)
        spectrogram = freq_masker(spectrogram)
        return spectrogram

    def pitch_shift(self, waveform, sample_rate, low, high):
        distribution = uniform.Uniform(low, high)
        semi_tones = distribution.sample()
        shifter = transforms.PitchShift(sample_rate, semi_tones)
        waveform = shifter(waveform)
        return waveform

    def get_melspectrogram(self, waveform, sample_rate, n_mels):
        transform = transforms.MelSpectrogram(sample_rate, n_mels=n_mels)
        mel_spectrogram = transform(waveform)

        return mel_spectrogram

    def __getitem__(self, index):
        wavpath = self.dataframe.iloc[index]['filepath']
        transcription = self.dataframe.iloc[index]['transcription'].lower()
        label = self.vocab_model.encode_as_ids(transcription)
        label_length = len(label)

        waveform, sample_rate = torchaudio.load(wavpath)

        if self.phase == 'train' or self.phase == 'valid':
            waveform = self.gain(waveform, low=8, high=11)
            waveform = self.pitch_shift(waveform, sample_rate, low=-4, high=4)

            spectrogram = self.get_melspectrogram(waveform, sample_rate, n_mels=128)
            spectrogram = self.time_stretch(spectrogram, low=0.9, high=1.1)
            spectrogram = self.time_mask(spectrogram, time_mask_param=10)
            spectrogram = self.freq_mask(spectrogram, freq_mask_param=27)

        elif self.phase == 'test':
            waveform = self.gain(waveform, low=8, high=11)
            spectrogram = self.get_melspectrogram(waveform, sample_rate, n_mels=128)

        return spectrogram, label, label_length


def collate_fn(batch):
    (specs, labels, label_lengths) = zip(*batch)
    all_label_lengths = torch.tensor(label_lengths)
    all_labels = pad_sequence([torch.tensor(label) for label in labels], batch_first=True, padding_value=0)
    all_specs = []
    for spec in specs:
        spec = nn.ConstantPad1d(padding=(0, 161 - spec.size(-1)), value=0)(spec)
        all_specs.append(spec)

    all_specs = torch.stack(all_specs, dim=0)
    return all_specs, all_labels, all_label_lengths


if __name__ == '__main__':
    distribution = uniform.Uniform(0.9, 1.2)
    x = distribution.sample()
    print(x)
