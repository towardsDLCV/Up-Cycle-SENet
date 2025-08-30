import numpy as np
import matplotlib.pyplot as plt
import librosa
import random
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset


class SpeechDataset(Dataset):

    def __init__(self, args, noisy_files, clean_files, unpaired=True):
        super(SpeechDataset, self).__init__()
        self.args = args
        self.unpaired = unpaired

        self.noisy_files = sorted(noisy_files)  # Noisy List
        self.clean_files = sorted(clean_files)  # Clean List

    def __len__(self):
        return max(len(self.noisy_files), len(self.clean_files))

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def load_sample(self, file):
        waveform, sr = torchaudio.load(file)
        return waveform

    def padding(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[0]

        output = np.zeros((32000), dtype='float32')
        output[-current_len:] = waveform[:32000]
        output = torch.from_numpy(output)
        return output

    def __getitem__(self, idx):
        x_noisy = self.load_sample(self.noisy_files[idx % len(self.noisy_files)])
        if self.unpaired:
            x_clean = self.load_sample(self.clean_files[random.randint(0, len(self.clean_files) - 1)])
        else:
            x_clean = self.load_sample(self.clean_files[idx % len(self.clean_files)])

        noisy_length = x_noisy.size(-1)
        clean_length = x_clean.size(-1)

        x_clean.squeeze_(0)
        x_noisy.squeeze_(0)

        # Train
        if self.unpaired:
            if noisy_length - 32000 - 1 > 0:
                n_start = torch.randint(0, noisy_length - 32000 - 1, (1, ))
                n_end = n_start + 32000
                x_noisy = x_noisy[n_start:n_end]
            else:
                x_noisy = self.padding(x_noisy)

            if clean_length - 32000 - 1 > 0:
                c_start = torch.randint(0, clean_length - 32000 - 1, (1, ))
                c_end = c_start + 32000
                x_clean = x_clean[c_start:c_end]
            else:
                x_clean = self.padding(x_clean)

            return x_noisy, x_clean
        # Test
        else:
            if noisy_length - 32000 - 1 > 0:
                n_start = torch.randint(0, noisy_length - 32000 - 1, (1, ))
                n_end = n_start + 32000
                x_noisy = x_noisy[n_start:n_end]
                x_clean = x_clean[n_start:n_end]
            else:
                x_noisy = self.padding(x_noisy)
                x_clean = self.padding(x_clean)

            return x_noisy, x_clean, clean_length


def display_spectrogram(x, title):
    plt.figure(figsize=(15, 10))
    plt.pcolormesh(x[0][0], cmap='hot')
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    plt.show()
