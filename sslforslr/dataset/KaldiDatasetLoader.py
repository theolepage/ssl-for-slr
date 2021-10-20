import numpy as np
from tensorflow.keras.utils import Sequence
import soundfile as sf
from sklearn.model_selection import train_test_split

import torch
import torchaudio

class KaldiDatasetGenerator(Sequence):
    def __init__(self, batch_size, frame_length, files, indices):
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.files = files
        self.indices = indices

    def __len__(self):
        # FIXME: Handle last batch having < self.batch_size elements
        return len(self.indices) // self.batch_size

    def __getitem__(self, i):
        X, y = [], []

        for j in range(self.batch_size):
            index = self.indices[i * self.batch_size + j]

            sample, sr = sf.read(self.files[index])
            data = sample.reshape((len(sample), 1))

            assert len(data) >= self.frame_length
            offset = np.random.randint(0, len(data) - self.frame_length + 1)
            data = data[offset:offset+self.frame_length]

            X.append(data)
            y.append(0)

        return np.array(X), np.array(y)

class KaldiDatasetLoader:
    def __init__(self, seed, config):
        self.config = config

        self.files = []
        for line in open(self.config['train']):
            _, file = line.rstrip().split()
            self.files.append(file)

    def load(self, batch_size):
        val_ratio = self.config.get('val_ratio', 0.1)

        # Create list of indices to shuffle easily during training
        max_samples = self.config.get('max_samples', None)
        nb_samples = max_samples if max_samples else len(self.files)
        indices = np.arange(nb_samples)
        indices_train, indices_val = train_test_split(indices, test_size=val_ratio)

        train = KaldiDatasetGenerator(batch_size,
                                      self.config['frame_length'],
                                      self.files,
                                      indices_train)

        val = KaldiDatasetGenerator(batch_size,
                                    self.config['frame_length'],
                                    self.files,
                                    indices_val)
        
        return (train, val)
