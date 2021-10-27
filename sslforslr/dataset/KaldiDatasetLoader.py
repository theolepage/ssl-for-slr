import math
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

from sslforslr.dataset.AudioAugmentation import AudioAugmentation
from sslforslr.dataset.utils import load_wav, extract_mfcc

class KaldiDatasetGenerator(Sequence):
    def __init__(
        self,
        batch_size,
        frame_length,
        frame_split,
        files,
        indices,
        augment=None,
        extract_mfcc=False
    ):
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.frame_split = frame_split
        self.files = files
        self.indices = indices
        self.augment = augment
        self.extract_mfcc = extract_mfcc

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def preprocess_data(self, data):
        assert data.ndim == 2 and data.shape[0] == 1 # (1, T)

        if self.augment: data = self.augment(data)        
        
        if self.extract_mfcc:
            data = extract_mfcc(data) # (1, T) -> (1, T, C)
            data = data.squeeze(axis=0) # (1, T, C) -> (T, C)
        else:
            data = data.T # (1, T) -> (T, 1)
        
        return data

    def __getitem__(self, i):
        # Last batch may have fewer samples
        curr_batch_size = self.batch_size
        is_last_batch = (i == self.__len__() - 1)
        remaining_samples = len(self.indices) % self.batch_size
        if is_last_batch and remaining_samples != 0:
            curr_batch_size = remaining_samples

        start = i * self.batch_size
        end   = i * self.batch_size + remaining_samples
        indices = self.indices[start:end]

        if is_last_batch and remaining_samples != 0:
            indices += np.random.choice(
                self.indices[:start],
                size=self.batch_size - remaining_samples
            )

        X1, X2, y = [], [], []
        for index in indices:
            data = load_wav(self.files[index], self.frame_length) # (1, T)
            # data = self.preprocess_data(data)
            
            if self.frame_split:
                pivot = len(data) // 2
                X1.append(data[:, :pivot])
                X2.append(data[:, pivot:])
                y.append(0)
                continue
            
            X1.append(data)
            y.append(0)

        if self.frame_split:
            return np.array(X1), np.array(X2), np.array(y)
        return np.array(X1), np.array(y)

    def on_epoch_end(self):
        # Randomize samples manually after each epoch
        np.random.shuffle(self.indices)

class KaldiDatasetLoader:
    def __init__(self, config):
        self.config = config

        # Create augmentation module
        self.augment = None
        if self.config.augment:
            self.augment = AudioAugmentation(self.config.augment)

        # Create a list of audio paths
        self.files = []
        for line in open(self.config.train):
            _, file = line.rstrip().split()
            self.files.append(file)

    def get_input_shape(self):
        if self.config.extract_mfcc:
            return (self.config.frame_length // 160, 40)
        return (self.config.frame_length, 1)

    def load(self, batch_size):
        count = self.config.max_samples if self.config.max_samples else len(self.files)
        indices = train_test_split(np.arange(count),
                                   test_size=self.config.val_ratio)

        train_gen = KaldiDatasetGenerator(
            batch_size,
            self.config.frame_length,
            self.config.frame_split,
            self.files,
            indices[0],
            self.augment,
            self.config.extract_mfcc
        )

        val_gen = KaldiDatasetGenerator(
            batch_size,
            self.config.frame_length,
            self.config.frame_split,
            self.files,
            indices[1],
            self.augment,
            self.config.extract_mfcc
        )

        return train_gen, val_gen
