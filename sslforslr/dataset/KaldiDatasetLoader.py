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

        if self.augment:      data = augment(data)        
        
        if self.extract_mfcc: data = extract_mfcc(data) # (1, T) -> (1, T, C)
        data = data.squeeze(axis=0) # (1, T, C) -> (T, C)
        
        return data

    def __getitem__(self, i):
        curr_batch_size = self.batch_size

        # Last batch may have fewer samples
        is_last_batch = (i == self.__len__() - 1)
        remaining_samples = len(self.indices) % self.batch_size
        if is_last_batch and remaining_samples != 0:
            curr_batch_size = remaining_samples

        X1, X2, y = [], [], []
        for j in range(curr_batch_size):
            index = self.indices[i * self.batch_size + j]

            data = load_wav(self.files[index], self.frame_length) # (N, T)
            data = self.preprocess_data(data)
            
            if self.frame_split:
                pivot = len(data) // 2
                X1.append(data[:pivot])
                X2.append(data[pivot:])
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

        self.frame_length = self.config['frame_length']
        self.val_ratio = self.config.get('val_ratio', 0.1)
        self.extract_mfcc = self.config.get('extract_mfcc', False)
        self.frame_split = self.config.get('frame_split', False)
        self.max_samples = self.config.get('max_samples', None)

        # Create augmentation module
        self.augment = None
        augment_config = self.config.get('augment', None)
        if augment_config and augment_config.get('enabled', True):
            self.augment = AudioAugmentation(augment_config)

        # Create a list of audio paths
        self.files = []
        for line in open(self.config['train']):
            _, file = line.rstrip().split()
            self.files.append(file)

    def get_input_shape(self):
        if self.extract_mfcc:
            return (self.frame_length // 160, 40)
        return (self.frame_length, 1)

    def load(self, batch_size):
        count = self.max_samples if self.max_samples else len(self.files)
        indices = train_test_split(np.arange(count), test_size=self.val_ratio)

        train_gen = KaldiDatasetGenerator(
            batch_size,
            self.frame_length,
            self.frame_split,
            self.files,
            indices[0],
            self.augment,
            self.extract_mfcc
        )

        val_gen = KaldiDatasetGenerator(
            batch_size,
            self.frame_length,
            self.frame_split,
            self.files,
            indices[1],
            self.augment,
            self.extract_mfcc
        )

        return train_gen, val_gen
