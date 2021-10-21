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
        # FIXME: Handle last batch having < self.batch_size elements
        return len(self.indices) // self.batch_size

    def preprocess_data(self, data):
        if self.augment:      data = augment(data)
        if self.extract_mfcc: data = extract_mfcc(data)
        return data

    def __getitem__(self, i):
        X1, X2, y = [], [], []

        for j in range(self.batch_size):
            index = self.indices[i * self.batch_size + j]

            data = load_wav(self.files[index], self.frame_length)

            if self.frame_split:
                pivot = self.frame_length // 2
                X1.append(self.preprocess_data(data[:pivot]))
                X2.append(self.preprocess_data(data[pivot:]))
                y.append(0)
                continue
            
            X1.append(self.preprocess_data(data))
            y.append(0)

        if self.frame_split:
            return np.array(X1), np.array(X2), np.array(y)
        return np.array(X1), np.array(y)

class KaldiDatasetLoader:
    def __init__(self, seed, config):
        self.config = config

        self.files = []
        for line in open(self.config['train']):
            _, file = line.rstrip().split()
            self.files.append(file)

    def load(self, batch_size):
        frame_length = self.config['frame_length']
        val_ratio = self.config.get('val_ratio', 0.1)
        extract_mfcc = self.config.get('extract_mfcc', False)
        frame_split = self.config.get('frame_split', False)

        augment = None
        augment_config = self.config.get('augment', None)
        if augment_config and augment_config.get('enabled', True):
            augment = AudioAugmentation(augment_config)

        # Create list of indices to shuffle easily during training
        max_samples = self.config.get('max_samples', None)
        nb_samples = max_samples if max_samples else len(self.files)
        indices = np.arange(nb_samples)
        indices_train, indices_val = train_test_split(indices, test_size=val_ratio)

        train = KaldiDatasetGenerator(batch_size,
                                      frame_length,
                                      frame_split,
                                      self.files,
                                      indices_train,
                                      augment,
                                      extract_mfcc)

        val = KaldiDatasetGenerator(batch_size,
                                    frame_length,
                                    frame_split,
                                    self.files,
                                    indices_val,
                                    augment,
                                    extract_mfcc)

        return (train, val)
