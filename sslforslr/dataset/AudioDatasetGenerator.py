import numpy as np
import math
import h5py
from tensorflow.keras.utils import Sequence

class AudioDatasetGenerator(Sequence):
    '''
    Keras generator to use with an existing cache file
    created with AudioDatasetLoader.
    '''

    def __init__(self,
                 cache_path,
                 batch_size,
                 frame_length,
                 subset='train',
                 indices=None,
                 pick_random=False):
        cache = h5py.File(cache_path, 'r')
        self.X = cache[subset + '_x']
        self.y = cache[subset + '_y']

        self.indices = np.arange(len(self.y)) if indices is None else indices
        
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.pick_random = pick_random

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)
  
    def get_random_frame(self, curr_batch_size, signals):
        X_batch = np.empty((curr_batch_size, self.frame_length, 1))

        for j in range(curr_batch_size):
            signal = signals[j]
            idx = np.random.randint(len(signal) - self.frame_length + 1)
            X_batch[j, :, 0] = signal[idx:idx+self.frame_length]

        return X_batch

    def __getitem__(self, i):
        curr_batch_size = self.batch_size

        # Last batch may have fewer samples
        is_last_batch = i == self.__len__() - 1
        remaining_samples = len(self.indices) % self.batch_size
        if is_last_batch and remaining_samples != 0:
            curr_batch_size = remaining_samples
        
        # Shuffling a h5py dataset directly is not possible and indices
        # must be in increasing order. 
        idx = self.indices[i*self.batch_size:i*self.batch_size+curr_batch_size]
        idx = np.sort(idx)

        X_batch = self.X[idx]
        y_batch = self.y[idx]

        if self.pick_random:
            return self.get_random_frame(curr_batch_size, X_batch), y_batch

        # Expected output shape is (batch_size, frame_length, 1)
        X_batch = np.expand_dims(X_batch, axis=-1)
        return X_batch, y_batch

    def on_epoch_end(self):
        # Randomize samples manually after each epoch
        np.random.shuffle(self.indices)