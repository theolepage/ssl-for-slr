import math
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

from sslforslr.dataset.AudioAugmentation import AudioAugmentation
from sslforslr.dataset.utils import load_wav, extract_mfcc

def sample_frames(audio, frame_length):
    audio_length = audio.shape[1]
    assert audio_length >= 2 * frame_length, \
        "audio_length should >= 2 * frame_length"

    dist = audio_length - 2 * frame_length
    dist = np.random.randint(0, dist + 1)

    lower = frame_length + dist // 2
    upper = audio_length - (frame_length + dist // 2)
    pivot = np.random.randint(lower, upper + 1)

    frame1_from = pivot - dist // 2 - frame_length
    frame1_to = pivot - dist // 2
    frame1 = audio[:, frame1_from:frame1_to]

    frame2_from = pivot + dist // 2
    frame2_to = pivot + dist // 2 + frame_length
    frame2 = audio[:, frame2_from:frame2_to]

    return frame1, frame2

class KaldiDatasetGenerator(Sequence):
    def __init__(
        self,
        batch_size,
        frame_length,
        frame_split,
        files,
        indices,
        augment=None,
        spec_augment=False,
        extract_mfcc=False
    ):
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.frame_split = frame_split
        self.files = files
        self.indices = indices
        self.augment = augment
        self.spec_augment = spec_augment
        self.extract_mfcc = extract_mfcc

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def preprocess_data(self, data):
        assert data.ndim == 2 and data.shape[0] == 1 # (1, T)

        if self.augment: data = self.augment(data)        
        
        if self.extract_mfcc:
            data = extract_mfcc(data, self.spec_augment) # (1, T) -> (1, T, C)
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
        end   = i * self.batch_size + curr_batch_size
        indices = self.indices[start:end]

        if is_last_batch and remaining_samples != 0:
            resampled_indices = np.random.choice(
                self.indices[:start],
                size=self.batch_size - remaining_samples
            )
            indices = np.concatenate((indices, resampled_indices))
        assert len(indices) == self.batch_size

        X1, X2, y = [], [], []
        for index in indices:
            if self.frame_split:
                data = load_wav(
                    self.files[index],
                    frame_length=None,
                    min_length=2*self.frame_length
                ) # (1, T)
                frame1, frame2 = sample_frames(data, self.frame_length)
                X1.append(self.preprocess_data(frame1))
                X2.append(self.preprocess_data(frame2))
                y.append(0)
            else:
                data = load_wav(self.files[index], self.frame_length) # (1, T)
                data = self.preprocess_data(data)
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
        
        train_indices = np.arange(count)
        np.random.shuffle(train_indices)
        if self.config.val_ratio:
            train_indices, val_indices = train_test_split(
                train_indices,
                test_size=self.config.val_ratio,
                random_state=0
            )

        train_gen = KaldiDatasetGenerator(
            batch_size,
            self.config.frame_length,
            self.config.frame_split,
            self.files,
            train_indices,
            self.augment,
            self.config.spec_augment,
            self.config.extract_mfcc
        )

        val_gen = None
        if self.config.val_ratio:
            val_gen = KaldiDatasetGenerator(
                batch_size,
                self.config.frame_length,
                self.config.frame_split,
                self.files,
                val_indices,
                self.augment,
                self.config.spec_augment,
                self.config.extract_mfcc
            )

        return train_gen, val_gen
