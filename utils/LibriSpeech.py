import numpy as np
import math
import time
from tensorflow.keras.utils import Sequence
import soundfile as sf
import glob
import h5py

class LibriSpeechGenerator(Sequence):
  
    def __init__(self,
                 path,
                 name,
                 batch_size,
                 frame_length,
                 pick_random=False):
        cache = h5py.File(path, 'r')
        self.X = cache[name + '_x']
        self.y = cache[name + '_y']
        self.indices = np.arange(len(self.y))

        self.batch_size = batch_size
        self.frame_length = frame_length
        self.pick_random = pick_random

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)
  
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
        remaining_samples = len(self.y) % self.batch_size
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

class LibriSpeechLoader:

    def __init__(self, seed, config):
        np.random.seed(seed)

        self.train_paths = config['train_paths']
        self.val_paths = config['val_paths']
        self.test_paths = config['test_paths']
        self.frames = config['frames']
        self.limits = config['limits'] if 'limits' in config else {}

    def get_frames(self, filename):
        length = self.frames['length']
        with sf.SoundFile(filename, 'r') as f:
            signal_length = f.frames

        assert signal_length > length

        if self.frames['pick'] == 'random':
            return [-1]

        elif self.frames['pick'] == 'sequence':
            stride = self.frames['stride']
            count = self.frames['count']

            # Determine frames indexes
            num_frames = int(np.floor((signal_length - length) / stride))
            num_frames = min(num_frames, count)
            indexes = np.arange(0, num_frames * stride, stride)

            return indexes

        raise Exception('LibriSpeech: frames picking method not handled')

    def scan_directories(self, paths):
        nb_speakers = 0
        filenames = []
        speakers = []

        limit_speakers = -1
        limit_utterances = -1
        if 'utterances_per_speaker' in self.limits:
            limit_utterances = self.limits['utterances_per_speaker']
        if 'speakers' in self.limits:
            limit_speakers = self.limits['speakers']

        # Scan datasets
        for dataset_id in range(len(paths)):
            speaker_dirs = glob.glob(paths[dataset_id])

            if (len(speaker_dirs) == 0):
                raise Exception('LibriSpeech: no data found in {}'.format(paths[dataset_id]))

            for speaker_id in range(len(speaker_dirs)):
                if nb_speakers == limit_speakers:
                    break

                nb_speakers += 1
                nb_speaker_utterances = 0
                chapter_dirs = glob.glob(speaker_dirs[speaker_id] + '/*')
                
                for chapter_id in range(len(chapter_dirs)):
                    files = glob.glob(chapter_dirs[chapter_id] + '/*.flac')

                    for file in files:
                        if nb_speaker_utterances == limit_utterances:
                            break

                        for frame in self.get_frames(file):
                            filenames.append([file, frame])
                            speakers.append(speaker_id)

                        nb_speaker_utterances += 1

        return filenames, speakers

    def create_cache(self, name, cache, paths):
        start = time.time()
        filenames, speakers = self.scan_directories(paths)
        nb_samples = len(filenames)
        frame_length = self.frames['length']

        # Create h5py dataset
        if self.frames['pick'] == 'random':
            # Picking frames randomly online implies storing
            # frames of different length
            dt = h5py.vlen_dtype(np.float64)
            X = cache.create_dataset(name + '_x', (nb_samples,), dtype=dt)
        else:
            X = cache.create_dataset(name + '_x', (nb_samples, frame_length))
        y = cache.create_dataset(name + '_y', (nb_samples))

        if nb_samples == 0:
            return

        for i in range(nb_samples):
            filename, frame = filenames[i]
            speaker = speakers[i]
            data, fs = sf.read(filename)

            if self.frames['pick'] == 'sequence':
                data = data[frame:frame+frame_length]

            X[i] = data
            y[i] = speaker
            
            # Log progress
            progress = math.ceil(100 * (i + 1) / nb_samples)
            if progress % 5 == 0:
                progress_text = '{}% {} files'.format(progress, name)
                print('LibriSpeech: caching ' + progress_text, end='\r')

        end = time.time()
        print()
        print('LibriSpeech: done in {}s'.format(end - start))

    def load(self, batch_size, checkpoint_dir):
        # Create cache during first use
        path = checkpoint_dir + '/librispeech.h5'
        cache = h5py.File(path, 'a')
        if len(cache) == 0:
            self.create_cache('train', cache, self.train_paths)
            self.create_cache('val', cache, self.val_paths)
            self.create_cache('test', cache, self.test_paths)        
        cache.close()

        # Create Keras generators
        frame_length = self.frames['length']
        pick_random = (self.frames['pick'] == 'random')
        train = LibriSpeechGenerator(path, 'train',
                                     batch_size, frame_length, pick_random)
        val = LibriSpeechGenerator(path, 'val',
                                   batch_size, frame_length, pick_random)
        test = LibriSpeechGenerator(path, 'test',
                                    batch_size, frame_length, pick_random)

        return train, val, test