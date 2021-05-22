import numpy as np
import math
import time
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import soundfile as sf
import glob
import h5py

class LibriSpeechGenerator(Sequence):
  
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

class LibriSpeechLoader:

    def __init__(self, seed, config):
        np.random.seed(seed)

        self.train_paths = config.get('train_paths', [])
        self.val_paths = config.get('val_paths', [])
        self.test_paths = config.get('test_paths', [])
        self.val_ratio = config.get('val_ratio', 0.0)
        self.test_ratio = config.get('test_ratio', 0.0)
        self.frames = config['frames']
        self.limits = config.get('limits', {})

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
            num_frames = 1 + math.floor((signal_length - length) / stride)
            num_frames = min(num_frames, count)
            indexes = np.arange(0, num_frames * stride, stride)

            return indexes

        raise Exception('LibriSpeech: frames picking method not handled')

    def scan_directories(self, paths):
        nb_speakers = 0
        filenames = []
        speakers = []

        limit_speakers = self.limits.get('speakers', -1)
        limit_utterances = self.limits.get('utterances_per_speaker', -1)
        total_ratio = self.limits.get('total_ratio', 1.0)

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

        # Keep only a specific ratio of all samples
        split_idx = int(total_ratio * len(filenames))
        filenames = filenames[0:split_idx]
        speakers = speakers[0:split_idx]

        return filenames, speakers

    def create_cache(self, name, cache, paths):
        start = time.time()
        print('LibriSpeech: creating dataset...')

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
            return 0

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

        return len(np.unique(speakers))

    def create_gens(self, cache_path, batch_size):
        random = (self.frames['pick'] == 'random')
        frame_length = self.frames['length']

        train = LibriSpeechGenerator(cache_path,
                                     batch_size, 
                                     frame_length,
                                     subset='train',
                                     pick_random=random)

        val = LibriSpeechGenerator(cache_path,
                                   batch_size,
                                   frame_length,
                                   subset='val',
                                   pick_random=random)

        test = LibriSpeechGenerator(cache_path,
                                    batch_size,
                                    frame_length,
                                    subset='test',
                                    pick_random=random)

        return [train, val, test]

    def create_gens_with_ratio(self, cache_path, nb_train_samples, batch_size):
        random = (self.frames['pick'] == 'random')
        frame_length = self.frames['length']
        indices = np.arange(nb_train_samples)

        ratio = self.val_ratio + self.test_ratio
        indices_train, indices_test = train_test_split(indices,
                                                       test_size=ratio)
        ratio = self.test_ratio / (self.test_ratio + self.val_ratio)
        indices_val, indices_test = train_test_split(indices_test,
                                                     test_size=ratio)
        train = LibriSpeechGenerator(cache_path,
                                     batch_size, 
                                     frame_length,
                                     indices=indices_train,
                                     pick_random=random)

        val = LibriSpeechGenerator(cache_path,
                                   batch_size,
                                   frame_length,
                                   indices=indices_val,
                                   pick_random=random)

        test = LibriSpeechGenerator(cache_path,
                                    batch_size,
                                    frame_length,
                                    indices=indices_test,
                                    pick_random=random)

        return [train, val, test]

    def load(self, batch_size, checkpoint_dir):
        # Create cache during first use
        cache_path = checkpoint_dir + '/librispeech.h5'
        cache = h5py.File(cache_path, 'a')
        if len(cache) == 0:
            nb_spk_train = self.create_cache('train', cache, self.train_paths)
            nb_spk_val = self.create_cache('val', cache, self.val_paths)
            nb_spk_test = self.create_cache('test', cache, self.test_paths)

            nb_speakers = nb_spk_train + nb_spk_val + nb_spk_test
            cache.attrs.create('nb_speakers', nb_speakers)
        
        nb_speakers = cache.attrs['nb_speakers']
        nb_train_samples = len(cache['train_y'])
        cache.close()

        # Create Keras generators
        if self.val_ratio > 0.0 and self.test_ratio > 0.0:
            return self.create_gens_with_ratio(cache_path,
                                               nb_train_samples,
                                               batch_size), nb_speakers

        return self.create_gens(cache_path,batch_size), nb_speakers