import numpy as np
import math
import time
from sklearn.model_selection import train_test_split
import soundfile as sf
import glob
import h5py

from .AudioDatasetGenerator import AudioDatasetGenerator

def get_frames(filename, frames_config):
    length = frames_config['length']
    with sf.SoundFile(filename, 'r') as f:
        signal_length = f.frames

    if signal_length < length:
        return []

    if frames_config['pick'] == 'random':
        return []

    elif frames_config['pick'] == 'sequence':
        stride = frames_config['stride']
        count = frames_config['count']

        # Determine frames indexes
        num_frames = 1 + math.floor((signal_length - length) / stride)
        num_frames = min(num_frames, count)
        indexes = np.arange(0, num_frames * stride, stride)

        return indexes

    raise Exception('AudioDatasetLoader: frames picking method not handled')

def scan_librispeech(paths, limits_config, frames_config):
    nb_speakers = 0
    filenames = []
    speakers = []

    limit_speakers = limits_config.get('speakers', -1)
    limit_utterances = limits_config.get('utterances_per_speaker', -1)
    total_ratio = limits_config.get('total_ratio', 1.0)

    # Scan datasets
    for dataset_id in range(len(paths)):
        speaker_dirs = glob.glob(paths[dataset_id])

        if (len(speaker_dirs) == 0):
            raise Exception('AudioDatasetLoader: no data found in %s' % paths[dataset_id])

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

                    frames = get_frames(file, frames_config)
                    for frame in frames:
                        filenames.append([file, frame])
                        speakers.append(speaker_id)

                    if len(frames) != 0:
                        nb_speaker_utterances += 1

    # Keep only a specific ratio of all samples
    split_idx = int(total_ratio * len(filenames))
    filenames = filenames[0:split_idx]
    speakers = speakers[0:split_idx]

    return filenames, speakers

def scan_voxlingua107(paths, limits_config, frames_config):
    nb_languages = 0
    filenames = []
    languages = []

    limit_languages = limits_config.get('languages', -1)
    limit_utterances = limits_config.get('utterances_per_language', -1)
    total_ratio = limits_config.get('total_ratio', 1.0)

    for dataset_id in range(len(paths)):
        language_dirs = glob.glob(paths[dataset_id])

        if (len(language_dirs) == 0):
            raise Exception('AudioDatasetLoader: no data found in %s' % paths[dataset_id])

        for language_id in range(len(language_dirs)):
            if nb_languages == limit_languages:
                break

            nb_language_utterances = 0
            nb_languages += 1
            files = glob.glob(language_dirs[language_id] + '/*.wav')

            for file in files:
                if nb_language_utterances == limit_utterances:
                        break

                frames = get_frames(file, frames_config)
                for frame in frames:
                    filenames.append([file, frame])
                    languages.append(language_id)

                if len(frames) != 0:
                    nb_language_utterances += 1


    # Keep only a specific ratio of all samples
    split_idx = int(total_ratio * len(filenames))
    filenames = filenames[0:split_idx]
    languages = languages[0:split_idx]

    return filenames, languages

class AudioDatasetLoader:

    def __init__(self, seed, config):
        np.random.seed(seed)

        self.type = config.get('type', 'LibriSpeech')
        self.train_paths = config.get('train_paths', [])
        self.val_paths = config.get('val_paths', [])
        self.test_paths = config.get('test_paths', [])
        self.val_ratio = config.get('val_ratio', None)
        self.test_ratio = config.get('test_ratio', None)
        self.frames = config['frames']
        self.limits = config.get('limits', {})

    def create_cache(self, name, cache, paths):
        start = time.time()
        print('AudioDatasetLoader: creating dataset...')

        if self.type == 'LibriSpeech':
            filenames, labels = scan_librispeech(paths, self.limits, self.frames)
        elif self.type == 'VoxLingua107':
            filenames, labels = scan_voxlingua107(paths, self.limits, self.frames)

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
            label = labels[i]
            data, fs = sf.read(filename)

            if self.frames['pick'] == 'sequence':
                data = data[frame:frame+frame_length]

            # Normalize input signal
            max_value = np.max(np.abs(data))
            data = data / max_value if max_value != 0 else np.zeros_like(data)

            X[i] = data
            y[i] = label
            
            # Log progress
            progress = math.ceil(100 * (i + 1) / nb_samples)
            if progress % 5 == 0:
                progress_text = '{}% {} files'.format(progress, name)
                print('AudioDatasetLoader: caching ' + progress_text, end='\r')

        end = time.time()
        print()
        print('AudioDatasetLoader: done in {}s'.format(end - start))

        return len(np.unique(labels))

    def create_gens(self, cache_path, batch_size):
        random = (self.frames['pick'] == 'random')
        frame_length = self.frames['length']

        train = AudioDatasetGenerator(cache_path,
                                     batch_size, 
                                     frame_length,
                                     subset='train',
                                     pick_random=random)

        val = AudioDatasetGenerator(cache_path,
                                   batch_size,
                                   frame_length,
                                   subset='val',
                                   pick_random=random)

        test = AudioDatasetGenerator(cache_path,
                                    batch_size,
                                    frame_length,
                                    subset='test',
                                    pick_random=random)

        return [train, val, test]

    def create_gens_with_ratio(self, cache_path, nb_train_samples, batch_size):
        indices = np.arange(nb_train_samples)
        indices_train = []
        indices_val = []
        indices_test = []

        if self.val_ratio > 0.0 and self.test_ratio > 0.0:
            ratio = self.val_ratio + self.test_ratio
            indices_train, indices_test = train_test_split(indices, test_size=ratio)
            ratio = self.test_ratio / (self.test_ratio + self.val_ratio)
            indices_val, indices_test = train_test_split(indices_test, test_size=ratio)
        elif self.val_ratio > 0.0:
            indices_train, indices_val = train_test_split(indices, test_size=self.val_ratio)
        else: # self.test_ratio > 0.0
            indices_train, indices_test = train_test_split(indices, test_size=self.test_ratio)

        random = (self.frames['pick'] == 'random')
        frame_length = self.frames['length']
        
        train = AudioDatasetGenerator(cache_path,
                                      batch_size, 
                                      frame_length,
                                      indices=indices_train,
                                      pick_random=random)

        val = AudioDatasetGenerator(cache_path,
                                    batch_size,
                                    frame_length,
                                    indices=indices_val,
                                    pick_random=random)

        test = AudioDatasetGenerator(cache_path,
                                     batch_size,
                                     frame_length,
                                     indices=indices_test,
                                     pick_random=random)

        return [train, val, test]

    def load(self, batch_size, checkpoint_dir):
        # Create cache during first use
        cache_path = checkpoint_dir + '/AudioDatasetLoader_cache.h5'
        cache = h5py.File(cache_path, 'a')
        if len(cache) == 0:
            nb_categories  = self.create_cache('train', cache, self.train_paths)
            nb_categories += self.create_cache('val', cache, self.val_paths)
            nb_categories += self.create_cache('test', cache, self.test_paths)
            cache.attrs.create('nb_categories', nb_categories)
        
        nb_categories = cache.attrs['nb_categories']
        nb_train_samples = len(cache['train_y'])
        cache.close()

        # Create Keras generators
        if self.val_ratio is not None and self.test_ratio is not None:
            return self.create_gens_with_ratio(cache_path,
                                               nb_train_samples,
                                               batch_size), nb_categories
        elif self.val_ratio is not None or self.test_ratio is not None:
            raise Exception('AudioDatasetLoader: you must specify both val_ratio and test_ratio')

        return self.create_gens(cache_path,batch_size), nb_categories