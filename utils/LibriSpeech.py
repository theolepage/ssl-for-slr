from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import soundfile as sf
import glob
import numpy as np
from pathlib import Path

class LibriSpeechGenerator(Sequence):
  
    def __init__(self, X, y, batch_size, frame_length):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.frame_length = frame_length

    def __len__(self):
        nb_batches = len(self.y) / float(self.batch_size)
        return np.ceil(nb_batches).astype(np.int)
  
    def __getitem__(self, batch_id):
        X_batch = np.zeros((self.batch_size, self.frame_length, 1))
        y_batch = np.zeros(self.batch_size)

        for i in range(self.batch_size):
            id = batch_id * self.batch_size + i
            if id >= len(self.y):
                id = np.random.randint(0, len(self.y))

            path, frame = self.X[id]

            #FIX: when loading frame indices from file, all values are string
            frame = int(frame)

            signal, fs = sf.read(path)

            X_batch[i, :, 0] = signal[frame:frame+self.frame_length]
            y_batch[i] = self.y[id]

        return X_batch, y_batch

class LibriSpeechLoader:

    def __init__(self, seed, config, checkpoint_dir):
        self.seed = seed
        self.checkpoint_path = checkpoint_dir + '/librispeech_data.tmp'
        self.path = config['path']
        self.frame_length = config['frame_length']
        self.frame_stride = config['frame_stride']
        self.max_frames_per_utterance = config['max_frames_per_utterance']
        self.max_speakers = config['nb_speakers']
        self.max_utterances_per_speaker = config['max_utterances_per_speaker']
        self.val_ratio = config['val_ratio']
        self.test_ratio = config['test_ratio']

    def get_frames_indices(self, filename):
        signal, fs = sf.read(filename)

        # Determine number of frames
        signal_length = len(signal)
        assert signal_length > self.frame_length
        num_frames = int(np.floor((signal_length - self.frame_length) / self.frame_stride))

        # Limit the number of frames
        num_frames = min(num_frames, self.max_frames_per_utterance)

        return np.arange(0, num_frames * self.frame_stride, self.frame_stride)

    def create_frames_list(self):
        X = []
        y = []

        files = glob.glob(self.path)
        if (len(files) == 0):
            raise Exception('LibriSpeech: no data files found.')

        for speaker_id in range(min(self.max_speakers, len(files))):
            speaker_files = glob.glob(files[speaker_id] + '/*')

            nb_utterances_for_speaker = 0

            for sentence_id in range(len(speaker_files)):
                sentence_files = glob.glob(speaker_files[sentence_id] + '/*.flac')

                for utterance_id in range(len(sentence_files)):
                    if nb_utterances_for_speaker >= self.max_utterances_per_speaker:
                        break

                    filename = sentence_files[utterance_id]
                    frames = self.get_frames_indices(filename)

                    for frame in frames:
                        X.append([filename, frame])
                        y.append(speaker_id)
                    
                    nb_utterances_for_speaker += 1

            print('LibriSpeech: loaded {}/{} speakers'.format(speaker_id, min(self.max_speakers, len(files))), end='\r')

        print()
        print('LibriSpeech: done')

        return X, y

    def load(self, batch_size):
        # Load pre-existing frames list
        if (Path(self.checkpoint_path).exists()):
            with open(self.checkpoint_path, 'rb') as file:
                X = np.load(file)
                y = np.load(file)
        else:
            X, y = self.create_frames_list()
            with open(self.checkpoint_path, 'wb') as file:
                np.save(file, X)
                np.save(file, y)

        # Split in train, val and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.val_ratio + self.test_ratio, random_state=self.seed)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=self.test_ratio / (self.test_ratio + self.val_ratio), random_state=self.seed)

        # Create Keras generators
        train_gen = LibriSpeechGenerator(X_train, y_train, batch_size, self.frame_length)
        val_gen = LibriSpeechGenerator(X_val, y_val, batch_size, self.frame_length)
        test_gen = LibriSpeechGenerator(X_test, y_test, batch_size, self.frame_length)
        return train_gen, val_gen, test_gen