import numpy as np
from tensorflow.keras.utils import Sequence
import kaldiio

class KaldiDatasetGenerator(Sequence):
    def __init__(self, batch_size, frame_length, scp_path, utt2spkid, max_samples = 0):
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.scp_path = scp_path

        self.rxfiles, self.labels = [], []
        index = 0
        for line in open(scp_path):
            if max_samples and index >= max_samples:
                break

            utt = line_parts[0]
            rxfile = ' '.join(line_parts[1:])
            label = utt2spkid[utt]

            self.rxfiles.append(rxfile)
            self.labels.append(label)

            index += 1

    def __len__(self):
        # FIXME: Handle last batch having < self.batch_size elements
        return len(self.labels) // self.batch_size

    def __getitem__(self, i):
        X, y = [], []

        for j in range(self.batch_size):
            index = i * self.batch_size + j
            sample = kaldiio.load_mat(self.rxfiles[index])
            label = self.labels[index]

            assert len(sample) >= self.frame_length
            offset = np.random.randint(0, len(sample) - self.frame_length + 1)
            sample = sample[offset:offset+self.frame_length, :]

            X.append(sample)
            y.append(label)
        
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        # FIXME: Shuffle elements in batch?
        pass

class KaldiDatasetLoader:
    def __init__(self, seed, config):
        self.config = config

        # Determine number of utterances per speaker 
        self.utt2spkid = {}
        self.spk2num = {}
        for line in open(self.config['utt2spkid']):
            utt, label = line.rstrip().split()
            label = int(label)
            self.utt2spkid[utt] = label

            if not label in self.spk2num:
                self.spk2num[label] = 0
            self.spk2num[label] += 1

    def load(self, batch_size, checkpoint_dir):
        frame_length = self.config['frames']['length']
        max_samples = self.config.get('max_samples', None)
        
        nb_categories = len(self.spk2num)

        train = KaldiDatasetGenerator(batch_size,
                                      frame_length,
                                      self.config['train_scp'],
                                      self.utt2spkid,
                                      max_samples)

        val = KaldiDatasetGenerator(batch_size,
                                    frame_length,
                                    self.config['val_scp'],
                                    self.utt2spkid,
                                    max_samples)
        
        return [train, val], nb_categories
