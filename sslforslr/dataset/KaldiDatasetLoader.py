import numpy as np
from tensorflow.keras.utils import Sequence
import kaldi_io

class KaldiDatasetGenerator(Sequence):
    def __init__(self, batch_size, frame_length, scp_path, utt2spkid_path):
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.scp_path = scp_path

        self.rxfiles, self.labels, self.utt2spkid = [], [], {}
        
        # Determine number of utterances per speaker 
        id_count = {}
        for line in open(utt2spkid_path):
            utt, label = line.rstrip().split()
            self.utt2spkid[utt] = int(label)

            if not int(label) in id_count:
                id_count[int(label)] = 0
            id_count[int(label)] += 1
        
        max_id_count = int((max(id_count.values()) + 1) / 2)
        self.nb_categories = len(id_count)

        # Duplicate samples for speaker with less utterances
        for line in open(scp_path):
            utt, rxfile = line.rstrip().split()
            label = self.utt2spkid[utt]
            repetition = max(1, max_id_count // id_count[label])
            self.rxfiles.extend([rxfile] * repetition)
            self.labels.extend([label] * repetition)

    def get_nb_categories(self):
        return self.nb_categories

    def __len__(self):
        return 5
        # FIXME
        #return len(self.labels) // self.batch_size

    def __getitem__(self, i):
        X, y = [], []

        for j in range(self.batch_size):
            index = i * self.batch_size + j
            sample = kaldi_io.read_mat(self.rxfiles[index])
            label = self.labels[index]

            #assert len(sample) >= self.frame_length
            offset = 0 #np.random.randint(0, len(sample) - self.frame_length + 1)
            sample = sample[offset:offset+self.frame_length, :]
           
            # FIXME: use correct data and set correct input shape
            X.append(np.arange(20480).reshape((20480, 1)).astype(np.float64))
            y.append(0)
            #X.append(sample)
            #y.append(label)
        
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        # Shuffle elements in batch?
        pass

class KaldiDatasetLoader:
    def __init__(self, seed, config):
        self.config = config

    def load(self, batch_size, checkpoint_dir):
        frame_length = self.config['frames']['length']

        train = KaldiDatasetGenerator(batch_size,
                                      frame_length,
                                      self.config['train_scp'],
                                      self.config['utt2spkid'])
        val = KaldiDatasetGenerator(batch_size,
                                    frame_length,
                                    self.config['val_scp'],
                                    self.config['utt2spkid'])
        
        nb_categories = train.get_nb_categories()

        return [train, val], nb_categories
