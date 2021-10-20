import numpy as np
from tensorflow.keras.utils import Sequence
import kaldiio
import soundfile as sf
from sklearn.model_selection import train_test_split

import torch
import torchaudio

def extract_mfcc(audio):
    mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(audio.T),
                                            num_ceps=30,
                                            num_mel_bins=30)
    mfcc = torchaudio.transforms.SlidingWindowCmn(norm_vars=False)(mfcc)
    return mfcc.numpy()

class KaldiDatasetGenerator(Sequence):
    def __init__(self, batch_size, frames_config, rxfiles, labels, indices):
        self.batch_size = batch_size
        self.frame_length = frames_config['length']
        self.extract_mfcc = frames_config.get('extract_mfcc', False)
        self.rxfiles = rxfiles
        self.labels = labels
        self.indices = indices

    def __len__(self):
        # FIXME: Handle last batch having < self.batch_size elements
        return len(self.indices) // self.batch_size

    def __getitem__(self, i):
        X, y = [], []

        for j in range(self.batch_size):
            index = self.indices[i * self.batch_size + j]

            sample, sr = sf.read(self.rxfiles[index])
            data = sample.reshape((len(sample), 1))
            label = self.labels[index]

            if self.extract_mfcc:
                data = extract_mfcc(sample)

            assert len(data) >= self.frame_length
            offset = np.random.randint(0, len(data) - self.frame_length + 1)
            data = data[offset:offset+self.frame_length]

            X.append(data)
            y.append(label)

        return np.array(X), np.array(y)

class KaldiDatasetLoader:
    def __init__(self, seed, config):
        self.config = config

        self.create_utt2spkid()

        # Create a list for rxfiles and labels of each utterance
        self.rxfiles = []
        self.labels = []
        for line in open(self.config['scp']):
            # Parse scp line
            line_parts = line.rstrip().split()
            utt = line_parts[0]
            rxfile = ' '.join(line_parts[1:])
            
            self.rxfiles.append(rxfile)
            self.labels.append(self.utt2spkid[utt])

    def create_utt2spkid(self):
        # Associate each utterance to a unique speaker id (starting from 0) 
        self.utt2spkid = {}
        speaker_ids = {}
        current_speaker_id = 0
        for line in open(self.config['utt2spk']):
            utt, label = line.rstrip().split()
            if label in speaker_ids:
                label = speaker_ids[label]
            else:
                speaker_ids[label] = current_speaker_id
                label = current_speaker_id
                current_speaker_id += 1
            self.utt2spkid[utt] = label
        self.nb_categories = current_speaker_id

    def load(self, batch_size, checkpoint_dir):
        frames_config = self.config['frames']
        val_ratio = self.config.get('val_ratio', 0.1)

        # Create list of indices to shuffle easily during training
        max_samples = self.config.get('max_samples', None)
        nb_samples = max_samples if max_samples else len(self.labels)
        indices = np.arange(nb_samples)
        indices_train, indices_val = train_test_split(indices, test_size=val_ratio)

        train = KaldiDatasetGenerator(batch_size,
                                      frames_config,
                                      self.rxfiles,
                                      self.labels,
                                      indices_train)

        val = KaldiDatasetGenerator(batch_size,
                                    frames_config,
                                    self.rxfiles,
                                    self.labels,
                                    indices_val)
        
        return [train, val], self.nb_categories
