import numpy as np
import torch
import torchaudio
import soundfile as sf

def load_wav(path, frame_length):
    data, sr = sf.read(path)
    data = data.reshape((len(data), 1))

    if len(data) < frame_length:
        data = np.pad(data, (0, frame_length - len(data) + 1), 'wrap')

    offset = np.random.randint(0, len(data) - frame_length + 1)
    data = data[offset:offset+frame_length]

    return data

def extract_mfcc(audio):
    mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(audio.T),
                                            num_ceps=40,
                                            num_mel_bins=40)
    # mfcc = torchaudio.transforms.SlidingWindowCmn(norm_vars=False)(mfcc)
    return mfcc.numpy()