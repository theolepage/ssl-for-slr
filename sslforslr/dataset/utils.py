import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

def load_wav(path, frame_length):
    data, sr = sf.read(path)

    # Load entire audio data if frame_length is not specified
    if frame_length is None: frame_length = len(data)

    if len(data) < frame_length:
        data = np.pad(data, (0, frame_length - len(data) + 1), 'wrap')

    offset = np.random.randint(0, len(data) - frame_length + 1)
    data = data[offset:offset+frame_length]
    data = data.reshape((len(data), 1))

    return data

def pre_emphasis(audio, coef=0.97):
    w = torch.FloatTensor([-coef, 1.0]).unsqueeze(0).unsqueeze(0)
    audio = audio.unsqueeze(1)
    audio = F.pad(audio, (1, 0), 'reflect')
    return F.conv1d(audio, w).squeeze(1)

def extract_mfcc(audio):
    audio = torch.from_numpy(audio.astype(np.float32).T) # (T, 1) -> (1, T)
    audio = pre_emphasis(audio)
    mfcc = torchaudio.transforms.MelSpectrogram(
        n_fft=512,
        win_length=400,
        hop_length=160,
        window_fn=torch.hamming_window,
        n_mels=40)(audio)

    return mfcc.numpy().squeeze(axis=0).T # (T, C) = (200, 40)