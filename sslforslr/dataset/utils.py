import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from SpecAugment.spec_augment_pytorch import spec_augment

def load_wav(path, frame_length, num_frames=1, min_audio_length=None):
    audio, sr = sf.read(path)

    # Pad signal if it is shorter than min_audio_length
    if min_audio_length and len(audio) < min_audio_length:
        audio = np.pad(audio, (0, min_audio_length - len(audio) + 1), 'wrap')

    # Load entire audio data if frame_length is not specified
    if frame_length is None: frame_length = len(audio)

    # Determine frames start indices
    idx = []
    if num_frames == 1:
        idx = [np.random.randint(0, len(audio) - frame_length + 1)]
    else:
        idx = np.linspace(0, len(audio) - frame_length, num=num_frames)

    # Extract frames
    data = [audio[int(i):int(i)+frame_length] for i in idx]
    data = np.stack(data, axis=0).astype(np.float32)

    return data # (num_frames, T)

def pre_emphasis(audio, coef=0.97):
    w = torch.FloatTensor([-coef, 1.0]).unsqueeze(0).unsqueeze(0)
    audio = audio.unsqueeze(1)
    audio = F.pad(audio, (1, 0), 'reflect')
    return F.conv1d(audio, w).squeeze(1)

def extract_mfcc(audio, enable_spec_augment=False):
    audio = torch.from_numpy(audio) # (N, T)

    audio = pre_emphasis(audio)
    
    mfcc = torchaudio.transforms.MelSpectrogram(
        n_fft=512,
        win_length=400,
        hop_length=160,
        window_fn=torch.hamming_window,
        n_mels=40)(audio) # mfcc: (N, C, T)
    
    if enable_spec_augment:
        mfcc = spec_augment(mfcc)

    mfcc = mfcc.numpy().transpose(0, 2, 1) # (N, T, C)
    
    # torchaudio MelSpectrogram method might return a larger sequence
    limit = audio.shape[1] // 160
    mfcc = mfcc[:, :limit, :]
    
    return mfcc