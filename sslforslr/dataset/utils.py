import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from collections import OrderedDict

AUDIO_FILE_CACHE_ENABLE = False
AUDIO_FILE_CACHE_LIMIT = -1

class AudioFileCache:
    data = OrderedDict()

def read_audio(path):
    if not AUDIO_FILE_CACHE_ENABLE:
        return sf.read(path)

    # Retrieve from cache or store file
    if path in AudioFileCache.data:
        return AudioFileCache.data[path]
    
    AudioFileCache.data[path] = sf.read(path)

    if AUDIO_FILE_CACHE_LIMIT > 0:
        if len(AudioFileCache.data) >= AUDIO_FILE_CACHE_LIMIT:
            AudioFileCache.data.popitem(last=False)

    return AudioFileCache.data[path]

def load_audio(path, frame_length, num_frames=1, min_length=None):
    audio, sr = read_audio(path)

    # Pad signal if it is shorter than min_length
    if min_length is None: min_length = frame_length
    if min_length and len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio) + 1), 'wrap')

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

def extract_mfcc(audio):
    audio = torch.from_numpy(audio) # (N, T)

    audio = pre_emphasis(audio)
    
    mfcc = torchaudio.transforms.MelSpectrogram(
        n_fft=512,
        win_length=400,
        hop_length=160,
        window_fn=torch.hamming_window,
        n_mels=40)(audio) # mfcc: (N, C, T)

    mfcc = mfcc.numpy().transpose(0, 2, 1) # (N, T, C)
    
    # torchaudio MelSpectrogram method might return a larger sequence
    limit = audio.shape[1] // 160
    mfcc = mfcc[:, :limit, :]
    
    return mfcc