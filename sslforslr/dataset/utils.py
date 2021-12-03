import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from io import BytesIO
import glob
import os
from tqdm import tqdm

AUDIO_CACHE_ENABLE = True
AUDIO_CACHE_LIMIT = -1

class AudioCache:
    data = {}
    base_path = None

def create_audio_cache():
    base_path = AudioCache.base_path

    files = []
    files += glob.glob(os.path.join(base_path, 'simulated_rirs', '*/*/*.wav'))
    files += glob.glob(os.path.join(base_path, 'musan_split', '*/*/*.wav'))
    files += glob.glob(os.path.join(base_path, 'voxceleb1', '*/*/*.wav'))
    
    print('Creating cache of audio files...')
    for path in tqdm(files):
        if AUDIO_CACHE_LIMIT > 0 and len(AudioCache.data) > AUDIO_CACHE_LIMIT:
            break
        with open(path, 'rb') as file_data:
            AudioCache.data[path] = file_data.read()

def read_audio(path):
    if AUDIO_CACHE_ENABLE:
        if not AudioCache.data:
            create_audio_cache()
        if path in AudioCache.data:
            return sf.read(BytesIO(AudioCache.data[path]))
    return sf.read(path)

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
