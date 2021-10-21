import glob
import os
import numpy as np
import random
from scipy.signal import convolve
import soundfile as sf

from sslforslr.dataset.utils import load_wav

MUSAN_SNR = {
    'noise': [0, 15],
    'speech': [13, 20],
    'music': [5, 15]
}

class AudioAugmentation:

    def __init__(self, config):
        self.rir_files = glob.glob(os.path.join(config['rir_path'], '*/*/*.wav'))

        self.musan_files = {}
        for file in glob.glob(os.path.join(config['musan_path'], '*/*/*.wav')):
            category = file.split('/')[-3]
            if not category in self.musan_files:
                self.musan_files[category] = []
            self.musan_files[category].append(file)

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, fs = sf.read(rir_file)
        rir = rir.reshape((-1, 1))
        rir = rir / np.sqrt(np.sum(rir ** 2))
        
        return convolve(audio, rir, mode='full')[:len(audio)]

    def add_noise(self, audio, category):
        noise_file = random.choice(self.musan_files[category])
        noise = load_wav(noise_file, len(audio))
        
        # Determine noise scale factor according to desired SNR
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4) 
        noise_db = 10 * np.log10(np.mean(noise[0] ** 2) + 1e-4) 
        noise_snr = random.uniform(
            MUSAN_SNR[category][0],
            MUSAN_SNR[category][1]
        )
        noise_scale = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10))

        return noise * noise_scale + audio

    def __call__(self, audio):
        transform_type = random.randint(0, 4)
        if transform_type == 1:
            audio = self.reverberate(audio)
        elif transform_type == 2:
            audio = self.add_noise(audio, 'music')
        elif transform_type == 3:
            audio = self.add_noise(audio, 'speech')
        elif transform_type == 4:
            audio = self.add_noise(audio, 'noise')
        return audio