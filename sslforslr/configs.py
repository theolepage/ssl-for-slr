from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = 'Adam'

@dataclass
class DatasetAugmentConfig:
    musan_path: str
    rir_path: str
    musan_noise_snr: Tuple[int, int] = (0, 15)
    musan_speech_snr: Tuple[int, int] = (13, 20)
    musan_music_snr: Tuple[int, int] = (5, 15)

@dataclass
class DatasetConfig:
    augment: DatasetAugmentConfig = None
    sample_frequency: int = 16000
    frame_length: int = 16000
    frame_split: bool = False
    max_samples: int = None
    extract_mfcc: bool = False
    val_ratio: float = 0.1
    train: str = './data/voxceleb2_train/wav.scp'
    test: str = './data/voxceleb1_test/wav.scp'
    trials: str = './data/voxceleb1_test/trials'

@dataclass
class ModelConfig:
    pass

@dataclass
class EncoderConfig:
    pass

@dataclass
class Config:
    training: TrainingConfig
    dataset: DatasetConfig

    _encoder: EncoderConfig = None
    _model: ModelConfig = None

    name: str = 'test'
    seed: int = 1717

    @property
    def encoder(self) -> EncoderConfig:
        return self._encoder

    @encoder.setter
    def encoder(self, v: EncoderConfig) -> None:
        self._encoder = v

    @property
    def model(self) -> ModelConfig:
        return self._model

    @model.setter
    def model(self, v: ModelConfig) -> None:
        self._model = v