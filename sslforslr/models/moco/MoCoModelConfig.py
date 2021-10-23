from dataclasses import dataclass

from sslforslr.configs import ModelConfig

@dataclass
class MoCoModelConfig(ModelConfig):
    queue_size: int = 10000
    info_nce_temp: float = 0.07
    embedding_dim: int = 512
    weight_reg: float = 1e-4

MoCoModelConfig.__NAME__ = 'moco'