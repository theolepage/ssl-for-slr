from dataclasses import dataclass

from sslforslr.configs import ModelConfig

@dataclass
class SimCLRModelConfig(ModelConfig):
    channel_loss_factor: float = 0.1
    weight_reg: float = 1e-4

SimCLRModelConfig.__NAME__ = 'simclr'