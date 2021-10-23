from dataclasses import dataclass
from enum import Enum

from sslforslr.configs import ModelConfig

class LIMLossFnEnum(Enum):
    BCE = 'bce'
    NCE = 'nce'
    MINE = 'mine'

@dataclass
class LIMModelConfig(ModelConfig):
    loss_fn: LIMLossFnEnum = 'bce'
    context_length: int = 1
    weight_reg: float = 1e-4

LIMModelConfig.__NAME__ = 'lim'