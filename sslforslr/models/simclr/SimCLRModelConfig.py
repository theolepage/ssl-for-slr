from dataclasses import dataclass
from typing import List

from sslforslr.configs import ModelConfig

@dataclass
class SimCLRModelConfig(ModelConfig):
    enable_mlp: bool = False
    mlp_dim: int = 2048
    
    infonce_loss_factor: float = 1.0

    vic_reg_factor: float = 0.1
    vic_reg_inv_weight: float = 1.0
    vic_reg_var_weight: float = 1.0
    vic_reg_cov_weight: float = 0.04
    
    barlow_twins_factor: float = 0.0
    barlow_twins_lambda: float = 0.05

    representations_loss_vic: bool = False
    representations_loss_nce: bool = False
    embeddings_loss_vic: bool = True
    embeddings_loss_nce: bool = True

    weight_reg: float = 1e-4

SimCLRModelConfig.__NAME__ = 'simclr'
