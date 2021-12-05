from dataclasses import dataclass

from sslforslr.configs import ModelConfig

@dataclass
class SimCLRModelConfig(ModelConfig):
    enable_mlp: bool = False
    
    infonce_loss_factor: float = 1.0

    vic_reg_factor: float = 0.1
    vic_reg_inv_weight: float = 1.0
    vic_reg_var_weight: float = 1.0
    vic_reg_cov_weight: float = 0.04
    
    barlow_twins_factor: float = 0.0
    barlow_twins_lambda: float = 0.05
    
    enable_mse_clean_aug: bool = False
    mse_clean_aug_factor: float = 0.1
    
    weight_reg: float = 1e-4

SimCLRModelConfig.__NAME__ = 'simclr'
