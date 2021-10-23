from dataclasses import dataclass

from sslforslr.utils.Config import ModelConfig

@dataclass
class CPCContextNetworkConfig:
    model_type: str = 'gru'
    dim: int = 256
    nb_layers: int = 1

@dataclass
class CPCModelConfig(ModelConfig):
    context_network: CPCContextNetworkConfig = CPCContextNetworkConfig()
    nb_timesteps_to_predict: int = 12
    bidirectional: bool = False
    weight_reg: float = 1e-4

CPCModelConfig.__NAME__ = 'cpc'