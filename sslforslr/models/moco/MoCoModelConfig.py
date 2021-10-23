from dataclasses import dataclass

from sslforslr.utils.Config import ModelConfig

@dataclass
class MoCoModelConfig(ModelConfig):
    queue_size: int = 10000
    info_nce_temp: float = 0.07
    embedding_dim: int = 512
    proto_nce_loss_factor: float = 0.25
    nb_clusters: int = 5000
    clustering_negs_count: int = 10000
    epochs_before_proto_nce: int = 1111111111111
    weight_reg: float = 1e-4

MoCoModelConfig.__NAME__ = 'moco'