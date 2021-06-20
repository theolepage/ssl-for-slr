from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class VQWave2VecConfig:
    nb_timesteps_to_predict: int = field(
        default=12,
        metadata={
            "help": "Number of timesteps to predict in contrastive loss"
        }
    )

    nb_timesteps_for_context: int = field(
        default=128-12,
        metadata={
            "help": "Number of timesteps used for context in contrastive loss"
        }
    )

    encoder_conv_layers: str = field(
        default="[(512, 10, 5)] + [(512, 8, 4)] + [(512, 4, 2)] * 3",
        metadata={
            "help": "Hyperparameters of encoder conv layers. "
            "Format: [(dim, kernel_size, stride), ...]"}
    )

    encoder_dim: int = field(
        default=512,
        metadata={
            "help": "Dimension of encoder embeddings"}
    )

    transformer_dim: int = field(
        default=768,
        metadata={"help": "Input dimension for transformer"}
    )

    transformer_layers: int = field(
        default=12,
        metadata={"help": "Number of layers for transformer"}
    )

    transformer_attention_heads: int = field(
        default=12,
        metadata={"help": "Number of attention heads for transformer"}
    )

    transformer_ffn_dim: int = field(
        default=3072, metadata={"help": "Embedding dimension for FFN in transformer"}
    )

    quantizer_dim: int = field(
        default=256,
        metadata={"help": "Dimension of codebooks vectors"}
    )

    quantizer_nb_groups: int = field(
        default=2,
        metadata={"help": "Number of codebooks groups"}
    )

    quantizer_nb_vars: int = field(
        default=320,
        metadata={"help": "Number of codebooks vars (per group)"}
    )

    quantizer_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "Temperature for Gumble softmax during vector quantization. "
            "Format: (start, end, decay)"
        },
    )

    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate applied to features after encoder"}
    )

    mask_length: int = field(
        default=10,
        metadata={"help": "Number of timesteps to mask"}
    )

    mask_prob: float = field(
        default=0.65,
        metadata={"help": "Probability of masking a timestep"}
    )

    diversity_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight of diversity loss used on quantizer"}
    )

    features_loss_weight: float = field(
        default=10,
        metadata={"help": "Weight of features penalty loss"}
    )