import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras import regularizers

from dataclasses import dataclass
from sslforslr.configs import EncoderConfig

@dataclass
class CPCEncoderConfig(EncoderConfig):
    encoded_dim: int = 512
    weight_reg: float = 1e-4

CPCEncoderConfig.__NAME__ = 'cpc'

class CPCEncoder(Model):
    '''
    Encoder of the original CPC implementation.

    "Representation Learning with Contrastive Predictive Coding"
    Aaron van den Oord, Yazhe Li, Oriol Vinyals
    https://arxiv.org/pdf/1807.03748.pdf
    '''

    def __init__(self, config):
        super(CPCEncoder, self).__init__()

        self.encoded_dim = config.encoded_dim
        self.reg = regularizers.l2(config.weight_reg)

        nb_filters = [512, 512, 512, 512, self.encoded_dim]
        kernel_sizes = [10, 8, 4, 4, 4]
        strides = [5, 4, 2, 2, 2]

        self.blocks = []
        for i in range(5):
            self.blocks.append(Conv1D(nb_filters[i],
                                      kernel_size=kernel_sizes[i],
                                      strides=strides[i],
                                      padding='same',
                                      kernel_regularizer=self.reg,
                                      bias_regularizer=self.reg))
            self.blocks.append(LayerNormalization())
            self.blocks.append(ReLU())

    def call(self, X):
        for layer in self.blocks:
            X = layer(X)
        return X

    def compute_output_shape(self, input_shape):
        nb_timesteps = input_shape[0] // 160
        return (nb_timesteps, self.encoded_dim)