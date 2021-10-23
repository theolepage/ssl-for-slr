from tensorflow_addons.layers import AdaptiveAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import regularizers
from sslforslr.modules import SincConv

from dataclasses import dataclass
from sslforslr.utils.Config import EncoderConfig

@dataclass
class SincEncoderConfig(EncoderConfig):
    weight_reg: float = 1e-4

SincEncoderConfig.__NAME__ = 'sinc'

class SincEncoder(Model):
    '''
    Encoder of the original PASE+ implementation based on SincNet (SincConv).

    "Multi-task self-supervised learning for Robust Speech Recognition"
    Mirco Ravanelli et al.
    https://arxiv.org/pdf/2001.09239.pdf
    '''

    def __init__(self, sample_frequency, config):
        super(SincEncoder, self).__init__()

        self.encoded_dim = config.encoded_dim
        self.reg = regularizers.l2(config.weight_reg)

        # conv_nb_filters = [64, 64, 128, 128, 256, 256, 512, 512]
        # conv_kernel_sizes = [251, 20, 11, 11, 11, 11, 11, 11]
        # conv_strides = [1, 10, 2, 1, 2, 1, 2, 2]
        conv_nb_filters = [512, 512, 512, 512, 512]
        conv_kernel_sizes = [10, 8, 4, 4, 4]
        conv_strides = [5, 4, 2, 2, 2]
        nb_blocks = len(conv_nb_filters)

        self.blocks = []
        for i, (f, w, s) in enumerate(zip(conv_nb_filters,
                                          conv_kernel_sizes,
                                          conv_strides)):
            self.blocks.append(SincEncoderBlock(f, w, s,
                                                sample_frequency,
                                                self.reg,
                                                i == 0))

        self.conv = Conv1D(filters=self.encoded_dim,
                           kernel_size=1,
                           kernel_regularizer=self.reg,
                           bias_regularizer=self.reg)
        self.bn = LayerNormalization()

    def call(self, X):
        for block in self.blocks: X = block(X)
        X = self.conv(X)
        X = self.bn(X)
        return X

    def compute_output_shape(self, input_shape):
        nb_timesteps = input_shape[0] // 160
        return (nb_timesteps, self.encoded_dim)


class SincEncoderBlock(Layer):

    def __init__(
        self,
        filters,
        kernel_size,
        stride,
        sample_frequency,
        reg,
        sincconv=False,
        **kwargs
    ):
        super(SincEncoderBlock, self).__init__(**kwargs)

        if sincconv:
            self.conv = SincConv(nb_filters=filters,
                                 kernel_size=kernel_size,
                                 sample_freq=sample_frequency,
                                 stride=stride,
                                 padding='SAME')
        else:
            self.conv = Conv1D(filters=filters,
                               kernel_size=kernel_size,
                               strides=stride,
                               padding='SAME',
                               kernel_regularizer=reg,
                               bias_regularizer=reg)

        self.normalization = LayerNormalization()
        self.activation = ReLU()

    def call(self, X):
        X = self.conv(X)
        X = self.normalization(X)
        X = self.activation(X)
        return X