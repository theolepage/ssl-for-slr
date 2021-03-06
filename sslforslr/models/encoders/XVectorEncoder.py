import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from dataclasses import dataclass
from sslforslr.configs import EncoderConfig

@dataclass
class XVectorEncoderConfig(EncoderConfig):
    encoded_dim: int = 3000
    weight_reg: float = 1e-4

XVectorEncoderConfig.__NAME__ = 'xvector'

class TDNN(Layer):
    '''
    Time delay neural network (TDNN) layer.

    "A time delay neural network architecture for efficient modeling of long temporal contexts"
    Vijayaditya Peddinti, Daniel Povey, Sanjeev Khudanpur
    https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
    '''

    def __init__(self, filters, kernel_size, reg, sub_sampling=False, dropout=0.5):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.sub_sampling = sub_sampling
        self.reg = reg

        self.activation = ReLU()
        self.bn = BatchNormalization()
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        input_dim = input_shape[-1] # Assuming BTC format

        # FIXME
        if self.sub_sampling: input_dim = 512
        else: input_dim = 40

        self.kernel_shape = (self.kernel_size, input_dim, self.filters)
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.kernel_shape,
                                      initializer="random_normal",
                                      trainable=True,
                                      regularizer=self.reg)
        
        if self.sub_sampling:
            self.mask = np.zeros(self.kernel_shape).astype(np.float32)
            self.mask[0][0] = 1
            self.mask[self.kernel_size - 1][0] = 1

    def call(self, X):
        kernel = self.kernel
        if self.sub_sampling:
            kernel = kernel * self.mask

        X = tf.nn.conv1d(X, kernel, stride=1, padding="SAME")
        
        X = self.activation(X)
        X = self.bn(X)
        X = self.dropout(X)
        return X

class XVectorEncoder(Model):
    '''
    Encoder based on x-vectors neural network architecture.

    "X-Vectors: Robust DNN Embeddings for Speaker Recognition"
    David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, Sanjeev Khudanpur
    https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
    '''

    def __init__(self, config):
        super().__init__()

        self.encoded_dim = config.encoded_dim
        self.reg = regularizers.l2(config.weight_reg)

        self.tdnn1 = TDNN(filters=512, kernel_size=5, sub_sampling=False, reg=self.reg)
        self.tdnn2 = TDNN(filters=512, kernel_size=5, sub_sampling=True, reg=self.reg)
        self.tdnn3 = TDNN(filters=512, kernel_size=7, sub_sampling=True, reg=self.reg)
        self.tdnn4 = TDNN(filters=512, kernel_size=1, sub_sampling=True, reg=self.reg)
        self.tdnn5 = TDNN(filters=self.encoded_dim // 2, kernel_size=1, sub_sampling=True, reg=self.reg)

    def call(self, X):
        # TDNN layers
        X = self.tdnn1(X)
        X = self.tdnn2(X)
        X = self.tdnn3(X)
        X = self.tdnn4(X)
        X = self.tdnn5(X)

        # Stats pooling
        mean = tf.math.reduce_mean(X, axis=1)
        std = tf.math.reduce_std(X, axis=1)
        stat_pooling = tf.concat([mean, std], axis=1)
        
        return stat_pooling

    def compute_output_shape(self, input_shape):
        return (self.encoded_dim)