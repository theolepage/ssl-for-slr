import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow_addons.layers import InstanceNormalization
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras import regularizers

from dataclasses import dataclass

@dataclass
class ThinResNet34EncoderConfig():
    encoded_dim: int = 512
    weight_reg: float = 1e-4

ThinResNet34EncoderConfig.__NAME__ = 'thinresnet34'

class SELayer(Layer):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.avg_pool = AdaptiveAveragePooling2D(1)
        self.fc = Sequential([
            Dense(channels // reduction, activation='relu'),
            Dense(channels, activation='sigmoid')
        ])

    def call(self, X):
        Z = self.avg_pool(X) # (B, 1, 1, C)
        Z = tf.squeeze(Z, (1, 2)) # (B, C)
        Z = self.fc(Z)
        Z = tf.expand_dims(tf.expand_dims(Z, axis=1), axis=2)
        return X * Z

class ResNetBlock(Layer):
    def __init__(self, filters, stride=1, reg=None):
        super().__init__()

        self.reg = reg

        self.conv1 = Conv2D(filters,
                            kernel_size=3,
                            padding='same',
                            strides=stride,
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.bn1 = BatchNormalization()
        
        self.conv2 = Conv2D(filters,
                            kernel_size=3,
                            padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.bn2 = BatchNormalization()
        self.se = SELayer(filters)

        self.relu = ReLU()

        self.downsample = Sequential()
        if stride != 1:
            self.downsample.add(Conv2D(filters=filters,
                                       kernel_size=1,
                                       strides=stride))
            self.downsample.add(BatchNormalization())

    def call(self, X):
        residual = self.downsample(X)

        Z = self.conv1(X)
        Z = self.relu(Z)
        Z = self.bn1(Z)

        Z = self.conv2(Z)
        Z = self.bn2(Z)
        Z = self.se(Z)
        
        Z = Add()([Z, residual])
        Z = self.relu(Z)
        return Z

class SAP(Layer):
    '''
    "Self-Attention Encoding and Pooling for Speaker Recognition"
    Pooyan Safari, Miquel India, Javier Hernando
    https://arxiv.org/pdf/2008.01077v1.pdf
    '''

    def __init__(self, outmap_size, reg):
        super().__init__()

        self.outmap_size = outmap_size

        self.attention = Sequential([
            Conv1D(128, kernel_size=1),
            ReLU(),
            BatchNormalization(),
            Conv1D(1280, kernel_size=1),
            Softmax(axis=1)
        ])

    def call(self, X):
        # (B, H, W, C) = (None, 5, 38, 256) 

        X = tf.transpose(X, (0, 2, 1, 3))
        # (B, W, H, C) = (None, 38, 5, 256) 

        X = tf.reshape(X, (tf.shape(X)[0], tf.shape(X)[1], self.outmap_size))
        # (B, W, H*C) = (None, 38, 1280)

        W = self.attention(X)
        X = tf.math.reduce_sum(W * X, axis=1)
        # (B, H*C) = (None, 1280)

        return X

class ThinResNet34Encoder(Model):
    '''
    Encoder based on thin-ResNet34 architecture.

    "Delving into VoxCeleb: environment invariant speaker recognition"
    Joon Son Chung, Jaesung Huh, Seongkyu Mun
    https://arxiv.org/pdf/1910.11238.pdf
    '''

    def __init__(self, config):
        super().__init__()

        self.encoded_dim = config.encoded_dim
        self.reg = regularizers.l2(config.weight_reg)

        self.instance_norm = InstanceNormalization(axis=1)

        self.conv1 = Conv2D(32, 3, 1, padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.bn = BatchNormalization()
        self.relu = ReLU()

        self.block1 = self.__make_block(3, 32, 1)
        self.block2 = self.__make_block(4, 64, 2)
        self.block3 = self.__make_block(6, 128, 2)
        self.block4 = self.__make_block(3, 256, 2)
        
        outmap_size = int(40 / 8 * 256) # n_mels / 8 * last_filter_size
        self.sap = SAP(outmap_size, self.reg)

        self.fc = Dense(self.encoded_dim)

    def __make_block(self, num, filters, stride=1):
        layers = []
        layers.append(ResNetBlock(filters, stride, self.reg))
        for i in range(1, num):
            layers.append(ResNetBlock(filters, 1, self.reg))
        return Sequential(layers)

    def call(self, X):
        # X shape: (B, T, C)

        X = tf.transpose(X, (0, 2, 1)) # (B, C, T) = (B, 40, 200)
        
        X = tf.math.log(X + 1e-06)

        X.set_shape((X.shape[0], 40, X.shape[-1]))
        X = self.instance_norm(X)

        X = tf.expand_dims(X, axis=-1) # (B, H, W, C) = (B, 40, 200, 1)

        Z = self.conv1(X)
        Z = self.relu(Z)
        Z = self.bn(Z)

        Z = self.block1(Z)
        Z = self.block2(Z)
        Z = self.block3(Z)
        Z = self.block4(Z)

        Z = self.sap(Z)
        Z = self.fc(Z)

        return Z

    def compute_output_shape(self, input_shape):
        return (self.encoded_dim)