import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras import regularizers

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
        Z = self.bn1(Z)
        Z = self.relu(Z)

        Z = self.conv2(Z)
        Z = self.bn2(Z)
        
        Z = Add()([Z, residual])
        Z = self.relu(Z)
        return Z

class SAP(Layer):
    '''
    "Self-Attention Encoding and Pooling for Speaker Recognition"
    Pooyan Safari, Miquel India, Javier Hernando
    https://arxiv.org/pdf/2008.01077v1.pdf
    '''

    def __init__(self, reg):
        super().__init__()

        self.W = Dense(1, kernel_regularizer=reg, bias_regularizer=reg)

    def call(self, X):
        W = tf.squeeze(self.W(X), axis=-1) # shape: (B, T)
        W = tf.nn.softmax(W)
        W = tf.expand_dims(W, axis=-1) # shape: (B, T, 1)
        out = tf.math.reduce_sum(W * X, axis=1)
        return out

class ThinResNet34Encoder(Model):
    '''
    Encoder based on thin-ResNet34 architecture.

    "Delving into VoxCeleb: environment invariant speaker recognition"
    Joon Son Chung, Jaesung Huh, Seongkyu Mun
    https://arxiv.org/pdf/1910.11238.pdf
    '''

    def __init__(self, encoded_dim=512, weight_regularizer=0.0):
        super().__init__()

        self.encoded_dim = encoded_dim
        self.reg = regularizers.l2(weight_regularizer)

        self.conv1 = Conv2D(16, 7, 2, padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp1 = MaxPooling2D(3, 2)

        self.block1 = self.__make_block(3, 16, 1)
        self.block2 = self.__make_block(4, 32, 2)
        self.block3 = self.__make_block(6, 64, 2)
        self.block4 = self.__make_block(3, 128, 2)

        self.conv2 = Conv2D(self.encoded_dim, (2, 2))
        self.gp = GlobalAveragePooling2D()

        self.sap = SAP(self.reg)

    def __make_block(self, num, filters, stride=1):
        layers = []
        layers.append(ResNetBlock(filters, stride, self.reg))
        for i in range(1, num):
            layers.append(ResNetBlock(filters, 1, self.reg))
        return Sequential(layers)

    def call(self, X):
        Z = tf.expand_dims(X, axis=-1)

        Z = self.conv1(Z)
        Z = self.bn(Z)
        Z = self.relu(Z)
        Z = self.mp1(Z)

        Z = self.block1(Z)
        Z = self.block2(Z)
        Z = self.block3(Z)
        Z = self.block4(Z)

        Z = self.conv2(Z)

        Z = tf.squeeze(Z, axis=2) # shape: BTC
        Z = self.sap(Z)

        return Z

    def compute_output_shape(self, input_shape):
        return (self.encoded_dim)