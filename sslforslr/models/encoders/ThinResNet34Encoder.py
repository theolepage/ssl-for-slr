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
    def __init__(self, filters, stride=1, reg=None, identity=False):
        super().__init__()

        self.reg = reg
        self.identity = identity

        self.conv1 = Conv2D(filters[0],
                            kernel_size=1,
                            strides=stride,
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.bn1 = BatchNormalization()
        
        self.conv2 = Conv2D(filters[1],
                            kernel_size=3,
                            padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(filters[2],
                            kernel_size=1,
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.bn3 = BatchNormalization()
        
        self.relu = ReLU()

        self.convert_dim = Conv2D(filters=filters[2],
                                  kernel_size=1,
                                  strides=stride)

    def call(self, X):
        residual = X

        Z = self.conv1(X)
        Z = self.bn1(Z)
        Z = self.relu(Z)

        Z = self.conv2(Z)
        Z = self.bn2(Z)
        Z = self.relu(Z)

        Z = self.conv3(Z)
        Z = self.bn3(Z)
        
        if not self.identity:
            residual = self.convert_dim(residual)
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

    "Utterance-level Aggregation For Speaker Recognition In The Wild"
    Weidi Xie, Arsha Nagrani, Joon Son Chung, Andrew Zisserman
    https://arxiv.org/pdf/1902.10107.pdf
    '''

    def __init__(self, encoded_dim, weight_regularizer=0.0):
        super().__init__()

        self.encoded_dim = encoded_dim
        self.reg = regularizers.l2(weight_regularizer)

        self.conv1 = Conv2D(64, 7, 1, padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp1 = MaxPooling2D(3, 2)

        self.block1 = self.__make_block(2, [48, 48, 96], 1)
        self.block2 = self.__make_block(3, [96, 96, 128], 2)
        self.block3 = self.__make_block(3, [128, 128, 256], 2)
        self.block4 = self.__make_block(3, [256, 256, 512], 2)

        self.mp2 = MaxPooling2D((3, 1), 2)
        self.conv2 = Conv2D(encoded_dim, (7, 1))

        self.sap = SAP(self.reg)

    def __make_block(self, num, filters, stride=1):
        layers = []
        for i in range(num):
            if i == 0:
               layers.append(ResNetBlock(filters, stride, self.reg))
            else:
               layers.append(ResNetBlock(filters, 1, self.reg, identity=True))
        return Sequential(layers)

    def call(self, X):
        Z = self.conv1(X)
        Z = self.bn(Z)
        Z = self.relu(Z)
        Z = self.mp1(Z)

        Z = self.block1(Z)
        Z = self.block2(Z)
        Z = self.block3(Z)
        Z = self.block4(Z)

        Z = self.mp2(Z)
        Z = self.conv2(Z)
        Z = tf.squeeze(Z, axis=1) # shape: BTC

        Z = self.sap(Z)

        return Z

    def compute_output_shape(self, input_shape):
        return (self.encoded_dim)