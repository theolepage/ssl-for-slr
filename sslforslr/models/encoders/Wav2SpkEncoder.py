import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras import regularizers

from tensorflow_addons.layers import InstanceNormalization

from dataclasses import dataclass
from sslforslr.utils.Config import EncoderConfig

@dataclass
class Wav2SpkEncoderConfig(EncoderConfig):
    weight_reg: float = 1e-4

Wav2SpkEncoderConfig.__NAME__ = 'wav2spk'

class TemporalGating(Layer):
    def __init__(self, nb_channels, nb_timesteps, reg):
        super().__init__()

        self.reg = reg

        self.tg_weights = self.add_weight(
            name='tg_weights',
            shape=(nb_channels, 1),
            initializer="random_normal",
            trainable=True,
            regularizer=self.reg)
        self.tg_biases = self.add_weight(
            name='tg_biases',
            shape=(nb_timesteps,),
            initializer="random_normal",
            trainable=True,
            regularizer=self.reg)
    
    def call(self, X):
        tmp = tf.transpose(self.tg_weights) @ tf.transpose(X, (0, 2, 1))
        tmp = tmp + self.tg_biases
        tmp = tf.squeeze(tmp, axis=1)
        X = tf.math.sigmoid(tmp) * tf.transpose(X, (2, 0, 1))
        X = tf.transpose(X, (1, 2, 0))
        return X

class Wav2SpkEncoder(Model):
    '''
    Encoder of the original Wav2Spk implementation.

    "Wav2Spk: A Simple DNN Architecture for Learning Speaker Embeddings from Waveforms"
    Weiwei Lin and Man-Wai Mak
    https://indico2.conference4me.psnc.pl/event/35/contributions/3613/attachments/1128/1170/Wed-3-5-1.pdf
    '''

    def __init__(self, config):
        super(Wav2SpkEncoder, self).__init__()

        self.encoded_dim = config.encoded_dim
        self.reg = regularizers.l2(config.weight_reg)

        nb_filters = [40, 200, 300, 512, 512]
        kernel_sizes = [10, 5, 5, 3, 3]
        strides = [5, 4, 2, 2, 2]

        self.blocks = []
        for i in range(5):
            self.blocks.append(Conv1D(nb_filters[i],
                                      kernel_size=kernel_sizes[i],
                                      strides=strides[i],
                                      padding='same',
                                      kernel_regularizer=self.reg,
                                      bias_regularizer=self.reg))
            self.blocks.append(ReLU())
            self.blocks.append(InstanceNormalization())

        self.temporal_gating = TemporalGating(nb_channels=512,
                                              nb_timesteps=128,
                                              reg=self.reg)

        self.conv5 = Conv1D(512,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.conv6 = Conv1D(512,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.conv7 = Conv1D(512,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.conv8 = Conv1D(512,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)

        self.fc0 = Dense(encoded_dim, kernel_regularizer=self.reg, bias_regularizer=self.reg)

    def call(self, X):
        # Feature encoder
        for layer in self.blocks:
            X = layer(X)

        # Temporal gating
        X = self.temporal_gating(X)

        # Frames aggregator
        X = self.conv5(X)
        X = self.conv6(X)
        X = self.conv7(X)
        X = self.conv8(X)

        # Stats pooling
        # mean = tf.math.reduce_mean(X, axis=1)
        # std = tf.math.reduce_std(X, axis=1)
        # stat_pooling = tf.concat([mean, std], axis=1)

        # Utterance layers
        # X = self.fc0(stat_pooling)

        return X

    def compute_output_shape(self, input_shape):
        nb_timesteps = input_shape[0] // 160
        return (nb_timesteps, self.encoded_dim)