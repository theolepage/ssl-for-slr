import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import regularizers

class SincEncoder(Model):
    '''
    Encoder of the original PASE+ implementation based on SincNet (SincConv).

    "Multi-task self-supervised learning for Robust Speech Recognition"
    Mirco Ravanelli et al.
    https://arxiv.org/pdf/2001.09239.pdf
    '''

    def __init__(self,
                 encoded_dim,
                 sample_frequency,
                 skip_connections_enabled,
                 rnn_enabled,
                 weight_regularizer=0.0):
        super(SincEncoder, self).__init__()

        self.skip_connections_enabled = skip_connections_enabled
        self.rnn_enabled = rnn_enabled
        self.reg = regularizers.l2(weight_regularizer)

        conv_nb_filters = [64, 64, 128, 128, 256, 256, 512, 512]
        conv_kernel_sizes = [251, 20, 11, 11, 11, 11, 11, 11]
        conv_strides = [1, 10, 2, 1, 2, 1, 2, 2]
        nb_blocks = len(conv_nb_filters)

        self.blocks = []
        self.skips = []
        for i, (f, w, s) in enumerate(zip(conv_nb_filters,
                                          conv_kernel_sizes,
                                          conv_strides)):
            self.blocks.append(SincEncoderBlock(f, w, s,
                                                sample_frequency,
                                                self.reg,
                                                i == 0))

            if self.skip_connections_enabled and i < nb_blocks - 1:
                self.skips.append(Conv1D(filters=self.encoded_dim,
                                         kernel_size=1,
                                         use_bias=False,
                                         kernel_regularizer=self.reg))

        if self.rnn_enabled:
            self.rnn = Bidirectional(GRU(units=self.encoded_dim,
                                         return_sequences=True,
                                         kernel_regularizer=self.reg,
                                         recurrent_regularizer=self.reg,
                                         bias_regularizer=self.reg))

        self.conv = Conv1D(filters=self.encoded_dim,
                           kernel_size=1,
                           kernel_regularizer=self.reg,
                           bias_regularizer=self.reg)
        self.bn = BatchNormalization() # FIXME: reproduce pytorch affine=False

    def call(self, X):
        skip_values = []
        
        for i, block in enumerate(self.blocks):
            X = block(X)

            if self.skip_connections_enabled and i < len(self.skips):
                skip_values.append(self.skips[i](X))

        if self.rnn_enabled:
            X = self.rnn(X)

        X = self.conv(X)

        if self.skip_connections_enabled:
            for skip in skip_values:
                skip = tfa.layers.AdaptiveAveragePooling1D(X.shape[1])(skip)
                X = Add()([X, skip])

        X = self.bn(X) 
        return X

    def compute_output_shape(self, input_shape):
        nb_timesteps = input_shape[0] // 160
        return (nb_timesteps, self.encoded_dim)


class SincEncoderBlock(Layer):

    def __init__(self, filters, kernel_size, stride, sample_frequency, reg,
                 sincconv=False, **kwargs):
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
                               kernel_regularizer=self.reg,
                               bias_regularizer=self.reg)
            
        self.normalization = BatchNormalization(center=False, scale=False)
        self.activation = PReLU(shared_axes=[1])

    def call(self, X):
        X = self.conv(X)
        X = self.normalization(X)
        X = self.activation(X)
        return X