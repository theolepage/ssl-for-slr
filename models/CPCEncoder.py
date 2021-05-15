import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras import regularizers

class CPCEncoder(Model):

    def __init__(self, encoded_dim, weight_regularizer=0.0):
        super(CPCEncoder, self).__init__()

        self.encoded_dim = encoded_dim
        self.reg = regularizers.l2(weight_regularizer)

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
            self.blocks.append(BatchNormalization())
            self.blocks.append(ReLU())

    def call(self, X):
        for layer in self.blocks:
            X = layer(X)
        return X

    def compute_output_shape(self, input_shape):
        nb_timesteps = input_shape[0] // 160
        return (nb_timesteps, self.encoded_dim)