import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization

@tf.function
def off_diagonal(matrix):
    mask = tf.math.logical_not(tf.eye(tf.shape(matrix)[0], dtype=tf.bool))
    return tf.cast(mask, tf.float32) * matrix

class BarlowTwins(Layer):

    def __init__(
        self,
        redundancy_reduction_weight=0.05
    ):
        super().__init__()

        self.l = redundancy_reduction_weight
        
    def call(self, data):
        Z_a, Z_b = data

        N = tf.cast(tf.shape(Z_a)[0], tf.float32)

        Z_a_mean = tf.math.reduce_mean(Z_a, axis=0)
        Z_b_mean = tf.math.reduce_mean(Z_b, axis=0)
        Z_a_std  = tf.math.reduce_std(Z_a, axis=0)
        Z_b_std  = tf.math.reduce_std(Z_b, axis=0)
        
        Z_a = (Z_a - Z_a_mean) / Z_a_std
        Z_b = (Z_b - Z_b_mean) / Z_b_std
        c = (tf.transpose(Z_a) @ Z_b) / N

        loss = tf.math.reduce_sum(tf.math.pow(tf.linalg.diag_part(c) - 1, 2))
        loss += self.l * tf.math.reduce_sum(tf.math.pow(off_diagonal(c), 2))
        return loss