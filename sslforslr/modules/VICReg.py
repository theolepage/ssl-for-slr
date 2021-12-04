import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.function
def off_diagonal(matrix):
    mask = tf.math.logical_not(tf.eye(tf.shape(matrix)[0], dtype=tf.bool))
    return tf.cast(mask, tf.float32) * matrix

class VICReg(Layer):

    def __init__(
        self,
        inv_weight=1.0,
        var_weight=1.0,
        cov_weight=0.04
    ):
        super().__init__()

        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def call(self, data):
        X_a, X_b = data

        N = tf.cast(tf.shape(X_a)[0], tf.float32)
        D = tf.cast(tf.shape(X_a)[1], tf.float32)

        X_a_mean, X_a_var = tf.nn.moments(X_a, axes=[0])
        X_b_mean, X_b_var = tf.nn.moments(X_b, axes=[0])

        # Invariance loss
        inv_loss = tf.keras.metrics.mean_squared_error(X_a, X_b)
        inv_loss = tf.math.reduce_mean(inv_loss)

        # Variance loss
        X_a_std = tf.math.sqrt(X_a_var + 1e-04)
        X_b_std = tf.math.sqrt(X_b_var + 1e-04)
        var_loss = tf.math.reduce_mean(tf.nn.relu(1 - X_a_std))
        var_loss += tf.math.reduce_mean(tf.nn.relu(1 - X_b_std))

        # Covariance loss
        X_a = X_a - X_a_mean
        X_b = X_b - X_b_mean
        X_a_cov = (tf.transpose(X_a) @ X_a) / (N - 1)
        X_b_cov = (tf.transpose(X_b) @ X_b) / (N - 1)
        cov_loss = tf.math.reduce_sum(tf.math.pow(off_diagonal(X_a_cov), 2)) / D
        cov_loss += tf.math.reduce_sum(tf.math.pow(off_diagonal(X_b_cov), 2)) / D

        loss = self.inv_weight * inv_loss
        loss += self.var_weight * var_loss
        loss += self.cov_weight * cov_loss
        return loss