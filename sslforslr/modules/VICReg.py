import tensorflow as tf

@tf.function
def off_diagonal(matrix):
    mask = tf.math.logical_not(tf.eye(tf.shape(matrix)[0], dtype=tf.bool))
    return tf.cast(mask, tf.float32) * matrix

class VICReg(Layer):

    def __init__(self, reg, lamda=25, mu=25, nu=1):
        super().__init__()

        self.lamda = lamda
        self.mu = mu
        self.nu = nu

    def call(self, data):
        X_a, X_b = data

        N, D = tf.shape(X_a)

        X_a_mean, X_a_var = tf.nn.moments(X_a, axes=[0])
        X_b_mean, X_b_var = tf.nn.moments(X_b, axes=[0])

        # Invariance loss
        sim_loss = tf.keras.metrics.mean_squared_error(X_a, X_b)
        sim_loss = tf.math.reduce_mean(sim_loss)

        # Variance loss
        X_a_std = tf.math.sqrt(X_a_var + 1e-04)
        X_b_std = tf.math.sqrt(X_b_var + 1e-04)
        std_loss = tf.math.reduce_mean(tf.nn.relu(1 - X_a_std))
        std_loss += tf.math.reduce_mean(tf.nn.relu(1 - X_b_std))

        # Covariance loss
        X_a = X_a - X_a_mean
        X_b = X_b - X_b_mean
        X_a_cov = (tf.transpose(X_a) @ X_a) / (N - 1)
        X_b_cov = (tf.transpose(X_b) @ X_b) / (N - 1)
        cov_loss = tf.math.reduce_sum(tf.math.pow(off_diagonal(X_a_cov), 2)) / D
        cov_loss += tf.math.reduce_sum(tf.math.pow(off_diagonal(X_b_cov), 2)) / D

        loss = self.lamda * sim_loss
        loss += self.mu * std_loss
        loss += self.nu * cov_loss
        return loss