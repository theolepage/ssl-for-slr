import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense

def gumbel_softmax_sample(logits, temperature, eps=1e-20):
    U = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
    n = -tf.math.log(-tf.math.log(U + eps) + eps)
    return tf.nn.softmax((logits + n) / temperature)

def st_gumbel_softmax(logits, temperature):
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = tf.one_hot(tf.argmax(y, axis=-1),
                        tf.shape(logits)[-1],
                        dtype=y.dtype)
    return tf.stop_gradient(y_hard - y) + y

class VectorQuantizer(Layer):

    def __init__(self, input_dim, dim, nb_groups, nb_vars, temp):
        super().__init__()

        self.input_dim = input_dim
        self.nb_groups = nb_groups
        self.nb_vars = nb_vars

        assert dim % nb_groups == 0
        self.dim = dim // nb_groups

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

        self.vars = self.add_weight(
            name='quantizer_vars_weights',
            shape=(1, self.nb_groups * self.nb_vars, self.dim),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            trainable=True
        )

        self.proj = Dense(self.nb_groups * self.nb_vars)

    def update_temp(self, epoch):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** epoch, self.min_temp
        )

    def call(self, X, training):
        B = tf.shape(X)[0]
        T = tf.shape(X)[1]
        C = tf.shape(X)[2]

        # When creating next Dense layer a static shape is required (self.input_dim)
        X = tf.reshape(X, (-1, self.input_dim))
        # shape: (B * T, C)
        
        X = self.proj(X)
        # shape: (B * T, C) -> (B * T, self.nb_groups * self.nb_vars)

        X = tf.reshape(X, (B * T * self.nb_groups, -1))
        # shape: (B * T * self.nb_groups, self.nb_vars)

        # Compute diversity loss
        avg_probs = tf.reshape(X, (B * T, self.nb_groups, -1))
        avg_probs = tf.nn.softmax(avg_probs, axis=-1)
        avg_probs = tf.math.reduce_mean(avg_probs, axis=0)
        loss = -tf.math.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-7), axis=-1)
        GV = self.nb_groups * self.nb_vars
        loss = GV * tf.math.reduce_sum(tf.math.exp(loss)) / GV

        X = st_gumbel_softmax(X, temperature=self.curr_temp)

        X = tf.reshape(X, (B * T, -1))
        # shape: (B * T, self.nb_groups * self.nb_vars)

        X = tf.expand_dims(X, axis=-1) * self.vars
        # shape: (B * T, self.nb_groups * self.nb_vars, self.dim)

        X = tf.reshape(X, (B * T, self.nb_groups, self.nb_vars, -1))
        # shape: (B * T, self.nb_groups, self.nb_vars, self.dim)

        X = tf.math.reduce_sum(X, axis=-2) # average over vars dim
        
        X = tf.reshape(X, (B, T, -1))
        # shape: (B, T, self.nb_groups * self.dim)

        return X, loss