import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

@tf.function
def lim_loss(loss_fn, pos, neg):
    # pos and neg shape: (batch_size, 1)

    batch_size = tf.shape(pos)[0]

    acc = tf.math.count_nonzero(tf.math.greater(pos, neg), dtype=tf.int32) / batch_size

    if loss_fn == 'bce':
        # Prevent numerical instability with log(x)
        epsilon = 1e-07
        pos = tf.clip_by_value(tf.math.sigmoid(pos), epsilon, 1.0 - epsilon)
        neg = tf.clip_by_value(tf.math.sigmoid(neg), epsilon, 1.0 - epsilon)

        loss = tf.math.reduce_mean(tf.math.log(pos))
        loss = loss + tf.math.reduce_mean(tf.math.log(1 - neg))
        return -loss, acc

    elif loss_fn == 'mine':
        loss = tf.math.reduce_mean(pos)
        loss = loss - tf.math.log(tf.math.reduce_mean(tf.math.exp(neg)))
        return -loss, acc

    elif loss_fn == 'nce':
        loss = tf.math.log(tf.math.exp(pos) + tf.math.reduce_sum(tf.math.exp(neg)))
        loss = tf.math.reduce_mean(pos - loss)
        return -loss, acc

    raise Exception('LIM: loss {} is not supported.'.format(loss_fn))

class Discriminator(Model):

    def __init__(self, reg):
        super(Discriminator, self).__init__()

        self.dense1 = Dense(units=256,
                            activation='relu',
                            kernel_regularizer=reg,
                            bias_regularizer=reg)
        self.dense2 = Dense(units=1,
                            kernel_regularizer=reg,
                            bias_regularizer=reg)

    def call(self, X):
        return self.dense2(self.dense1(X))

class LIMModel(Model):

    def __init__(self,
                 encoder,
                 nb_timesteps,
                 loss_fn='bce',
                 context_length=1,
                 weight_regularizer=0.0):
        super(LIMModel, self).__init__()

        self.nb_timesteps = nb_timesteps
        self.loss_fn = loss_fn
        self.context_length = context_length
        self.reg = regularizers.l2(weight_regularizer)

        self.encoder = encoder
        self.discriminator = Discriminator(self.reg)

    def compile(self, optimizer):
        super(LIMModel, self).compile()
        self.optimizer = optimizer

    def call(self, X):
        return self.encoder(X)

    @tf.function
    def extract_chunks(self, X):
        batch_size = tf.shape(X)[0]

        max_idx = self.nb_timesteps - self.context_length + 1
        idx = tf.random.uniform(shape=[3],
                                minval=0,
                                maxval=max_idx,
                                dtype=tf.int32)

        shift = tf.random.uniform(shape=[1],
                                  minval=1,
                                  maxval=batch_size,
                                  dtype=tf.int32)

        C1 = X[:, idx[0]:idx[0]+self.context_length, ...]
        C1 = tf.math.reduce_mean(C1, axis=1)

        C2 = X[:, idx[1]:idx[1]+self.context_length, ...]
        C2 = tf.math.reduce_mean(C2, axis=1)

        CR = X[:, idx[2]:idx[2]+self.context_length, ...]
        CR = tf.math.reduce_mean(CR, axis=1)
        CR = tf.roll(CR, shift=shift[0], axis=0)

        return C1, C2, CR

    def train_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator

        with tf.GradientTape() as tape:
            # X shape: (batch_size, frame_length, 1)

            X_encoded = self.encoder(X, training=True)
            # Out shape: (batch_size, frame_length // 160, encoded_dim)

            C1, C2, CR = self.extract_chunks(X_encoded)
            # Out shape: (batch_size, encoded_dim)

            C1_and_C2 = tf.concat([C1, C2], axis=1)
            C1_and_CR = tf.concat([C1, CR], axis=1)
            # Out shape: (batch_size, 2*encoded_dim)

            pos = self.discriminator(C1_and_C2, training=True)
            neg = self.discriminator(C1_and_CR, training=True)
            # Out shape: (batch_size, 1)

            loss, accuracy = lim_loss(self.loss_fn, pos, neg)
            # Out shape: (batch_size)

        trainable_params = self.encoder.trainable_weights
        trainable_params += self.discriminator.trainable_weights
        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        return { 'loss': loss, 'accuracy': accuracy }

    def test_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator
        
        X_encoded = self.encoder(X, training=False)

        C1, C2, CR = self.extract_chunks(X_encoded)

        C1_and_C2 = tf.concat([C1, C2], axis=1)
        C1_and_CR = tf.concat([C1, CR], axis=1)

        pos = self.discriminator(C1_and_C2, training=False)
        neg = self.discriminator(C1_and_CR, training=False)

        loss, accuracy = lim_loss(self.loss_fn, pos, neg)

        return { 'loss': loss, 'accuracy': accuracy }