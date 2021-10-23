import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import Callback

class MoCoModel(Model):
    '''
    Momentum contrastive learning (MoCo) for audio signals,
    implemented as a Keras model.

    "Self-supervised Text-independent Speaker Verification using Prototypical Momentum Contrastive Learning"
    Wei Xia, Chunlei Zhang, Chao Weng, Meng Yu, Dong Yu
    https://arxiv.org/pdf/2012.07178.pdf
    '''

    def __init__(self,
                 encoder_q,
                 encoder_k,
                 config):
        super().__init__()

        self.config = config
        self.reg = regularizers.l2(config.weight_reg)

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.mlp = MLP(self.config.embedding_dim)

        with tf.device("CPU:0"):
            queue_shape = [self.config.queue_size, self.config.embedding_dim]
            self.queue = tf.random.normal(queue_shape)

        update_model_weights_with_ema(self.encoder_q, self.encoder_k, 0.1)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def call(self, X):
        return self.mlp(self.encoder_q(X))

    def train_step(self, data):
        X, _ = data # Discard labels provided by the dataset generator
        # X shape: (batch_size, 300, 40)

        X_1_aug = X
        X_2_aug = tf.identity(X)

        with tf.GradientTape() as tape:
            Z_q = self.encoder_q(X_1_aug, training=True)
            Z_k = self.encoder_k(X_2_aug, training=False)
            # Out shape: (batch_size, encoded_dim)

            Z_q = self.mlp(Z_q, training=True)
            Z_k = self.mlp(Z_k, training=True)
            # Out shape: (batch_size, 512)

            loss, accuracy = moco_loss(Z_q,
                                       Z_k,
                                       self.queue,
                                       self.config.info_nce_temp)

        trainable_params = self.encoder_q.trainable_weights
        trainable_params += self.mlp.trainable_weights

        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        update_model_weights_with_ema(self.encoder_q, self.encoder_k)

        metrics = { 'loss': loss, 'accuracy': accuracy, 'keys': Z_k }
        return metrics

    def test_step(self, data):
        X, _ = data # Discard labels provided by the dataset generator
        
        X_1_aug = X
        X_2_aug = tf.identity(X)

        Z_q = self.encoder_q(X_1_aug, training=False)
        Z_k = self.encoder_k(X_2_aug, training=False)

        Z_q = self.mlp(Z_q, training=False)
        Z_k = self.mlp(Z_k, training=False)

        loss, accuracy = moco_loss(Z_q,
                                   Z_k,
                                   self.queue,
                                   self.config.info_nce_temp)

        return { 'loss': loss, 'accuracy': accuracy }


@tf.function
def moco_loss(Z_q, Z_k, queue, info_nce_temp):
    anchor = Z_q
    pos = tf.stop_gradient(Z_k)
    neg = queue
    # anchor: (B, C), pos: (B, C), neg: (K, C)

    batch_size = tf.shape(anchor)[0]

    # Determine loss
    l_pos = tf.einsum('nc,nc->n', anchor, pos) # Shape: (B)
    l_pos = tf.expand_dims(l_pos, axis=-1)  # Shape: (B, 1)
    l_neg = tf.einsum('nc,ck->nk', anchor, tf.transpose(neg)) # Shape: (B, queue_size)
    logits = tf.concat((l_pos, l_neg), axis=1) # Shape: (B, 1+queue_size)
    logits /= info_nce_temp

    labels = tf.zeros(batch_size, dtype=tf.int32)
    loss = sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)

    # Determine accuracy
    logits_softmax = tf.nn.softmax(logits, axis=1)
    pred_indices = tf.math.argmax(logits_softmax, axis=1, output_type=tf.int32)
    preds_acc = tf.math.equal(pred_indices, labels)
    accuracy = tf.math.count_nonzero(preds_acc, dtype=tf.int32) / batch_size

    return loss, accuracy


def update_model_weights_with_ema(encoder_q, encoder_k, momentum=0.999):
    for v1, v2 in zip(encoder_q.variables, encoder_k.variables):
        v2.assign(momentum * v2 + (1.0 - momentum) * v1)


class MLP(Model):

    def __init__(self, dim):
        super().__init__()

        self.fc1 = Dense(512)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.fc2 = Dense(dim)

    def call(self, X):
        Z = self.fc1(X)
        Z = self.bn1(Z)
        Z = self.relu(Z)
        Z = self.fc2(Z)
        Z = tf.math.l2_normalize(Z, axis=-1)
        return Z


class MoCoUpdateCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_batch_end(self, data, logs=None):
        keys = logs.pop('keys')
        self.model.queue = tf.concat([keys, self.model.queue], axis=0)
        self.model.queue = self.model.queue[:self.model.config.queue_size]