import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer

from sslforslr.modules.VICReg import VICReg

class SimCLRModel(Model):
    '''
    A simple framework for contrastive learning (SimCLR) for audio signals,
    implemented as a Keras model.

    "Contrastive Self-Supervised Learning for Text-Independent Speaker Verification"
    Haoran Zhang, Yuexian Zou, Helin Wang
    '''

    def __init__(self,
                 encoder,
                 config):
        super().__init__()

        self.loss_factor = config.loss_factor
        self.vic_reg_factor = config.vic_reg_factor
        self.reg = regularizers.l2(config.weight_reg)

        self.encoder = encoder
        self.nce_loss = AngularPrototypicalLoss(self.reg)
        self.vic_reg = VICReg()

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def call(self, X):
        return tf.math.l2_normalize(self.encoder(X), axis=-1)

    def train_step(self, data):
        X_1_aug, X_2_aug, _ = data
        # X shape: (B, H, W, C) = (B, 40, 200, 1)

        with tf.GradientTape() as tape:
            Z_1_aug = self.encoder(X_1_aug, training=True)
            Z_2_aug = self.encoder(X_2_aug, training=True)
            # Z shape: (B, encoded_dim)

            loss, accuracy = self.nce_loss((Z_1_aug, Z_2_aug), training=True)
            loss = self.loss_factor * loss
            loss += self.vic_reg_factor * self.vic_reg((Z_1_aug, Z_2_aug), training=True)

        trainable_params = self.encoder.trainable_weights
        trainable_params += self.nce_loss.trainable_weights

        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        return { 'loss': loss, 'accuracy': accuracy }

    def test_step(self, data):
        X_1_aug, X_2_aug, _ = data
        
        Z_1_aug = self.encoder(X_1_aug, training=False)
        Z_2_aug = self.encoder(X_2_aug, training=False)

        loss, accuracy = self.nce_loss((Z_1_aug, Z_2_aug), training=False)
        loss = self.loss_factor * loss
        loss += self.vic_reg_factor * self.vic_reg((Z_1_aug, Z_2_aug), training=False)

        return { 'loss': loss, 'accuracy': accuracy }


class AngularPrototypicalLoss(Layer):

    def __init__(self, reg, init_w=10.0, init_b=-5.0):
        super().__init__()

        self.w = self.add_weight(
            name='w',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(init_w),
            trainable=True,
            regularizer=reg)

        self.b = self.add_weight(
            name='b',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(init_b),
            trainable=True,
            regularizer=reg)

    def call(self, data):
        Z_1_aug, Z_2_aug = data
        # Shape: (batch_size, encoded_dim)
    
        batch_size = tf.shape(Z_1_aug)[0]

        # Normalize embeddings for cosine distance
        Z_1_aug = tf.math.l2_normalize(Z_1_aug, axis=-1)
        Z_2_aug = tf.math.l2_normalize(Z_2_aug, axis=-1)

        # Determine loss
        dot = tf.linalg.matmul(Z_1_aug, Z_2_aug, transpose_b=True)

        # Angular prototypical loss
        w_clamped = tf.clip_by_value(self.w, clip_value_min=1e-6, clip_value_max=1e+6)
        dot = w_clamped * dot + self.b
        
        log_softmax_dot = tf.nn.log_softmax(dot, axis=0)
        diag = tf.linalg.tensor_diag_part(log_softmax_dot)
        loss = -tf.math.reduce_mean(diag)

        # Determine accuracy
        softmax_dot = tf.nn.softmax(dot, axis=0)
        pred_indices = tf.math.argmax(softmax_dot, axis=0, output_type=tf.int32)
        preds_acc = tf.math.equal(pred_indices, tf.range(0, batch_size))
        accuracy = tf.math.count_nonzero(preds_acc, dtype=tf.int32) / batch_size

        return loss, accuracy


@tf.function
def channel_loss(Z_clean, Z_aug):
    mse = tf.keras.metrics.mean_squared_error(Z_clean, Z_aug)
    return tf.math.reduce_mean(mse)