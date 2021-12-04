import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sslforslr.modules.VICReg import VICReg
from sslforslr.modules.BarlowTwins import BarlowTwins

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

        self.enable_mlp = config.enable_mlp
        self.enable_mse_clean_aug = config.enable_mse_clean_aug
        self.infonce_loss_factor = config.infonce_loss_factor
        self.vic_reg_factor = config.vic_reg_factor
        self.barlow_twins_factor = config.barlow_twins_factor
        self.mse_clean_aug_factor = config.mse_clean_aug_factor
        self.reg = regularizers.l2(config.weight_reg)

        self.encoder = encoder
        self.mlp = MLP()
        self.infonce_loss = InfoNCELoss()
        self.vic_reg = VICReg(
            config.vic_reg_inv_weight,
            config.vic_reg_var_weight,
            config.vic_reg_cov_weight
        )
        self.barlow_twins = BarlowTwins()

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def call(self, X):
        if len(X.shape) == 4 and self.enable_mse_clean_aug:
            X, _ = self.extract_clean_and_aug(X)
        return self.encoder(X)

    @tf.function
    def get_embeddings(self, X_1, X_2):
        Z_1 = self.encoder(X_1, training=True)
        Z_2 = self.encoder(X_2, training=True)
        if self.enable_mlp:
            Z_1 = self.mlp(Z_1, training=True)
            Z_2 = self.mlp(Z_2, training=True)
        return Z_1, Z_2

    @tf.function
    def extract_clean_and_aug(self, X):
        X_clean, X_aug = tf.split(X, 2, axis=-1)
        X_clean = tf.squeeze(X_clean, axis=-1)
        X_aug = tf.squeeze(X_aug, axis=-1)
        return X_clean, X_aug

    def train_step(self, data):
        X_1_aug, X_2_aug, _ = data
        # X shape: (B, H, W, C) = (B, 40, 200, 1)

        if self.enable_mse_clean_aug:
            X_1_clean, X_1_aug = self.extract_clean_and_aug(X_1_aug)
            X_2_clean, X_2_aug = self.extract_clean_and_aug(X_2_aug)

        with tf.GradientTape() as tape:
            Z_1_aug, Z_2_aug = self.get_embeddings(X_1_aug, X_2_aug)

            loss, accuracy = self.infonce_loss((Z_1_aug, Z_2_aug))
            loss = self.infonce_loss_factor * loss
            loss += self.vic_reg_factor * self.vic_reg((Z_1_aug, Z_2_aug))
            loss += self.barlow_twins_factor * self.barlow_twins((Z_1_aug, Z_2_aug))

            if self.enable_mse_clean_aug:
                Z_1_clean, Z_2_clean = self.get_embeddings(X_1_clean, X_2_clean)
                loss += self.mse_clean_aug_factor * mse_loss(Z_1_clean, Z_1_aug)
                loss += self.mse_clean_aug_factor * mse_loss(Z_2_clean, Z_2_aug)

        trainable_params = self.encoder.trainable_weights
        if self.enable_mlp:
            trainable_params += self.mlp.trainable_weights

        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        return { 'loss': loss, 'accuracy': accuracy }


class MLP(Model):

    def __init__(self):
        super().__init__()

        self.relu = ReLU()

        self.fc1 = Dense(2048)
        self.bn1 = BatchNormalization()

        self.fc2 = Dense(2048)
        self.bn2 = BatchNormalization()

        self.fc3 = Dense(512)

    def call(self, X):
        Z = self.fc1(X)
        Z = self.bn1(Z)
        Z = self.relu(Z)

        Z = self.fc2(Z)
        Z = self.bn2(Z)
        Z = self.relu(Z)

        Z = self.fc3(Z)
        return Z


class InfoNCELoss(Layer):

    def __init__(self):
        super().__init__()

        self.ce = SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM,
                from_logits=True
        )
    
    def call(self, data):
        Z_1, Z_2 = data
        # Shape: (batch_size, encoded_dim)
        
        batch_size = tf.shape(Z_1)[0]
        labels = tf.range(batch_size)

        Z_1 = tf.math.l2_normalize(Z_1, axis=-1)
        Z_2 = tf.math.l2_normalize(Z_2, axis=-1)
        
        # Determine loss
        dot = tf.linalg.matmul(Z_1, Z_2, transpose_b=True)
        dot = dot / 0.07
        loss = self.ce(labels, dot) / tf.cast(batch_size, tf.float32)

        # Determine accuracy
        softmax_dot = tf.nn.softmax(dot, axis=1)
        pred_indices = tf.math.argmax(softmax_dot, axis=1, output_type=tf.int32)
        preds_acc = tf.math.equal(pred_indices, labels)
        accuracy = tf.math.count_nonzero(preds_acc, dtype=tf.int32) / batch_size

        return loss, accuracy


@tf.function
def mse_loss(Z_clean, Z_aug):
    mse = tf.keras.metrics.mean_squared_error(Z_clean, Z_aug)
    return tf.math.reduce_mean(mse)
