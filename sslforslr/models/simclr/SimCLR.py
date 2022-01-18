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
        self.infonce_loss_factor = config.infonce_loss_factor
        self.vic_reg_factor = config.vic_reg_factor
        self.barlow_twins_factor = config.barlow_twins_factor
        self.reg = regularizers.l2(config.weight_reg)

        self.representations_loss_vic = config.representations_loss_vic
        self.representations_loss_nce = config.representations_loss_nce
        self.embeddings_loss_vic = config.embeddings_loss_vic
        self.embeddings_loss_nce = config.embeddings_loss_nce

        self.encoder = encoder
        self.mlp = MLP(config.mlp_dim)
        self.infonce_loss = InfoNCELoss()
        self.vic_reg = VICReg(
            config.vic_reg_inv_weight,
            config.vic_reg_var_weight,
            config.vic_reg_cov_weight
        )
        self.barlow_twins = BarlowTwins(config.barlow_twins_lambda)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def call(self, X):
        return self.encoder(X)

    @tf.function
    def representations_loss(self, Z_1, Z_2):
        loss, accuracy = 0, 0
        if self.representations_loss_nce:
            loss, accuracy = self.infonce_loss((Z_1, Z_2))
            loss = self.infonce_loss_factor * loss
        if self.representations_loss_vic:
            loss += self.vic_reg_factor * self.vic_reg((Z_1, Z_2))
        return loss, accuracy

    @tf.function
    def embeddings_loss(self, Z_1, Z_2):
        loss, accuracy = 0, 0
        if self.embeddings_loss_nce:
            loss, accuracy = self.infonce_loss((Z_1, Z_2))
            loss = self.infonce_loss_factor * loss
        if self.embeddings_loss_vic:
            loss += self.vic_reg_factor * self.vic_reg((Z_1, Z_2))
        return loss, accuracy

    def train_step(self, data):
        X_1, X_2, _ = data
        # X shape: (B, H, W, C) = (B, 40, 200, 1)

        with tf.GradientTape() as tape:
            Z_1 = self.encoder(X_1, training=True)
            Z_2 = self.encoder(X_2, training=True)
            representations_loss, representations_accuracy = self.representations_loss(
                Z_1,
                Z_2
            )

            if self.enable_mlp:
                Z_1 = self.mlp(Z_1, training=True)
                Z_2 = self.mlp(Z_2, training=True)
                embeddings_loss, embeddings_accuracy = self.embeddings_loss(
                    Z_1,
                    Z_2
                )

        # Apply representations loss
        params = self.encoder.trainable_weights
        grads = tape.gradient(representations_loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        # Aplly embeddings loss
        params = self.encoder.trainable_weights
        params += self.mlp.trainable_weights
        grads = tape.gradient(embeddings_loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return {
            'representations_loss': representations_loss,
            'representations_accuracy': representations_accuracy,
            'embeddings_loss': embeddings_loss,
            'embeddings_accuracy': embeddings_accuracy
        }


class MLP(Model):

    def __init__(self, dim):
        super().__init__()

        self.relu = ReLU()

        self.fc1 = Dense(dim)
        self.bn1 = BatchNormalization()

        self.fc2 = Dense(dim)
        self.bn2 = BatchNormalization()

        self.fc3 = Dense(dim)

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