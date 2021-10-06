import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, LayerNormalization, Dropout
from tensorflow_addons.layers import GELU
from tensorflow.keras import regularizers
from tensorflow.keras import losses

from .VQWav2VecConfig import VQWav2VecConfig
from sslforslr.modules import TransformerEncoder, VectorQuantizer

class VQWav2VecModel(Model):
    '''
    vq-wav2vec implemented as a Keras model.

    It combines the principle of CPC and a quantization module.

    "vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations"
    Alexei Baevski, Steffen Schneider, Michael Auli
    https://arxiv.org/pdf/1910.05453.pdf
    '''

    def __init__(self, config: VQWav2VecConfig):
        super().__init__()

        self.config = config

        self.encoder = VQWav2VecEncoder(config.encoder_conv_layers)
        self.quantizer = VectorQuantizer(input_dim=config.encoder_dim,
                                         dim=config.quantizer_dim,
                                         nb_groups=config.quantizer_nb_groups,
                                         nb_vars=config.quantizer_nb_vars,
                                         temp=config.quantizer_temp)
        self.transformer = TransformerEncoder(config)
        self.predictor = Predictor(config.quantizer_dim,
                                   config.nb_timesteps_to_predict)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(config.dropout)
        self.proj_before_transformer = Dense(config.transformer_dim)

        self.mask_weights = self.add_weight(
            name='mask_weights',
            shape=(self.config.transformer_dim,),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            trainable=True
        )

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def call(self, X):
        Z = self.encoder(X, training=False)
        Z = self.layer_norm(Z)
        Z = self.dropout(Z)

        Q, _ = self.quantizer(Z, training=False)
        Q.set_shape((Q.shape[0],
                     Q.shape[1],
                     self.config.quantizer_dim))
        
        Q = self.proj_before_transformer(Q)
        C = self.transformer(Q, training=False)
        C = C[:, -1, :] # Keep only last timestep

        return C

    def get_mask_indices(self, Z):
        B, T, F = Z.numpy().shape
    
        num_mask = int(
            (self.config.mask_prob * T)
            / float(self.config.mask_length)
            + np.random.rand()
        )

        indices = []
        for i in range(B):
            mask_idx = np.random.choice(T - self.config.mask_length,
                                        num_mask,
                                        replace=False)
            mask_idx = np.asarray(
                [
                    mask_idx[j] + offset
                    for j in range(num_mask)
                    for offset in range(self.config.mask_length)
                ]
            )
            mask_idx = np.unique(mask_idx[mask_idx < T])

            # FIXME: better vectorization
            for j in mask_idx:
                indices.append([i, j])

        return tf.convert_to_tensor(indices)

    @tf.function
    def apply_mask(self, Z):
        mask_indices = tf.py_function(func=self.get_mask_indices,
                                      inp=[Z],
                                      Tout=tf.int32)
        nb_masked_timesteps = tf.shape(mask_indices)[0]
        mask_updates = tf.repeat(self.mask_weights, [nb_masked_timesteps])
        mask_updates = tf.reshape(mask_updates, (nb_masked_timesteps, -1))
        Z = tf.tensor_scatter_nd_update(Z, mask_indices, mask_updates)
        return Z

    @tf.function
    def compute_loss(self, predictions, Q_future, diversity_loss, features_loss):
        # preds    shape: (B, nb_timesteps_to_predict, quantizer_dim)
        # Q_future shape: (B, nb_timesteps_to_predict, quantizer_dim)

        batch_size = tf.shape(predictions)[0]

        losses = tf.zeros((batch_size))

        for t in range(self.config.nb_timesteps_to_predict):
            dot = tf.linalg.matmul(Q_future[:, t, :],
                                    predictions[:, t, :],
                                    transpose_b=True)
            
            # Determine loss
            log_softmax_dot = tf.nn.log_softmax(dot, axis=0)
            diag = tf.linalg.tensor_diag_part(log_softmax_dot)
            losses += diag

        losses /= tf.cast(self.config.nb_timesteps_to_predict, dtype=tf.float32)

        # Determine accuracy
        softmax_dot = tf.nn.softmax(dot, axis=0)
        pred_indices = tf.math.argmax(softmax_dot, axis=0, output_type=tf.int32)
        preds_acc = tf.math.equal(pred_indices, tf.range(0, batch_size))
        accuracies = tf.math.count_nonzero(preds_acc, dtype=tf.int32) / batch_size

        # Compute the average loss and accuracy across all batches
        loss = tf.math.reduce_mean(losses)
        accuracy = tf.math.reduce_mean(accuracies)

        # Add additional losses: features penalty, codebook penalty
        d_loss = self.config.diversity_loss_weight * diversity_loss
        f_loss = self.config.features_loss_weight * features_loss

        return -loss + d_loss + f_loss, accuracy

    def train_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator

        with tf.GradientTape() as tape:
            # X shape: (B, T, 1)

            Z = self.encoder(X, training=True)
            # Z shape: (B, T, encoded_dim)

            features_loss = tf.math.reduce_mean(tf.math.pow(Z, 2))
            
            Z = self.layer_norm(Z)
            Z = self.dropout(Z)

            Q, diversity_loss = self.quantizer(Z, training=True)
            Q.set_shape((Q.shape[0],
                         Q.shape[1],
                         self.config.quantizer_dim))
            # Q shape: (B, T, quantizer_dim)

            # Split past and future timesteps
            Q_past = Q[:, 0:self.config.nb_timesteps_for_context, ...]
            Q_future = Q[:, self.config.nb_timesteps_for_context:, ...]

            # Apply mask on Q_past and determine context C from past timesteps
            Q_past = self.proj_before_transformer(Q_past)
            Q_past = self.apply_mask(Q_past)
            C = self.transformer(Q_past, training=True)
            C = C[:, -1, :] # Keep only last timestep
            # C shape: (B, transformer_dim)

            # Compute predictions with C
            preds = self.predictor(C, training=True)
            # preds shape: (B, nb_timesteps_to_predict, quantizer_dim)

            # Contrastive loss between predictions and Q_future
            loss, accuracy = self.compute_loss(preds,
                                               Q_future,
                                               diversity_loss,
                                               features_loss)

        trainable_params = self.trainable_weights
        trainable_params += self.encoder.trainable_weights
        trainable_params += self.transformer.trainable_weights
        trainable_params += self.predictor.trainable_weights
        trainable_params += self.quantizer.trainable_weights
        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        return { 'loss': loss, 'accuracy': accuracy }

    def test_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator
        
        Z = self.encoder(X, training=True)

        features_loss = tf.math.reduce_mean(tf.math.pow(Z, 2))
        
        Z = self.layer_norm(Z)
        Z = self.dropout(Z)

        Q, diversity_loss = self.quantizer(Z, training=False)
        Q.set_shape((Q.shape[0],
                     Q.shape[1],
                     self.config.quantizer_dim))

        # Split past and future timesteps
        Q_past = Q[:, 0:self.config.nb_timesteps_for_context, ...]
        Q_future = Q[:, self.config.nb_timesteps_for_context:, ...]

        # Apply mask on Q_past and determine context C from past timesteps
        Q_past = self.proj_before_transformer(Q_past)
        Q_past = self.apply_mask(Q_past)
        C = self.transformer(Q_past, training=False)
        C = C[:, -1, :] # Keep only last timestep

        # Compute predictions with C
        preds = self.predictor(C, training=False)

        # Contrastive loss between predictions and Q_future
        loss, accuracy = self.compute_loss(preds,
                                           Q_future,
                                           diversity_loss,
                                           features_loss)

        return { 'loss': loss, 'accuracy': accuracy }


class VQWav2VecEncoder(Model):

    def __init__(self, config):
        super().__init__()

        conv_layers = eval(config)

        self.layers_ = []
        for dim, size, stride in conv_layers:
            self.layers_.append(Conv1D(dim, size, strides=stride, padding='same'))
            self.layers_.append(LayerNormalization())
            self.layers_.append(GELU())

    def call(self, X):
        for layer in self.layers_:
            X = layer(X)
        return X


class Predictor(Model):

    def __init__(self, encoded_dim, nb_timesteps_to_predict):
        super(Predictor, self).__init__()

        self.layers_ = []
        for i in range(nb_timesteps_to_predict):
            self.layers_.append(Dense(units=encoded_dim))

    def call(self, context):
        predictions = []
        for layer in self.layers_:
            predictions.append(layer(context))

        return tf.stack(predictions, axis=1)