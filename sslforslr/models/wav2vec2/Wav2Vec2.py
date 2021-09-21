import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, LayerNormalization, Dropout
from tensorflow_addons.layers import GELU
from tensorflow.keras import regularizers
from tensorflow.keras import losses

from .Wav2Vec2Config import Wav2Vec2Config
from sslforslr.modules import TransformerEncoder, VectorQuantizer

class Wav2Vec2Model(Model):
    '''
    wav2vec 2.0 implemented as a Keras model.

    "wav2vec 2.0: A Framework for Self-Supervised Learning
    of Speech Representations"
    Alexei Baevski et al.
    https://arxiv.org/pdf/2006.11477.pdf
    '''

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()

        self.config = config

        self.encoder = Wav2Vec2Encoder(config.encoder_conv_layers)
        self.quantizer = VectorQuantizer(input_dim=config.encoder_dim,
                                         dim=config.quantizer_dim,
                                         nb_groups=config.quantizer_nb_groups,
                                         nb_vars=config.quantizer_nb_vars,
                                         temp=config.quantizer_temp)
        self.transformer = TransformerEncoder(config)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(config.dropout)
        self.proj_Z = Dense(config.transformer_dim)
        self.proj_Q = Dense(config.quantizer_dim)
        self.proj_C = Dense(config.quantizer_dim)

        self.mask_weights = self.add_weight(
            name='mask_weights',
            shape=(self.config.transformer_dim,),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            trainable=True
        )

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, X):
        Z = self.encoder(X, training=False)
        Z = self.layer_norm(Z)
        Z = self.proj_Z(Z)
        Z = self.dropout(Z)
        C = self.transformer(Z, training=False)
        # C shape: (B, T, transformer_dim)
        return C

    @tf.function
    def sample_negatives(self, Q):
        # Q shape: (B, T, self.config.quantizer_dim)

        B = tf.shape(Q)[0]
        T = tf.shape(Q)[1]
        F = tf.shape(Q)[2]

        nb_negatives = self.config.nb_negatives

        shift_utterances = tf.range(B)
        shift_utterances = tf.roll(shift_utterances, shift=-1, axis=0)
        shift_utterances = tf.repeat(shift_utterances, T * nb_negatives) * T
        shift_utterances = tf.reshape(shift_utterances, (B, -1))

        idxs = tf.random.uniform(shape=[B, nb_negatives * T],
                                        minval=0,
                                        maxval=T,
                                        dtype=tf.int32)

        idxs = idxs + shift_utterances
        idxs = tf.reshape(idxs, [-1])

        Q = tf.reshape(Q, (B * T, F))

        Q_negs = tf.gather(Q, idxs)
        Q_negs = tf.reshape(Q_negs, (B, T, nb_negatives, F))
        Q_negs = tf.transpose(Q_negs, perm=[2, 0, 1, 3])
        return Q_negs

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
    def compute_loss(self, C, Q, Q_negs, diversity_loss, features_loss):
        # Q      shape: (B, T, F)
        # Q_negs shape: (nb_negatives, B, T, F)

        B = tf.shape(Q)[0]
        T = tf.shape(Q)[1]

        Q = tf.expand_dims(Q, axis=0)
        targets = tf.concat([Q, Q_negs], axis=0)
        
        dist = losses.CosineSimilarity(axis=-1,
                                       reduction=losses.Reduction.NONE)(C, targets)
        dist = dist / self.config.cos_dist_temp
        # dist shape: (nb_negatives + 1, B, T)
        dist = tf.reshape(dist, (B * T, -1))

        loss = tf.nn.log_softmax(dist, axis=-1)
        loss = loss[:, 0] # Keep first column as it represents Q positive
        loss = tf.math.reduce_mean(loss)

        # Add additional losses: features penalty, codebook penalty
        d_loss = self.config.diversity_loss_weight * diversity_loss
        f_loss = self.config.features_loss_weight * features_loss

        return -loss + d_loss + f_loss

    def train_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator

        with tf.GradientTape() as tape:
            # X shape: (B, T, 1)

            Z = self.encoder(X, training=True)
            # Z shape: (B, T, encoded_dim)

            features_loss = tf.math.reduce_mean(tf.math.pow(Z, 2))

            Z = self.layer_norm(Z)
            Z_unmasked = tf.identity(Z)
            
            Z = self.proj_Z(Z)
            Z = self.dropout(Z)

            Z_unmasked = self.dropout(Z_unmasked)

            Z = self.apply_mask(Z)
            
            C = self.transformer(Z, training=True)
            # C shape: (B, T, transformer_dim)

            Q, diversity_loss = self.quantizer(Z_unmasked, training=True)
            # Q shape: (B, T, quantizer_dim)

            # When creating next Dense layer a static shape is required (quantizer_dim)
            Q.set_shape((Q.shape[0], Q.shape[1], self.config.quantizer_dim))
            Q = self.proj_Q(Q)
            # Q shape: (B, T, quantizer_dim)

            Q_negs = self.sample_negatives(Q)

            C = self.proj_C(C)

            loss = self.compute_loss(C, Q, Q_negs, diversity_loss, features_loss)

        trainable_params = self.trainable_weights
        trainable_params += self.encoder.trainable_weights
        trainable_params += self.transformer.trainable_weights
        trainable_params += self.quantizer.trainable_weights
        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        return { 'loss': loss }

    def test_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator
        
        Z = self.encoder(X, training=False)

        features_loss = tf.math.reduce_mean(tf.math.pow(Z, 2))

        Z = self.layer_norm(Z)
        Z_unmasked = tf.identity(Z)
        
        Z = self.proj_Z(Z)
        Z = self.dropout(Z)

        Z_unmasked = self.dropout(Z_unmasked)

        Z = self.apply_mask(Z)
        
        C = self.transformer(Z, training=False)

        Q, diversity_loss = self.quantizer(Z_unmasked, training=False)

        # When creating next Dense layer a static shape is required (quantizer_dim)
        Q.set_shape((Q.shape[0], Q.shape[1], self.config.quantizer_dim))
        Q = self.proj_Q(Q)

        Q_negs = self.sample_negatives(Q)

        C = self.proj_C(C)

        loss = self.compute_loss(C, Q, Q_negs, diversity_loss, features_loss)

        return { 'loss': loss }


class Wav2Vec2Encoder(Model):

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