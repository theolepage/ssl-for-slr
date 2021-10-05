import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer

import torch
import torchaudio

def sample_frames_from_utterance(utterance):
    utterance_length = len(utterance)
    frame_length = 32000

    pivot = np.random.randint(frame_length, utterance_length - frame_length + 1)

    offset = (pivot - frame_length) // 2
    first_frame = utterance[offset:offset+frame_length]

    offset = (utterance_length - pivot - frame_length) // 2
    second_frame = utterance[offset:offset+frame_length]

    return first_frame, second_frame


def wav_augment(wav):
    return wav


def spec_augment(spec):
    return spec


def extract_mfcc(audio):
    mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(audio.T),
                                            num_ceps=30,
                                            num_mel_bins=30)
    mfcc = torchaudio.transforms.SlidingWindowCmn(norm_vars=False)(mfcc)
    return mfcc.numpy()


def training_data_pipeline(audio):
    X_1_clean, X_2_clean, X_1_aug, X_2_aug = [], [], [], []
    for i in range(len(audio)):
        x_1, x_2 = sample_frames_from_utterance(audio[i])

        x_1_clean, x_2_clean = extract_mfcc(x_1), extract_mfcc(x_2)
        X_1_clean.append(x_1_clean)
        X_2_clean.append(x_2_clean)
        
        x_1_aug, x_2_aug = wav_augment(x_1), wav_augment(x_2)
        x_1_aug, x_2_aug = extract_mfcc(x_1_aug), extract_mfcc(x_2_aug)
        x_1_aug, x_2_aug = spec_augment(x_1_aug), spec_augment(x_2_aug)
        X_1_aug.append(x_1_aug)
        X_2_aug.append(x_2_aug)

    # X1 = np.arange(64*4000*40).reshape((64, 4000, 40, 1)).astype(np.float32)
    return np.array(X_1_clean), \
           np.array(X_2_clean), \
           np.array(X_1_aug),   \
           np.array(X_2_aug)


class SimCLRModel(Model):
    '''
    A simple framework for contrastive learning (SimCLR) for audio signals,
    implemented as a Keras model.

    "Contrastive Self-Supervised Learning for Text-Independent Speaker Verification"
    Haoran Zhang, Yuexian Zou, Helin Wang1
    '''

    def __init__(self,
                 encoder,
                 channel_loss_factor,
                 weight_regularizer=0.0):
        super().__init__()

        self.channel_loss_factor = channel_loss_factor
        self.reg = regularizers.l2(weight_regularizer)

        self.encoder = encoder
        self.simclr_loss = AngularPrototypicalLoss(self.reg)

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, X):
        return X # FIXME: self.encoder(X)

    def train_step(self, data):
        X, _ = data # Discard labels provided by the dataset generator
        # X shape: (batch_size, frame_length, 40, 1)

        # FIXME: replace by dataset generator
        audio = np.arange(64*64000*1).reshape((64, 64000, 1)).astype(np.float32)
        X_1_clean, X_2_clean, X_1_aug, X_2_aug = training_data_pipeline(audio)


        with tf.GradientTape() as tape:
            Z_1_clean = self.encoder(X_1_clean, training=True)
            Z_2_clean = self.encoder(X_2_clean, training=True)
            Z_1_aug   = self.encoder(X_1_aug,   training=True)
            Z_2_aug   = self.encoder(X_2_aug,   training=True)
            # Out shape: (batch_size, encoded_dim)

            loss, accuracy = self.simclr_loss([Z_1_aug, Z_2_aug])
            loss += self.channel_loss_factor * channel_loss(Z_1_clean, Z_1_aug)
            loss += self.channel_loss_factor * channel_loss(Z_2_clean, Z_2_aug)

        trainable_params = self.encoder.trainable_weights

        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        return { 'loss': loss, 'accuracy': accuracy }

    def test_step(self, data):
        X, _ = data # Discard labels provided by the dataset generator
        
        # FIXME: replace by dataset generator
        audio = np.arange(64*64000*1).reshape((64, 64000, 1)).astype(np.float32)
        X_1_clean, X_2_clean, X_1_aug, X_2_aug = training_data_pipeline(audio)

        Z_1_clean = self.encoder(X_1_clean, training=False)
        Z_2_clean = self.encoder(X_2_clean, training=False)
        Z_1_aug   = self.encoder(X_1_aug,   training=False)
        Z_2_aug   = self.encoder(X_2_aug,   training=False)

        loss, accuracy = self.simclr_loss([Z_1_aug, Z_2_aug])
        loss += self.channel_loss_factor * channel_loss(Z_1_clean, Z_1_aug)
        loss += self.channel_loss_factor * channel_loss(Z_2_clean, Z_2_aug)

        return { 'loss': loss, 'accuracy': accuracy }


class AngularPrototypicalLoss(Layer):
    def __init__(self, reg):
        super().__init__()

        self.w = self.add_weight(
            name='w',
            shape=(1,),
            initializer="random_normal",
            trainable=True,
            regularizer=reg)

        self.b = self.add_weight(
            name='b',
            shape=(1,),
            initializer="random_normal",
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
        dot = self.w * dot + self.b # Angular prototypical loss
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