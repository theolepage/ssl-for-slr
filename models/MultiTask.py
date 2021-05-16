# Add higher directory to python modules path
import sys
sys.path.append("..")

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras import regularizers

from .CPC import CPCModel
from .CPC import cpc_loss
from .LIM import LIMModel
from .LIM import lim_loss
from utils.create_model import create_model

class MFCCWorker(Model):

    def __init__(self,
                 weight_regularizer,
                 loss_scaler,
                 sample_frequency=16000,
                 nb_coefficients=20,
                 hop_length=160):
        super(MFCCWorker, self).__init__()

        self.reg = regularizers.l2(weight_regularizer)
        self.loss_scaler = loss_scaler

        self.sample_frequency = sample_frequency
        self.nb_coefficients = nb_coefficients
        self.hop_length = hop_length

        self.conv1 = Conv1D(filters=256,
                            kernel_size=1,
                            padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        
        # PReLU shared_axes option implies that one parameter
        # per channel will be learned.
        self.activation1 = PReLU(shared_axes=[1]) 

        self.last_conv = Conv1D(filters=nb_coefficients,
                                kernel_size=1,
                                padding='same',
                                kernel_regularizer=self.reg,
                                bias_regularizer=self.reg)

    def call(self, X):
        X = self.conv1(X)
        X = self.activation1(X)
        X = self.last_conv(X)
        return X

    def get_target(self, X):
        max_frames = X.shape[1] // self.hop_length
        res = np.empty((X.shape[0], max_frames, self.nb_coefficients))

        for i in range(X.shape[0]):
            mfcc = librosa.feature.mfcc(X[i].flatten(),
                                        sr=self.sample_frequency,
                                        hop_length=self.hop_length).T
            res[i] = mfcc[:max_frames, :]
            
        return res

    def compute_loss(self, Y, Y_pred):
        return MeanSquaredError()(Y, Y_pred) * self.loss_scaler

class WaveformWorkerBlock(Layer):

    def __init__(self, filters, kernel_size, stride, reg, **kwargs):
        super(WaveformWorkerBlock, self).__init__(**kwargs)

        self.conv = Conv1DTranspose(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    kernel_regularizer=reg,
                                    bias_regularizer=reg)
        # self.normalization = BatchNormalization(center=False, scale=False)
        # self.activation = PReLU(shared_axes=[1])

    def call(self, X):
        X = self.conv(X)
        # X = self.normalization(X)
        # X = self.activation(X)
        return X

class WaveformWorker(Model):

    def __init__(self, weight_regularizer, loss_scaler):
        super(WaveformWorker, self).__init__()

        self.reg = regularizers.l2(weight_regularizer)
        self.loss_scaler = loss_scaler

        self.nb_filters = [256, 256, 128, 128, 128, 64]
        self.kernel_sizes = [2, 2, 2, 2, 2, 5]
        self.strides = [2, 2, 2, 2, 2, 5]

        self.blocks = []
        for i, (f, w, s) in enumerate(zip(self.nb_filters,
                                          self.kernel_sizes,
                                          self.strides)):
            self.blocks.append(WaveformWorkerBlock(f, w, s, self.reg))

        self.conv1 = Conv1D(filters=64,
                            kernel_size=1,
                            padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        self.activation1 = PReLU(shared_axes=[1]) 

        self.last_conv = Conv1D(filters=1,
                                kernel_size=1,
                                kernel_regularizer=self.reg,
                                bias_regularizer=self.reg)

    def call(self, X):
        for block in self.blocks:
            X = block(X)
        X = self.conv1(X)
        X = self.activation1(X)
        X = self.last_conv(X)
        return X

    def get_target(self, X):
        return X

    def compute_loss(self, Y, Y_pred):
        return MeanAbsoluteError()(Y, Y_pred) * self.loss_scaler

class CPCWorker(Model):

    def __init__(self,
                 cpc,
                 loss_scaler):
        super(CPCWorker, self).__init__()

        self.cpc = cpc
        self.loss_scaler = loss_scaler

    def call(self, X_encoded):
        X_past_encoded = X_encoded[:, 0:self.cpc.nb_timesteps_for_context, ...]
        X_future_encoded = X_encoded[:, self.cpc.nb_timesteps_for_context:, ...]

        X_past_context = self.cpc.ar(X_past_encoded, training=True)

        predictions = self.cpc.predictor(X_past_context, training=True)

        return predictions, X_future_encoded

    def compute_loss(self, Y, Y_pred):
        # Y is empty and Y_pred contains tensors computed during last call
        predictions, X_future_encoded = Y_pred

        loss, _ = cpc_loss(self.cpc.nb_timesteps_to_predict,
                           predictions,
                           X_future_encoded)
        return loss * self.loss_scaler

class LIMWorker(Model):

    def __init__(self,
                 lim,
                 loss_scaler):
        super(LIMWorker, self).__init__()

        self.lim = lim
        self.loss_scaler = loss_scaler

    def call(self, X_encoded):
        C1, C2, CR = self.lim.extract_chunks(X_encoded)

        C1_and_C2 = tf.concat([C1, C2], axis=1)
        C1_and_CR = tf.concat([C1, CR], axis=1)

        pos = self.lim.discriminator(C1_and_C2, training=True)
        neg = self.lim.discriminator(C1_and_CR, training=True)
        
        return pos, neg

    def compute_loss(self, Y, Y_pred):
        # Y is empty and Y_pred contains tensors computed during last call
        pos, neg = Y_pred

        loss, _ = lim_loss(self.lim.loss_fn, pos, neg)
        return loss * self.loss_scaler

class WorkerTargetsGenerator(Sequence):
  
    def __init__(self, gen, modules):
        self.gen = gen
        self.modules = modules

    def __len__(self):
        return len(self.gen)
  
    def __getitem__(self, batch_id):
        X, _ = self.gen[batch_id]
        Y = {}

        for module_type, model in self.modules.items():
            if module_type == 'Waveform' or module_type == 'MFCC':
                Y[module_type] = model.get_target(X)

        return X, Y

class MultiTaskModel(Model):

    REGRESSORS_NAMES = ['Waveform', 'MFCC']
    DISCRIMINATORS_NAMES = ['CPC', 'LIM']

    def __init__(self, encoder, in_shape, modules):
        super(MultiTaskModel, self).__init__()

        self.encoder = encoder
        self.in_shape = in_shape
        self.modules = self.create_modules(modules)

    def create_modules(self, modules_config):
        modules = {}
        
        for module in modules_config:
            module_type = module['type']
            loss_scaler = module.get('loss_scaler', 1.0)
            weight_regularizer = module.get('weight_regularizer', 0.0)

            if module_type in self.DISCRIMINATORS_NAMES:
                module_model = create_model(module,
                                            self.encoder,
                                            self.in_shape)

            if module_type == 'CPC':
                modules[module_type] = CPCWorker(module_model, loss_scaler)
            elif module_type == 'LIM':
                modules[module_type] = LIMWorker(module_model, loss_scaler)
            elif module_type == 'Waveform':
                modules[module_type] = WaveformWorker(weight_regularizer, loss_scaler)
            elif module_type == 'MFCC':
                modules[module_type] = MFCCWorker(weight_regularizer, loss_scaler)

        return modules

    def add_targets_to_gen(self, gen):
        return WorkerTargetsGenerator(gen, self.modules)

    def compile(self, optimizer):
        super(MultiTaskModel, self).compile()
        self.optimizer = optimizer

    def call(self, X):
        return self.encoder(X)

    def train_step(self, data):
        X, Y = data
        total_loss = 0
        losses = {name:0 for name in self.modules.keys()}
        trainable_params = []

        with tf.GradientTape() as tape:
            X_encoded = self.encoder(X, training=True)
            
            for module_type, model in self.modules.items():
                Y_target = Y.get(module_type, None)
                Y_pred = model(X_encoded, training=True)

                loss = model.compute_loss(Y_target, Y_pred)

                total_loss += loss
                losses[module_type] += loss
                trainable_params += model.trainable_weights

        trainable_params += self.encoder.trainable_weights
        grads = tape.gradient(total_loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        losses['loss'] = total_loss
        return losses

    def test_step(self, data):
        X, Y = data
        total_loss = 0
        losses = {name:0 for name in self.modules.keys()}

        X_encoded = self.encoder(X, training=False)

        for module_type, model in self.modules.items():
            Y_target = Y.get(module_type, None)
            Y_pred = model(X_encoded, training=False)

            loss = model.compute_loss(Y_target, Y_pred)

            total_loss += loss
            losses[module_type] += loss

        losses['loss'] = total_loss
        return losses