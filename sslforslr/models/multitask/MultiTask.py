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

from sslforslr.models import CPCModel, LIMModel, cpc_loss, lim_loss

class MultiTaskModel(Model):
    '''
    Keras model combining different self-supervised workers
    similarly to PASE and PASE+.

    "Multi-task self-supervised learning for Robust Speech Recognition"
    Mirco Ravanelli et al.
    https://arxiv.org/pdf/2001.09239.pdf
    '''

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

            encoder_output_shape = self.encoder.compute_output_shape(self.in_shape)
            nb_timesteps = encoder_output_shape[0]
            encoded_dim = encoder_output_shape[1]

            if module_type == 'CPC':
                nb_timesteps_to_predict = module['nb_timesteps_to_predict']
                bidirectional = module.get('bidirectional', False)
                module_model = CPCModel(encoder,
                                        encoded_dim,
                                        nb_timesteps,
                                        nb_timesteps_to_predict,
                                        bidirectional,
                                        weight_regularizer)
                modules[module_type] = CPCWorker(module_model, loss_scaler)
            elif module_type == 'LIM':
                loss_fn = model_config['loss_fn']
                context_length = model_config.get('context_length', 1)
                module_model = LIMModel(encoder,
                                        nb_timesteps,
                                        loss_fn,
                                        context_length,
                                        weight_regularizer)
                modules[module_type] = LIMWorker(module_model, loss_scaler)
            elif module_type == 'Waveform':
                modules[module_type] = WaveformWorker(weight_regularizer, loss_scaler)
            elif module_type == 'MFCC':
                modules[module_type] = MFCCWorker(weight_regularizer, loss_scaler)
            elif module_type == 'LPS':
                modules[module_type] = LPSWorker(weight_regularizer, loss_scaler)

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
                # Handle module CPC bidirectional
                if module_type == 'CPC' and model.cpc.bidirectional:
                    X_r = tf.reverse(X, axis=[1])
                    X_encoded_r = self.encoder(X_r, training=True)
                    Y_pred = model((X_encoded, X_encoded_r), training=True)
                else:
                    Y_pred = model(X_encoded, training=True)
                
                Y_target = Y.get(module_type, None)

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
            # Handle module CPC bidirectional
            if module_type == 'CPC' and model.cpc.bidirectional:
                X_r = tf.reverse(X, axis=[1])
                X_encoded_r = self.encoder(X_r, training=False)
                Y_pred = model((X_encoded, X_encoded_r), training=False)
            else:
                Y_pred = model(X_encoded, training=False)
            
            Y_target = Y.get(module_type, None)

            loss = model.compute_loss(Y_target, Y_pred)

            total_loss += loss
            losses[module_type] += loss

        losses['loss'] = total_loss
        return losses


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
            if module_type in ['Waveform', 'MFCC', 'LPS']:
                Y[module_type] = model.get_target(X)

        return X, Y


class LPSWorker(Model):

    def __init__(self,
                 weight_regularizer,
                 loss_scaler,
                 fft_length=2048,
                 hop_length=160):
        super(LPSWorker, self).__init__()

        self.reg = regularizers.l2(weight_regularizer)
        self.loss_scaler = loss_scaler

        self.fft_length = fft_length
        self.hop_length = hop_length
        self.nb_outputs = fft_length // 2 + 1

        self.conv1 = Conv1D(filters=256,
                            kernel_size=1,
                            padding='same',
                            kernel_regularizer=self.reg,
                            bias_regularizer=self.reg)
        
        # PReLU shared_axes option implies that one parameter
        # per channel will be learned.
        self.activation1 = PReLU(shared_axes=[1]) 

        self.last_conv = Conv1D(filters=self.nb_outputs,
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
        frame_length = X.shape[1]

        Y = tf.signal.stft(np.squeeze(X, axis=-1),
                           frame_length=self.hop_length,
                           frame_step=self.hop_length,
                           fft_length=self.fft_length)
        Y = tf.math.abs(Y)
        Y = 10 * tf.experimental.numpy.log10(Y ** 2)
        return Y

    def compute_loss(self, Y, Y_pred):
        return MeanSquaredError()(Y, Y_pred) * self.loss_scaler


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
        if self.cpc.bidirectional:
            X_encoded, X_encoded_r = X_encoded[0], X_encoded[1]

        # X_encoded = audio sequence in correct order
        X_past_encoded = X_encoded[:, 0:self.cpc.nb_timesteps_for_context, ...]
        X_future_encoded = X_encoded[:, self.cpc.nb_timesteps_for_context:, ...]
        X_past_context = self.cpc.ar1(X_past_encoded, training=True)
        predictions = self.cpc.predictor1(X_past_context, training=True)

        if not self.cpc.bidirectional:
            return predictions, X_future_encoded

        # X_encoded_r = audio sequence in reversed order
        X_past_encoded_r = X_encoded_r[:, 0:self.cpc.nb_timesteps_for_context, ...]
        X_future_encoded_r = X_encoded_r[:, self.cpc.nb_timesteps_for_context:, ...]
        X_past_context_r = self.cpc.ar2(X_past_encoded_r, training=True)
        predictions_r = self.cpc.predictor2(X_past_context_r, training=True)

        return predictions, X_future_encoded, predictions_r, X_future_encoded_r

    def compute_loss(self, Y, Y_pred):
        # Y is empty and Y_pred contains tensors computed during last call
        if self.cpc.bidirectional:
            predictions, X_future_encoded, predictions_r, X_future_encoded_r = Y_pred
        else:
            predictions, X_future_encoded = Y_pred

        loss, _ = cpc_loss(self.cpc.nb_timesteps_to_predict,
                           predictions,
                           X_future_encoded)

        if self.cpc.bidirectional:
            loss2, _ = cpc_loss(self.cpc.nb_timesteps_to_predict,
                                predictions_r,
                                X_future_encoded_r)
            loss = (loss + loss2) / 2.0

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