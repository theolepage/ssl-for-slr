import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import regularizers

class CPCModel(Model):
    '''
    Contrastive Predictive Coding (CPC) for audio signals,
    implemented as a Keras model.

    "Representation Learning with Contrastive Predictive Coding"
    Aaron van den Oord, Yazhe Li, Oriol Vinyals
    https://arxiv.org/pdf/1807.03748.pdf
    '''

    def __init__(self,
                 encoder,
                 encoded_dim,
                 nb_timesteps,
                 config):
        super(CPCModel, self).__init__()

        self.encoded_dim = encoded_dim
        self.nb_timesteps = nb_timesteps
        self.nb_t_to_predict = config.nb_timesteps_to_predict
        self.nb_t_for_context = nb_timesteps - self.nb_t_to_predict
        self.bidirectional = config.bidirectional

        self.reg = regularizers.l2(config.weight_reg)

        # Instantiate sub models
        self.encoder = encoder
        self.ar1 = Autoregressive(config.context_network, self.reg)
        self.predictor1 = Predictor(self.encoded_dim,
                                    self.nb_t_to_predict,
                                    self.reg)

        if self.bidirectional:
            self.ar2 = Autoregressive(context_network, self.reg)
            self.predictor2 = Predictor(self.encoded_dim,
                                        self.nb_t_to_predict,
                                        self.reg)

    def compile(self, optimizer, **kwargs):
        super(CPCModel, self).compile(**kwargs)
        self.optimizer = optimizer

    def call(self, X):
        if self.bidirectional:
            X_r = tf.reverse(X, axis=[1])
            X_1 = self.ar1(self.encoder(X))
            X_2 = self.ar2(self.encoder(X_r))
            return tf.concat([X_1, X_2], axis=-1)

        return self.ar1(self.encoder(X))

    def train_step(self, data):
        X, _ = data # Discard labels provided by the dataset generator
        # X shape: (batch_size, frame_length, 1)

        with tf.GradientTape() as tape:
            Z = self.encoder(X, training=True)
            # Out shape: (batch_size, frame_length / 160, encoded_dim)

            Z_past = Z[:, 0:self.nb_t_for_context, ...]
            Z_future = Z[:, self.nb_t_for_context:, ...]

            C = self.ar1(Z_past, training=True)
            # Out shape: (batch_size, context_dim)

            predictions = self.predictor1(C, training=True)
            # Out shape: (batch_size, nb_timesteps_to_predict, encoded_dim)

            loss, accuracy = cpc_loss(self.nb_t_to_predict,
                                      predictions,
                                      Z_future)

            if self.bidirectional:
                X_r = tf.reverse(X, axis=[1])

                Z = self.encoder(X_r, training=True)

                Z_past = Z[:, 0:self.nb_t_for_context, ...]
                Z_future = Z[:, self.nb_t_for_context:, ...]

                C = self.ar2(Z_past, training=True)

                predictions = self.predictor2(C, training=True)

                loss2, accuracy2 = cpc_loss(self.nb_t_to_predict,
                                            predictions,
                                            Z_future)

                loss = (loss + loss2) / 2.0
                accuracy = (accuracy + accuracy2) / 2.0

        trainable_params = self.encoder.trainable_weights
        trainable_params += self.ar1.trainable_weights
        trainable_params += self.predictor1.trainable_weights
        if self.bidirectional:
            trainable_params += self.ar2.trainable_weights
            trainable_params += self.predictor2.trainable_weights

        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        return { 'loss': loss, 'accuracy': accuracy }

    def test_step(self, data):
        X, _ = data # Discard labels provided by the dataset generator
        
        Z = self.encoder(X, training=False)
        Z_past = Z[:, 0:self.nb_t_for_context, ...]
        Z_future = Z[:, self.nb_t_for_context:, ...]

        C = self.ar1(Z_past, training=False)
        predictions = self.predictor1(C, training=False)

        loss, accuracy = cpc_loss(self.nb_t_to_predict,
                                  predictions,
                                  Z_future)

        if self.bidirectional:
            X_r = tf.reverse(X, axis=[1])
           
            Z = self.encoder(X, training=False)
            Z_past = Z[:, 0:self.nb_t_for_context, ...]
            Z_future = Z[:, self.nb_t_for_context:, ...]

            C = self.ar2(Z_past, training=False)
            predictions = self.predictor2(C, training=False)

            loss2, accuracy2 = cpc_loss(self.nb_t_to_predict,
                                        predictions,
                                        Z_future)

            loss = (loss + loss2) / 2.0
            accuracy = (accuracy + accuracy2) / 2.0

        return { 'loss': loss, 'accuracy': accuracy }


class Autoregressive(Model):

    def __init__(self, context_network, reg):
        super(Autoregressive, self).__init__()

        self.layers_ = []
        for i in range(context_network.nb_layers):
            return_sequence = i != (context_network.nb_layers - 1)
            if context_network.model_type == 'gru':
                self.layers_.append(GRU(units=context_network.dim,
                                        return_sequences=return_sequence,
                                        kernel_regularizer=reg,
                                        recurrent_regularizer=reg,
                                        bias_regularizer=reg))
            elif context_network.model_type == 'lstm':
                self.layers_.append(LSTM(units=context_network.dim,
                                         return_sequences=return_sequence,
                                         kernel_regularizer=reg,
                                         recurrent_regularizer=reg,
                                         bias_regularizer=reg))
            else:
                raise Exception('CPC: context network model type not supported')

    def call(self, X):
        for layer in self.layers_:
            X = layer(X)
        return X


class Predictor(Model):

    def __init__(self, encoded_dim, nb_timesteps_to_predict, reg):
        super(Predictor, self).__init__()

        self.layers_ = []
        for i in range(nb_timesteps_to_predict):
            self.layers_.append(Dense(units=encoded_dim,
                                      kernel_regularizer=reg,
                                      bias_regularizer=reg))

    def call(self, context):
        predictions = []
        for layer in self.layers_:
            predictions.append(layer(context))

        return tf.stack(predictions, axis=1)


@tf.function
def cpc_loss(nb_timesteps_to_predict, predictions, Z_future):
    # Shape: (batch_size, nb_timesteps_to_predict, encoded_dim)
    
    batch_size = tf.shape(predictions)[0]

    losses = tf.zeros((batch_size))

    # Vectorized implementation of InfoNCE loss
    # Note: "distractors" (anchor-negative pairs) are sampled in the
    # current batch as we make the assumption that each utterance
    # belong to a different speaker.
    for t in range(nb_timesteps_to_predict):
        dot = tf.linalg.matmul(Z_future[:, t, :],
                               predictions[:, t, :],
                               transpose_b=True)
        
        # Determine loss
        log_softmax_dot = tf.nn.log_softmax(dot, axis=-1)
        diag = tf.linalg.tensor_diag_part(log_softmax_dot)
        losses += diag

    losses /= tf.cast(nb_timesteps_to_predict, dtype=tf.float32)
    loss = -tf.math.reduce_mean(losses)

    # Determine accuracy
    # (i.e. the percentage of correct predictions for the last timestep)
    softmax_dot = tf.nn.softmax(dot, axis=-1)
    pred_indices = tf.math.argmax(softmax_dot, axis=0, output_type=tf.int32)
    preds_acc = tf.math.equal(pred_indices, tf.range(0, batch_size))
    accuracy = tf.math.count_nonzero(preds_acc, dtype=tf.int32) / batch_size

    return loss, accuracy