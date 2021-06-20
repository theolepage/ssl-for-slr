import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
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
                 nb_timesteps_to_predict,
                 bidirectional=False,
                 weight_regularizer=0.0):
        super(CPCModel, self).__init__()

        self.encoded_dim = encoded_dim
        self.nb_timesteps = nb_timesteps
        self.nb_timesteps_to_predict = nb_timesteps_to_predict
        self.nb_timesteps_for_context = nb_timesteps - nb_timesteps_to_predict
        self.bidirectional = bidirectional

        self.reg = regularizers.l2(weight_regularizer)

        # Instantiate sub models
        self.encoder = encoder
        self.ar1 = Autoregressive(self.reg)
        self.predictor1 = Predictor(self.encoded_dim,
                                    self.nb_timesteps_to_predict,
                                    self.reg)

        if self.bidirectional:
            self.ar2 = Autoregressive(self.reg)
            self.predictor2 = Predictor(self.encoded_dim,
                                        self.nb_timesteps_to_predict,
                                        self.reg)

    def compile(self, optimizer):
        super(CPCModel, self).compile()
        self.optimizer = optimizer

    def call(self, X):
        if self.bidirectional:
            X_r = tf.reverse(X, axis=[1])
            X_1 = self.ar1(self.encoder(X))
            X_2 = self.ar2(self.encoder(X_r))
            return tf.concat([X_1, X_2], axis=-1)

        return self.ar1(self.encoder(X))

    def train_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator
        # X shape: (batch_size, frame_length, 1)

        with tf.GradientTape() as tape:
            X_encoded = self.encoder(X, training=True)
            # Out shape: (batch_size, frame_length / 160, encoded_dim)

            X_past_encoded = X_encoded[:, 0:self.nb_timesteps_for_context, ...]
            X_future_encoded = X_encoded[:, self.nb_timesteps_for_context:, ...]

            X_past_context = self.ar1(X_past_encoded, training=True)
            # Out shape: (batch_size, 256)

            predictions = self.predictor1(X_past_context, training=True)
            # Out shape: (batch_size, nb_timesteps_to_predict, encoded_dim)

            loss, accuracy = cpc_loss(self.nb_timesteps_to_predict,
                                    predictions,
                                    X_future_encoded)
            # Out shape: (batch_size)

            if self.bidirectional:
                X_r = tf.reverse(X, axis=[1])

                X_encoded = self.encoder(X_r, training=True)

                X_past_encoded = X_encoded[:, 0:self.nb_timesteps_for_context, ...]
                X_future_encoded = X_encoded[:, self.nb_timesteps_for_context:, ...]

                X_past_context = self.ar2(X_past_encoded, training=True)

                predictions = self.predictor2(X_past_context, training=True)

                loss2, accuracy2 = cpc_loss(self.nb_timesteps_to_predict,
                                            predictions,
                                            X_future_encoded)

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
        X, _ = data # Discard Y provided by the dataset generator
        
        X_encoded = self.encoder(X, training=False)
        X_past_encoded = X_encoded[:, 0:self.nb_timesteps_for_context, ...]
        X_future_encoded = X_encoded[:, self.nb_timesteps_for_context:, ...]

        X_past_context = self.ar1(X_past_encoded, training=False)
        predictions = self.predictor1(X_past_context, training=False)

        loss, accuracy = cpc_loss(self.nb_timesteps_to_predict,
                                  predictions,
                                  X_future_encoded)

        if self.bidirectional:
            X_r = tf.reverse(X, axis=[1])
           
            X_encoded = self.encoder(X, training=False)
            X_past_encoded = X_encoded[:, 0:self.nb_timesteps_for_context, ...]
            X_future_encoded = X_encoded[:, self.nb_timesteps_for_context:, ...]

            X_past_context = self.ar2(X_past_encoded, training=False)
            predictions = self.predictor2(X_past_context, training=False)

            loss2, accuracy2 = cpc_loss(self.nb_timesteps_to_predict,
                                        predictions,
                                        X_future_encoded)

            loss = (loss + loss2) / 2.0
            accuracy = (accuracy + accuracy2) / 2.0

        return { 'loss': loss, 'accuracy': accuracy }


class Autoregressive(Model):

    def __init__(self, reg):
        super(Autoregressive, self).__init__()

        self.rnn = GRU(units=256,
                       return_sequences=False,
                       kernel_regularizer=reg,
                       recurrent_regularizer=reg,
                       bias_regularizer=reg)

    def call(self, X):
        return self.rnn(X)


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
def cpc_loss(nb_timesteps_to_predict, predictions, X_future_encoded):
    # Shape: (batch_size, nb_timesteps_to_predict, encoded_dim)
    
    batch_size = tf.shape(predictions)[0]

    losses = tf.zeros((batch_size))

    for t in range(nb_timesteps_to_predict):
        dot = tf.linalg.matmul(X_future_encoded[:, t, :],
                                predictions[:, t, :],
                                transpose_b=True)
        
        # Determine loss
        log_softmax_dot = tf.nn.log_softmax(dot, axis=0)
        diag = tf.linalg.tensor_diag_part(log_softmax_dot)
        losses += diag

    losses /= tf.cast(nb_timesteps_to_predict, dtype=tf.float32)

    # Determine accuracy
    softmax_dot = tf.nn.softmax(dot, axis=0)
    pred_indices = tf.math.argmax(softmax_dot, axis=0, output_type=tf.int32)
    preds_acc = tf.math.equal(pred_indices, tf.range(0, batch_size))
    accuracies = tf.math.count_nonzero(preds_acc, dtype=tf.int32) / batch_size

    # Compute the average loss and accuracy across all batches
    loss = tf.math.reduce_mean(losses)
    accuracy = tf.math.reduce_mean(accuracies)

    return -1.0 * loss, accuracy