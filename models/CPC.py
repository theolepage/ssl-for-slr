import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import regularizers

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
        outputs = []
        for layer in self.layers_:
            outputs.append(layer(context))

        output = Lambda(lambda X: tf.stack(X, axis=1))(outputs)

        return output

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

class CPCModel(Model):

    def __init__(self,
                 encoder,
                 encoded_dim,
                 nb_timesteps,
                 nb_timesteps_to_predict,
                 weight_regularizer=0.0):
        super(CPCModel, self).__init__()

        self.encoded_dim = encoded_dim
        self.nb_timesteps = nb_timesteps
        self.nb_timesteps_to_predict = nb_timesteps_to_predict
        self.nb_timesteps_for_context = nb_timesteps - nb_timesteps_to_predict

        self.reg = regularizers.l2(weight_regularizer)

        self.encoder = encoder
        self.ar = Autoregressive(self.reg)
        self.predictor = Predictor(self.encoded_dim,
                                   self.nb_timesteps_to_predict,
                                   self.reg)

    def compile(self, optimizer):
        super(CPCModel, self).compile()
        self.optimizer = optimizer

    def call(self, X):
        return self.ar(self.encoder(X))

    def train_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator

        with tf.GradientTape() as tape:
            # X shape: (batch_size, frame_length, 1)

            X_encoded = self.encoder(X, training=True)
            # Out shape: (batch_size, frame_length / 160, encoded_dim)

            X_past_encoded = X_encoded[:, 0:self.nb_timesteps_for_context, ...]
            X_future_encoded = X_encoded[:, self.nb_timesteps_for_context:, ...]

            X_past_context = self.ar(X_past_encoded, training=True)
            # Out shape: (batch_size, 256)

            predictions = self.predictor(X_past_context, training=True)
            # Out shape: (batch_size, nb_timesteps_to_predict, encoded_dim)

            loss, accuracy = cpc_loss(self.nb_timesteps_to_predict,
                                      predictions,
                                      X_future_encoded)
            # Out shape: (batch_size)

        trainable_params = self.encoder.trainable_weights
        trainable_params += self.ar.trainable_weights
        trainable_params += self.predictor.trainable_weights
        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        return { 'loss': loss, 'accuracy': accuracy }

    def test_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator
        
        X_encoded = self.encoder(X, training=False)
        X_past_encoded = X_encoded[:, 0:self.nb_timesteps_for_context, ...]
        X_future_encoded = X_encoded[:, self.nb_timesteps_for_context:, ...]

        X_past_context = self.ar(X_past_encoded, training=False)
        predictions = self.predictor(X_past_context, training=False)

        loss, accuracy = cpc_loss(self.nb_timesteps_to_predict,
                                  predictions,
                                  X_future_encoded)

        return { 'loss': loss, 'accuracy': accuracy }