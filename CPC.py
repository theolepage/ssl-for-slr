frame_length = 20480   # 1.28s at 16kHz (LibriSpeech)
frame_stride = 20480   # 1.28s at 16kHz (LibriSpeech)

nb_timesteps = int(frame_length // 160)  # 128
nb_timesteps_to_predict = 12
nb_timesteps_for_context = nb_timesteps - nb_timesteps_to_predict

encoded_dim = 512

batch_size = 64

max_frames_per_utterance = 1
nb_speakers = 64
max_utterances = 1000






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

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint




from LibriSpeech import LibriSpeechLoader

lb = LibriSpeechLoader("D:/Datasets/LibriSpeech/train-clean-100/*",
                       frame_length=frame_length,
                       frame_stride=frame_stride,
                       max_frames_per_utterance=max_frames_per_utterance,
                       max_speakers=nb_speakers,
                       max_utterances=max_utterances,
                       val_split=0.2)

train_gen, val_gen = lb.load(batch_size)

print("Number of training batches:", len(train_gen))
print("Number of validation batches:", len(val_gen))



import numpy as np

class Encoder(Model):

    def __init__(self):
        super(Encoder, self).__init__()

        nb_filters = [512, 512, 512, 512, encoded_dim]
        kernel_sizes = [10, 8, 4, 4, 4]
        strides = [5, 4, 2, 2, 2]

        self.blocks = []
        for i in range(5):
            self.blocks.append(Conv1D(nb_filters[i],
                                      kernel_size=kernel_sizes[i],
                                      strides=strides[i],
                                      padding='same'))
            self.blocks.append(BatchNormalization())
            self.blocks.append(ReLU())

    def call(self, X):
        for layer in self.blocks:
            X = layer(X)
        return X

    def compute_output_shape(self, input_shape):
        return (nb_timesteps, encoded_dim)




class Autoregressive(Model):

    def __init__(self):
        super(Autoregressive, self).__init__()

        self.rnn = GRU(units=256, return_sequences=False)

    def call(self, X):
        return self.rnn(X)



class Predictor(Model):

    def __init__(self):
        super(Predictor, self).__init__()

        self.layers_ = []
        for i in range(nb_timesteps_to_predict):
            self.layers_.append(Dense(units=encoded_dim))

    def call(self, context):
        outputs = []
        for layer in self.layers_:
            outputs.append(layer(context))

        output = Lambda(lambda X: tf.stack(X, axis=1))(outputs)

        return output


class CPCLayer(Layer):

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, data):
        predictions, X_future_encoded = data
        # Shape: (batch_size, nb_timesteps_to_predict, encoded_dim)
        
        losses = tf.zeros((batch_size))
        accuracies = tf.zeros((batch_size), dtype=tf.float64)

        for t in range(nb_timesteps_to_predict):
            dot = tf.linalg.matmul(X_future_encoded[:, t, :],
                                   predictions[:, t, :],
                                   transpose_b=True)
            
            # Determine loss
            log_softmax_dot = tf.nn.log_softmax(dot, axis=0)
            diag = tf.linalg.tensor_diag_part(log_softmax_dot)
            losses += diag

            # Determine accuracy
            softmax_dot = tf.nn.softmax(dot, axis=0)
            pred_indices = tf.math.argmax(softmax_dot, axis=0)
            preds_acc = tf.math.equal(pred_indices, np.arange(0, batch_size))
            accuracies += tf.math.count_nonzero(preds_acc) / batch_size

        losses /= nb_timesteps_to_predict
        accuracies /= nb_timesteps_to_predict

        # Compute the average loss and accuracy across all batches
        loss = tf.math.reduce_mean(losses)
        accuracy = tf.math.reduce_mean(accuracies)

        return -1.0 * loss, accuracy



class CPCModel(Model):

    def __init__(self):
        super(CPCModel, self).__init__()
        self.encoder = Encoder()
        self.ar = Autoregressive()
        self.predictor = Predictor()

    def compile(self, optimizer):
        super(CPCModel, self).compile()
        self.optimizer = optimizer

    def call(self, X):
        return self.encoder(X) # or self.ar(X)?

    def train_step(self, data):
        X, _ = data # Discard Y provided by the dataset generator
        
        with tf.GradientTape() as tape:
            # X shape: (batch_size, frame_length, 1)

            X_encoded = self.encoder(X, training=True)
            # Out shape: (batch_size, frame_length / 160, encoded_dim)

            X_past_encoded = X_encoded[:, 0:nb_timesteps_for_context, ...]
            X_future_encoded = X_encoded[:, nb_timesteps_for_context:, ...]

            X_past_context = self.ar(X_past_encoded, training=True)
            # Out shape: (batch_size, 256)

            predictions = self.predictor(X_past_context, training=True)
            # Out shape: (batch_size, nb_timesteps_to_predict, encoded_dim)

            loss, accuracy = CPCLayer()([predictions, X_future_encoded])
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
        X_past_encoded = X_encoded[:, 0:nb_timesteps_for_context, ...]
        X_future_encoded = X_encoded[:, nb_timesteps_for_context:, ...]

        X_past_context = self.ar(X_past_encoded, training=False)
        predictions = self.predictor(X_past_context, training=False)

        loss, accuracy = CPCLayer()([predictions, X_future_encoded])

        return { 'loss': loss, 'accuracy': accuracy }


cpc_model = CPCModel()
cpc_model.compile(Adam(learning_rate=0.0001))

checkpoint_path = "./checkpoints/cpc-training-{epoch:04d}.ckpt"
save_callback = ModelCheckpoint(filepath=checkpoint_path,
                                monitor="loss",
                                save_best_only=True,
                                save_weights_only=True,
                                verbose=1)

# negative samples from same speaker + current and other sentences
# accuracy only on last timestep
# early stopping, reduce lr on plateau

history = cpc_model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=30,
                        callbacks=[save_callback])


import pandas as pd

hist_df = pd.DataFrame(history.history)
hist_json_file = './checkpoints/history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)