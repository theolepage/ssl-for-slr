import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

from sslforslr.utils.callbacks import TimeHistoryCallback
from sslforslr.utils.helpers import load_config, load_dataset, load_model

class Classifier(Model):

    def __init__(self, nb_categories):
        super(Classifier, self).__init__()

        self.nb_categories = nb_categories

        self.flatten = Flatten()
        self.dense1 = Dense(units=256)
        self.dense2 = Dense(units=nb_categories, activation='softmax')

    def call(self, X):
        X = self.flatten(X)
        X = self.dense1(X)
        X = self.dense2(X)
        return X

def create_classifier(config, input_shape, nb_categories, model):
    learning_rate = config['evaluate']['learning_rate']

    inputs = Input(input_shape)
    inputs_encoded = model(inputs)
    outputs = Classifier(nb_categories)(inputs_encoded)

    classifier = Model(inputs, outputs)
    classifier.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    classifier.summary()

    return classifier

def train_evaluate(config_path):
    config, checkpoint_dir, eval_checkpoint_dir = load_config(config_path)

    # Prevent re-training model
    if tf.train.latest_checkpoint(eval_checkpoint_dir):
        raise Exception('%s already contains checkpoints.' % eval_checkpoint_dir)

    gens, input_shape, nb_categories = load_dataset(config,
                                                    eval_checkpoint_dir,
                                                    key='evaluate')

    model = load_model(config, input_shape)

    # Load pre-trained weights
    last_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if last_checkpoint_path:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model.load_weights(last_checkpoint_path)
    print('Loading pretrained model: ', last_checkpoint_path is not None)
    
    model.trainable = config['evaluate'].get('train_encoder', True)

    # Create classifier
    classifier = create_classifier(config, input_shape, nb_categories, model)

    # Setup callbacks
    save_callback = ModelCheckpoint(filepath=eval_checkpoint_dir + '/training',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)
    tensorboard = TensorBoard(log_dir=eval_checkpoint_dir + '/logs/',
                              histogram_freq=1)
    time_history = TimeHistoryCallback()

    # Start training
    train_gen, val_gen, test_gen = gens
    nb_epochs = config['evaluate']['epochs']
    callbacks = [save_callback, early_stopping, time_history]
    if config['evaluate'].get('tensorboard', False):
        callbacks.append(tensorboard)
    history = classifier.fit(train_gen,
                             validation_data=val_gen,
                             epochs=nb_epochs,
                             callbacks=callbacks)

    # Save training history
    hist_path = eval_checkpoint_dir + '/history.npy'
    history = time_history.update_history(history)
    np.save(hist_path, history.history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train_evaluate(args.config)
