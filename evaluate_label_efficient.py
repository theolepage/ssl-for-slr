import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import numpy as np
import prettyprinter as pp

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler

from sslforslr.utils.helpers import load_config, load_dataset, load_model
from sslforslr.utils.callbacks import SVMetricsCallback

class Classifier(Model):

    def __init__(self, nb_classes):
        super(Classifier, self).__init__()

        self.fc1 = Dense(512, activation='relu')
        # self.fc1_ = Dense(512, activation='relu')
        self.fc2 = Dense(nb_classes, activation='softmax')

    def call(self, X):
        return self.fc2(self.fc1(X))


def create_classifier(input_shape, nb_classes, lr, model):
    inputs = Input(input_shape)
    embeddings = model(inputs)
    outputs = Classifier(nb_classes)(embeddings)

    classifier = Model(inputs, outputs)
    classifier.compile(optimizer=Adam(learning_rate=lr),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    classifier.summary()

    return classifier


def create_callbacks(config, checkpoint_dir):
    return [
        SVMetricsCallback(config),
        ModelCheckpoint(
            filepath=checkpoint_dir + '/training',
            monitor='test_eer',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=checkpoint_dir + '/logs/',
            histogram_freq=1
        ),
        EarlyStopping(
            monitor='test_eer',
            mode='min',
            patience=10
        )
    ]


def train(
    config_path,
    labels_ratio,
    epochs,
    lr,
    fine_tune=False,
    supervised=False
):
    config, checkpoint_dir = load_config(config_path)

    # Disable features required only by self-supervised training
    config.dataset.frame_split = False
    config.dataset.provide_clean_and_aug = False

    gens, input_shape, nb_classes = load_dataset(config, labels_ratio)
    (train_gen, val_gen) = gens

    model = load_model(config, input_shape)

    if not supervised:
        last_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        if last_checkpoint_path:
            model.load_weights(last_checkpoint_path)
        model.trainable = fine_tune

    classifier = create_classifier(input_shape, nb_classes, lr, model)

    # Determine name of current model
    checkpoint_dir += '_label-efficient-'
    checkpoint_dir += str(labels_ratio) + '-'
    if not supervised:
        checkpoint_dir += 'supervised'
    else:
        checkpoint_dir += 'fine-tuned' if fine_tune else 'frozen'

    callbacks = create_callbacks(config, checkpoint_dir)
    history = classifier.fit(train_gen,
                             validation_data=val_gen,
                             epochs=epochs,
                             callbacks=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    parser.add_argument(
        '--supervised',
        action='store_true',
        help='Use supervised instead of self-supervised model.'
    )
    parser.add_argument(
        '--epochs',
        default=100,
        help='Number of epochs for training.'
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        help='Learning rate used during training.'
    )
    parser.add_argument(
        '--labels_ratio',
        default=[0.1, 0.5, 1],
        nargs='*',
        help='List of labels ratio (in %) per speaker.'
    )
    args = parser.parse_args()

    for labels_ratio in args.labels_ratio:
        if args.supervised:
            train(
                args.config,
                labels_ratio,
                args.epochs,
                args.lr,
                supervised=True
            )
        else:
            train(
                args.config,
                labels_ratio,
                args.epochs,
                args.lr,
                fine_tune=False
            )
            train(
                args.config,
                labels_ratio,
                args.epochs,
                args.lr,
                fine_tune=True
            )
