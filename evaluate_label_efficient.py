import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler

from sslforslr.utils.helpers import load_config, load_dataset, load_model, summary_for_shape
from sslforslr.utils.callbacks import SVMetricsCallback
from sslforslr.models.simclr import InfoNCELoss

class Classifier(Model):

    def __init__(self, model, add_last_layer=False):
        super().__init__()

        self.add_last_layer = add_last_layer
        self.encoder = model

        self.infonce_loss = InfoNCELoss()
        if self.add_last_layer:
            self.classifier_fc = Dense(512)

    def call(self, X):
        Z = self.encoder(X)
        if self.add_last_layer: Z = self.classifier_fc(Z)
        return Z

    def train_step(self, data):
        X_1, X_2, _ = data

        with tf.GradientTape() as tape:
            Z_1 = self.encoder(X_1, training=True)
            Z_2 = self.encoder(X_2, training=True)
            if self.add_last_layer:
                Z_1 = self.classifier_fc(Z_1, training=True)
                Z_2 = self.classifier_fc(Z_2, training=True)
            loss, accuracy = self.infonce_loss((Z_1, Z_2))

        params = self.encoder.trainable_weights
        if self.add_last_layer: params += self.classifier_fc.trainable_weights
        grads = tape.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return { 'loss': loss, 'accuracy': accuracy }

def lr_scheduler(epoch, lr):
    activate = (epoch != 0 and epoch % 5 == 0)
    return lr - lr * 0.05 if activate else lr

def create_callbacks(config, checkpoint_dir, patience):
    callbacks = [
        LearningRateScheduler(lr_scheduler),
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
            patience=patience
        )
    ]
    return callbacks


def train(
    config_path,
    nb_labels_per_spk,
    epochs,
    lr,
    batch_size,
    patience,
    fine_tune=False,
    supervised=False
):
    config, checkpoint_dir = load_config(config_path)

    # Disable features required only by self-supervised training
    config.dataset.wav_augment.enable = False
    config.dataset.frame_split = False
    config.dataset.provide_clean_and_aug = False

    gens, input_shape, nb_classes = load_dataset(config)
    (train_gen, val_gen) = gens
    
    train_gen.enable_supervision(nb_labels_per_spk)

    model = load_model(config, input_shape)

    if not supervised:
        last_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        if last_checkpoint_path:
            model.load_weights(last_checkpoint_path)
        model.trainable = fine_tune

    add_last_layer = not fine_tune and not supervised
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        classifier = Classifier(model, add_last_layer)
    classifier.compile(optimizer=Adam(learning_rate=lr))
    summary_for_shape(classifier, input_shape)

    # Determine name of current model
    checkpoint_dir += '_label-efficient-'
    checkpoint_dir += str(nb_labels_per_spk) + '-'
    if supervised:
        checkpoint_dir += 'supervised'
    else:
        checkpoint_dir += 'fine-tuned' if fine_tune else 'frozen'

    callbacks = create_callbacks(config, checkpoint_dir, patience)
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
        default=200,
        help='Number of epochs for trainings.'
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        help='Learning rate used during trainings.'
    )
    parser.add_argument(
        '--batch_size',
        default=64,
        help='Batch size used during trainings.'
    )
    parser.add_argument(
        '--patience',
        default=20,
        help='Number of epochs without a lower EER before ending training.'
    )
    parser.add_argument(
        '--nb_labels',
        default=[100, 10],
        nargs='*',
        help='Numbers of labels provided per speaker for each training.'
    )
    args = parser.parse_args()

    for nb_labels in args.nb_labels:
        nb_labels = int(nb_labels)
        if args.supervised:
            train(
                args.config,
                nb_labels,
                args.epochs,
                args.lr,
                args.batch_size,
                args.patience,
                supervised=True
            )
        else:
            train(
                args.config,
                nb_labels,
                args.epochs,
                args.lr,
                args.batch_size,
                args.patience,
                fine_tune=False
            )
            train(
                args.config,
                nb_labels,
                args.epochs,
                args.lr / 10,
                args.batch_size,
                args.patience,
                fine_tune=True
            )
