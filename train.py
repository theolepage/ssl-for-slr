import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import numpy as np
import prettyprinter as pp

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler

from sslforslr.utils.helpers import load_config, load_dataset, load_model
from sslforslr.utils.callbacks import SVMetricsCallback

from sslforslr.models.moco import MoCoUpdateCallback

def simclr_lr_scheduler(epoch, lr):
    activate = (epoch != 0 and epoch % 1 == 0)
    return lr - lr * 0.05 if activate else lr

def create_callbacks(config, checkpoint_dir):
    callbacks = []

    if config.model.__NAME__ == 'simclr':
        callbacks.append(LearningRateScheduler(simclr_lr_scheduler))
    elif config.model.__NAME__ == 'moco':
        callbacks.append(MoCoUpdateCallback(train_gen))

    callbacks.append(SVMetricsCallback(config))

    callbacks.append(
        ModelCheckpoint(filepath=checkpoint_dir + '/training',
                        monitor='test_eer',
                        mode='min',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1)
    )

    callbacks.append(
        TensorBoard(log_dir=checkpoint_dir + '/logs/',
                    histogram_freq=1)
    )

    callbacks.append(
        EarlyStopping(monitor='test_eer',
                      mode='min',
                      patience=5)
    )

    return callbacks

def train(config_path):
    config, checkpoint_dir = load_config(config_path)
    (train_gen, val_gen), input_shape = load_dataset(config)
    model = load_model(config, input_shape)

    pp.install_extras(include=['dataclasses'])
    pp.pprint(config)
    print("Number of training batches:", len(train_gen))
    print("Number of val batches:", len(val_gen))
    
    # Prevent re-training model
    if tf.train.latest_checkpoint(checkpoint_dir):
        raise Exception('%s has already been trained.' % config.name)

    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=config.training.epochs,
                        callbacks=create_callbacks(config, checkpoint_dir))
                        # use_multiprocessing=True,
                        # workers=8)

    np.save(checkpoint_dir + '/history.npy', history.history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train(args.config)
