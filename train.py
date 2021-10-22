import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

from sslforslr.utils.helpers import load_config, load_dataset, load_model
from sslforslr.utils.callbacks import SVMetricsCallback

from sslforslr.models.moco import MoCoUpdateCallback

def train(config_path):
    # Load config, model and dataset
    config, checkpoint_dir = load_config(config_path)
    model = load_model(config)

    gens = load_dataset(config)
    train_gen, val_gen = gens
    print("Number of training batches:", len(train_gen))
    print("Number of val batches:", len(val_gen))

    # Prevent re-training model
    if tf.train.latest_checkpoint(checkpoint_dir):
        raise Exception('%s has already been trained.' % config['name'])

    # Setup callbacks
    sv_metrics = SVMetricsCallback(config)
    save_callback = ModelCheckpoint(filepath=checkpoint_dir + '/training',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)
    tensorboard = TensorBoard(log_dir=checkpoint_dir + '/logs/',
                              histogram_freq=1)

    # Start training
    nb_epochs = config['training']['epochs']
    callbacks = [save_callback, sv_metrics, tensorboard, early_stopping]
    if config['model']['type'] == 'MoCo':
        callbacks.append(MoCoUpdateCallback(train_gen))
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=nb_epochs,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=8)

    np.save(checkpoint_dir + '/history.npy', history.history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train(args.config)
