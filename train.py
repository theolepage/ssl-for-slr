import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from utils.helpers import load_config

def train(config_path):
    config, model, gens, checkpoint_dir = load_config(config_path)
    train_gen, val_gen, test_gen = gens

    # Load weights
    if (tf.train.latest_checkpoint(checkpoint_dir)):
        raise Exception('Train: model {} has already been trained.'.format(config['name']))

    # Setup callbacks
    save_callback = ModelCheckpoint(filepath=checkpoint_dir + '/training',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)

    # Start training
    nb_epochs = config['training']['epochs']
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=nb_epochs,
                        callbacks=[save_callback, early_stopping])

    # Save training history
    hist_path = checkpoint_dir + '/history.npy'
    np.save(hist_path, history.history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train(args.config)