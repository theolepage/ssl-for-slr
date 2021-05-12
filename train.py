import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from utils.helpers import load_config

def train(config_path):
    config, model, dataset = load_config(config_path)

    checkpoint_dir = './checkpoints/' + config['name']
    last_checkpoint_path = checkpoint_dir + '/training'
    batch_size = config['training']['batch_size']
    nb_epochs = config['training']['epochs']
   
    # Create subfolder for saving checkpoints
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_gen, val_gen, test_gen = dataset.load(batch_size, checkpoint_dir)
    print("Number of training batches:", len(train_gen))
    print("Number of validation batches:", len(val_gen))

    # Load weights
    if (tf.train.latest_checkpoint(checkpoint_dir)):
        raise Exception('Train: model {} has already been trained.'.format(config['name']))

    # Setup callbacks
    save_callback = ModelCheckpoint(filepath=last_checkpoint_path,
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)

    # Start training
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