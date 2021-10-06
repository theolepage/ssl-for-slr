import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

from sslforslr.utils.helpers import load_config, load_dataset, load_model
from sslforslr.utils.callbacks import TimeHistoryCallback

from sslforslr.models.moco import MoCoUpdateCallback

def train(config_path):
    config, checkpoint_dir, _ = load_config(config_path)

    # Prevent re-training model
    if tf.train.latest_checkpoint(checkpoint_dir):
        raise Exception('%s already contains checkpoints.' % checkpoint_dir)

    gens, input_shape, _ = load_dataset(config,
                                        checkpoint_dir,
                                        key='training')
    train_gen = gens[0]
    val_gen = gens[1]

    model = load_model(config, input_shape)

    # For multitask model: add targets to data generator
    if config['model']['type'] == 'multitask':
        for i in range(len(gens)):
            gens[i] = model.add_targets_to_gen(gens[i])

    # Setup callbacks
    save_callback = ModelCheckpoint(filepath=checkpoint_dir + '/training',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)
    tensorboard = TensorBoard(log_dir=checkpoint_dir + '/logs/',
                              histogram_freq=1)
    time_history = TimeHistoryCallback()

    # Start training
    nb_epochs = config['training']['epochs']
    callbacks = [save_callback, early_stopping, time_history]
    if config['training'].get('tensorboard', False):
        callbacks.append(tensorboard)
    if config['model']['type'] == 'MoCo':
        callbacks.append(MoCoUpdateCallback(train_gen))
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=nb_epochs,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=8)

    # Save training history
    hist_path = checkpoint_dir + '/history.npy'
    history = time_history.update_history(history)
    np.save(hist_path, history.history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train(args.config)
