import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from models.SpeakerIdClassifier import SpeakerIdClassifier
from utils.helpers import load_config

def train(config_path, batch_size, epochs, lr, epochs_ft, lr_ft, layers_ft, no_ft):
    config, encoder, dataset = load_config(config_path)

    checkpoint_dir = './checkpoints/' + config['name']
    last_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint_dir_spkid = './checkpoints/' + config['name'] + '_spkid'
    last_checkpoint_spkid_path = checkpoint_dir_spkid + '/training'
    frame_length = config['dataset']['frames']['length']
    nb_speakers = config['dataset']['nb_speakers']
    
    # Create subfolder for saving checkpoints
    Path(checkpoint_dir_spkid).mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, test_gen = dataset.load(batch_size, checkpoint_dir)
    print("Number of training batches:", len(train_gen))
    print("Number of validation batches:", len(val_gen))

    # Load pre-trained encoder
    # Otherwise, train the encoder from scratch (supervised baseline)
    if last_checkpoint_path:
        print('Loading pretrained model')
        encoder.trainable = False
        encoder.load_weights(last_checkpoint_path)

    # Create model: encoder + classifier
    inputs = Input((frame_length, 1))
    inputs_encoded = encoder(inputs)
    outputs = SpeakerIdClassifier(nb_speakers)(inputs_encoded)

    model_spkid = Model(inputs, outputs)
    model_spkid.compile(optimizer=Adam(learning_rate=lr),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    model_spkid.summary()

    # Setup callbacks
    save_callback = ModelCheckpoint(filepath=last_checkpoint_spkid_path,
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)

    # Start training
    history = model_spkid.fit(train_gen,
                              validation_data=val_gen,
                              epochs=epochs,
                              callbacks=[save_callback, early_stopping])

    # Save training history
    hist_path = checkpoint_dir_spkid + '/history.npy'
    np.save(hist_path, history.history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=64)
    parser.add_argument('--epochs', help='Number of epochs for training.', type=int, default=50)
    parser.add_argument('--lr', help='Learning rate for training.', type=float, default=0.001)
    args = parser.parse_args()

    train(args.config, args.batch_size, args.epochs, args.lr)