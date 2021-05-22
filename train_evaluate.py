import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from models.SpeakerIdClassifier import SpeakerIdClassifier
from utils.helpers import load_config, load_dataset, load_model

def create_spkid_model(config, input_shape, nb_speakers, model):
    learning_rate = config['evaluate']['learning_rate']

    inputs = Input(input_shape)
    inputs_encoded = model(inputs)
    outputs = SpeakerIdClassifier(nb_speakers)(inputs_encoded)

    model_spkid = Model(inputs, outputs)
    model_spkid.compile(optimizer=Adam(learning_rate=learning_rate),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    model_spkid.summary()

    return model_spkid

def train_evaluate(config_path):
    config, checkpoint_dir, eval_checkpoint_dir = load_config(config_path)

    # Prevent re-training model
    if tf.train.latest_checkpoint(eval_checkpoint_dir):
        raise Exception('%s already contains checkpoints.' % eval_checkpoint_dir)

    gens, input_shape, nb_speakers = load_dataset(config,
                                                  eval_checkpoint_dir,
                                                  key='evaluate')

    model = load_model(config, input_shape)

    # Load pre-trained weights
    last_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if last_checkpoint_path:
        model.load_weights(last_checkpoint_path)
    print('Loading pretrained model: ', last_checkpoint_path is not None)
    
    model.trainable = config['evaluate'].get('train_encoder', True)

    # Create classifier
    model_evaluate = create_spkid_model(config,
                                        input_shape,
                                        nb_speakers,
                                        model)

    # Setup callbacks
    save_callback = ModelCheckpoint(filepath=eval_checkpoint_dir + '/training',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)

    # Start training
    train_gen, val_gen, test_gen = gens
    nb_epochs = config['evaluate']['epochs']
    history = model_evaluate.fit(train_gen,
                                 validation_data=val_gen,
                                 epochs=nb_epochs,
                                 callbacks=[save_callback, early_stopping])

    # Save training history
    hist_path = eval_checkpoint_dir + '/history.npy'
    np.save(hist_path, history.history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train_evaluate(args.config)