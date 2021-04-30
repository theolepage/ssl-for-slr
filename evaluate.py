import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from models.SpeakerIdClassifier import SpeakerIdClassifier
from utils.helpers import load_config

def load(config_path):
    config, model, dataset = load_config(config_path)

    checkpoint_dir = './checkpoints/' + config['name']
    checkpoint_dir_spkid = './checkpoints/' + config['name'] + '_spkid'
    batch_size = config['training']['batch_size']
    nb_speakers = config['dataset']['nb_speakers']
    frame_length = config['dataset']['frame_length']

    _, _, test_gen = dataset.load(batch_size, checkpoint_dir)

    history = np.load(checkpoint_dir + '/history.npy', allow_pickle=True).item()
    history_spkid = np.load(checkpoint_dir_spkid + '/history.npy', allow_pickle=True).item()

    # Create model: model + classifier
    inputs = Input((frame_length, 1))
    inputs_encoded = model(inputs)
    outputs = SpeakerIdClassifier(nb_speakers)(inputs_encoded)

    model_spkid = Model(inputs, outputs)

    # Load pre-trained model
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir_spkid)
    if Path(checkpoint_dir_spkid).exists():
        model_spkid.load_weights(checkpoint_path)
    else:
        raise Exception('Evaluate: model {}-spkid has no checkpoints.'.format(config['name']))

    model_spkid.compile(optimizer=Adam(learning_rate=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    model_spkid.summary()

    return model, history, model_spkid, history_spkid, test_gen

if __name__ == "__main__":
    pass