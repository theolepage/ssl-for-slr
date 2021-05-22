import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from train_evaluate import create_spkid_model
from utils.helpers import load_config, load_dataset, load_model

def load(config_path):
    config, checkpoint_dir, eval_checkpoint_dir = load_config(config_path)

    gens, input_shape, nb_speakers = load_dataset(config,
                                                  eval_checkpoint_dir,
                                                  key='evaluate')

    model = load_model(config, input_shape)

    # Create classifier
    model_evaluate = create_spkid_model(config,
                                        input_shape,
                                        nb_speakers,
                                        model)

    # Load pre-trained model
    last_checkpoint_path = tf.train.latest_checkpoint(eval_checkpoint_dir)
    if last_checkpoint_path:
        model_evaluate.load_weights(last_checkpoint_path)
    else:
        raise Exception('Evaluate: no checkpoints found.')

    # Load trainings history
    history = np.load(checkpoint_dir + '/history.npy', allow_pickle=True).item()
    history_evaluate = np.load(eval_checkpoint_dir + '/history.npy', allow_pickle=True).item()

    _, _, test_gen = gens
    return model, history, model_evaluate, history_evaluate, test_gen

if __name__ == "__main__":
    pass