import json
import copy
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from models.CPC import CPCModel
from models.LIM import LIMModel
from .LibriSpeech import LibriSpeechLoader

def summary_for_shape(model, input_shape):
    x = Input(shape=input_shape)

    model_copy = copy.deepcopy(model)

    model_ = Model(inputs=x, outputs=model_copy.call(x))
    return model_.summary()

def load_config(config_path, create_checkpoint_dir=False):
    # Load config file
    with open(config_path) as config_file:
        config = json.load(config_file)

    seed = config['seed']
    batch_size = config['training']['batch_size']
    encoded_dim = config['model']['encoded_dim']
    model_type = config['model']['type']
    
    # Usually 20480 (1.28s at 16kHz on LibriSpeech)
    # => nb_timesteps = 128
    frame_length = config['dataset']['frame_length']
    nb_timesteps = int(frame_length // 160)
    
    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load dataset
    dataset = LibriSpeechLoader(seed=seed, config=config['dataset'])

    # Create and compile model
    if (model_type == 'CPC'):
        nb_timesteps_to_predict = config['model']['nb_timesteps_to_predict']

        model = CPCModel(batch_size, encoded_dim, nb_timesteps, nb_timesteps_to_predict)
        model.compile(Adam(learning_rate=0.0001)) # 1e-4

    elif (model_type == 'LIM'):
        loss_fn = config['model']['loss_fn']

        model = LIMModel(batch_size, encoded_dim, nb_timesteps, loss_fn)
        model.compile(Adam(learning_rate=0.001)) # 1e-3
    
    summary_for_shape(model.encoder, (frame_length, 1))

    # Create subfolder for saving checkpoints
    checkpoint_dir = './checkpoints/' + config['name']
    if (create_checkpoint_dir):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    return config, model, dataset, checkpoint_dir