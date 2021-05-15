import json
import copy
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from models.CPCEncoder import CPCEncoder
from models.SincEncoder import SincEncoder
from models.CPC import CPCModel
from models.LIM import LIMModel
from .LibriSpeech import LibriSpeechLoader

def summary_for_shape(model, input_shape):
    x = Input(shape=input_shape)

    model_copy = copy.deepcopy(model)

    model_ = Model(inputs=x, outputs=model_copy.call(x))
    return model_.summary()

def load_config(config_path, name_suffix=''):
    # Load config file
    with open(config_path) as config_file:
        config = json.load(config_file)

    seed = config['seed']
    learning_rate = config['training']['learning_rate']
    model_type = config['model']['type']
    encoder_type = config['encoder']['type']
    encoded_dim = config['encoder']['encoded_dim']
    
    # Create encoder
    encoder_weight_regularizer = config['encoder'].get('weight_regularizer', 0.0)
    if encoder_type == 'CPC':
        encoder = CPCEncoder(encoded_dim, encoder_weight_regularizer)
    elif encoder_type == 'CPC':
        sample_frequency = config['dataset']['sample_frequency']
        skip_connections_enabled = config['encoder'].get('skip_connections_enabled')
        rnn_enabled = config['encoder'].get('rnn_enabled')
        encoder = SincEncoder(encoded_dim,
                              sample_frequency,
                              skip_connections_enabled,
                              rnn_enabled,
                              encoder_weight_regularizer)
    else:
        raise Exception('Config: encoder {} is not supported.'.format(encoder_type))

    # Usually 20480 (1.28s at 16kHz on LibriSpeech)
    # => nb_timesteps = 128
    frame_length = config['dataset']['frames']['length']
    input_shape = (frame_length, 1)
    nb_timesteps = encoder.compute_output_shape(input_shape)[0]

    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load dataset
    dataset = LibriSpeechLoader(seed, config['dataset'])

    # Create and compile model
    model_weight_regularizer = config['model'].get('weight_regularizer', 0.0)
    if model_type == 'CPC':
        nb_timesteps_to_predict = config['model']['nb_timesteps_to_predict']
        model = CPCModel(encoder,
                         encoded_dim,
                         nb_timesteps,
                         nb_timesteps_to_predict,
                         model_weight_regularizer)
        model.compile(Adam(learning_rate=learning_rate))
    elif model_type == 'LIM':
        loss_fn = config['model']['loss_fn']
        model = LIMModel(encoder,
                         nb_timesteps,
                         loss_fn,
                         model_weight_regularizer)
        model.compile(Adam(learning_rate=learning_rate))
    else:
        raise Exception('Config: model {} is not supported.'.format(model_type))
    
    summary_for_shape(model, (frame_length, 1))

    return config, model, dataset