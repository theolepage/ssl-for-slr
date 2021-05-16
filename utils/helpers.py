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
from models.MultiTask import MultiTaskModel
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

    # Set seed
    seed = config['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create subfolder for saving checkpoints
    checkpoint_dir = './checkpoints/' + config['name']
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    batch_size = config['training']['batch_size']
    dataset = LibriSpeechLoader(seed, config['dataset'])
    gens = dataset.load(batch_size, checkpoint_dir)
    print("Number of training batches:", len(gens[0]))
    print("Number of val batches:", len(gens[1]))
    print("Number of test batches:", len(gens[2]))

    # Determine input shape
    # Usually 20480 (1.28s at 16kHz on LibriSpeech) => nb_timesteps = 128
    frame_length = config['dataset']['frames']['length']
    input_shape = (frame_length, 1)

    # Create encoder
    encoder_type = config['encoder']['type']
    encoded_dim = config['encoder']['encoded_dim']
    encoder_weight_regularizer = config['encoder'].get('weight_regularizer', 0.0)
    if encoder_type == 'CPC':
        encoder = CPCEncoder(encoded_dim, encoder_weight_regularizer)
    elif encoder_type == 'Sinc':
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

    # Create model
    model_type = config['model']['type']
    if model_type == 'multitask':
        modules = config['model']['modules']
        model = MultiTaskModel(encoder,
                               input_shape,
                               modules)

        # Add regressor targets to data generator
        for i in range(len(gens)):
            gens[i] = model.add_targets_to_gen(gens[i])
    else:
        model = create_model(config['model'], encoder, input_shape)
    
    # Compile model
    learning_rate = config['training']['learning_rate']
    model.compile(Adam(learning_rate=learning_rate))

    # Print model architecture
    summary_for_shape(model, input_shape)

    return config, model, gens, checkpoint_dir