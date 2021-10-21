import json
import copy
from pathlib import Path
import random
import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import CosineDecay

from sslforslr.models.cpc import CPCModel
from sslforslr.models.lim import LIMModel
from sslforslr.models.simclr import SimCLRModel
from sslforslr.models.moco import MoCoModel
from sslforslr.models.encoders import CPCEncoder, SincEncoder, Wav2SpkEncoder, XVectorEncoder, ThinResNet34Encoder
from sslforslr.dataset.KaldiDatasetLoader import KaldiDatasetLoader

def summary_for_shape(model, input_shape):
    x = Input(shape=input_shape)

    model_copy = copy.deepcopy(model)

    model_ = Model(inputs=x, outputs=model_copy.call(x))
    return model_.summary()

def load_config(config_path):
    # Load config file
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Set seed
    seed = config['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create checkpoint dir
    checkpoint_dir = './checkpoints/' + config['name']
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    return config, checkpoint_dir

def load_dataset(config):
    dataset_config = config['dataset']
    batch_size = config['training']['batch_size']
    seed = config['seed']

    dataset = KaldiDatasetLoader(seed, dataset_config)
    gens = dataset.load(batch_size)

    return gens

def create_encoder(config):
    encoder_type = config['encoder']['type']
    encoded_dim = config['encoder']['encoded_dim']
    encoder_weight_regularizer = config['encoder'].get('weight_regularizer', 0.0)

    if encoder_type == 'CPC':
        encoder = CPCEncoder(encoded_dim, encoder_weight_regularizer)
    elif encoder_type == 'Sinc':
        sample_frequency = config['training']['dataset']['sample_frequency']
        skip_connections_enabled = config['encoder'].get('skip_connections_enabled')
        rnn_enabled = config['encoder'].get('rnn_enabled')
        frame_length = config['training']['dataset']['frames']['length']
        encoder = SincEncoder(encoded_dim,
                              frame_length,
                              sample_frequency,
                              skip_connections_enabled,
                              rnn_enabled,
                              encoder_weight_regularizer)
    elif encoder_type == 'Wav2Spk':
        encoder = Wav2SpkEncoder(encoded_dim, encoder_weight_regularizer)
    elif encoder_type == 'XVector':
        encoder = XVectorEncoder(encoded_dim, encoder_weight_regularizer)
    elif encoder_type == 'ThinResNet34':
        encoder = ThinResNet34Encoder(encoded_dim, encoder_weight_regularizer)
    else:
        raise Exception('Encoder {} is not supported.'.format(encoder_type))

    return encoder

def create_model(config, input_shape):
    model_config = config['model']
    model_type = model_config['type']
    weight_regularizer = model_config.get('weight_regularizer', 0.0)

    encoder = create_encoder(config)
    encoder_output_shape = encoder.compute_output_shape(input_shape)

    if model_type == 'CPC':
        nb_timesteps = encoder_output_shape[0]
        encoded_dim = encoder_output_shape[1]
        nb_timesteps_to_predict = model_config['nb_timesteps_to_predict']
        bidirectional = model_config.get('bidirectional', False)
        context_network = model_config.get('context_network', {})
        model = CPCModel(encoder,
                         encoded_dim,
                         nb_timesteps,
                         nb_timesteps_to_predict,
                         bidirectional,
                         context_network,
                         weight_regularizer)
    elif model_type == 'LIM':
        nb_timesteps = encoder_output_shape[0]
        loss_fn = model_config['loss_fn']
        context_length = model_config.get('context_length', 1)
        model = LIMModel(encoder,
                         nb_timesteps,
                         loss_fn,
                         context_length,
                         weight_regularizer)
    elif model_type == 'SimCLR':
        channel_loss_factor = model_config.get('channel_loss_factor', 0.1)
        model = SimCLRModel(encoder, channel_loss_factor, weight_regularizer)
    elif model_type == 'MoCo':
        encoder_k = create_encoder(config)
        model = MoCoModel(encoder, encoder_k, model_config, weight_regularizer)
    else:
        raise Exception('Model {} is not supported.'.format(model_type))

    return model

def load_model(config):
    # Determine input shape
    # Usually 20480 (1.28s at 16kHz on LibriSpeech) => nb_timesteps = 128
    input_shape = (config['dataset']['frame_length'], 1)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = create_model(config, input_shape)
    
    # Create learning rate scheduler
    learning_rate = config['training']['learning_rate']
    if isinstance(learning_rate, dict):
        lr_type = learning_rate['scheduler']
        lr_start = learning_rate['start']
        lr_end = learning_rate['end']
        learning_rate = CosineDecay(initial_learning_rate=lr_start,
                                    decay_steps=config['training']['epochs'],
                                    alpha=lr_end/lr_start)

    # Create optimizer
    optimizer_config = config['training'].get('optimizer', {})
    opt_type = optimizer_config.get('type', 'Adam')
    if opt_type == 'Adam':
        optimizer = Adam(learning_rate)
    elif opt_type == 'SGD':
        momentum = optimizer_config.get('momentum', 0.0)
        optimizer = SGD(learning_rate, momentum)
    else:
        raise Exception('Optimizer {} is not supported.'.format(opt_type))
    
    # Compile and print model
    run_eagerly = config['training'].get('run_eagerly', False)
    model.compile(optimizer, run_eagerly=run_eagerly)
    summary_for_shape(model, input_shape)

    return model
