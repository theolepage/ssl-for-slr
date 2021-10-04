import json
import copy
from pathlib import Path
import random
import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from sslforslr.models.cpc import CPCModel
from sslforslr.models.lim import LIMModel
from sslforslr.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from sslforslr.models.vqwav2vec import VQWav2VecModel, VQWav2VecConfig
from sslforslr.models.multitask import MultiTaskModel
from sslforslr.models.encoders import CPCEncoder, SincEncoder, Wav2SpkEncoder, XVectorEncoder
from sslforslr.dataset.AudioDatasetLoader import AudioDatasetLoader
from sslforslr.dataset.KaldiDatasetLoader import KaldiDatasetLoader
from sslforslr.dataset.AudioAugmentationGenerator import AudioAugmentationGenerator

def summary_for_shape(model, input_shape):
    x = Input(shape=input_shape)

    model_copy = copy.deepcopy(model)

    model_ = Model(inputs=x, outputs=model_copy.call(x))
    return model_.summary()

def load_config(config_path, evaluate=False):
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
    
    eval_checkpoint_dir = None
    if evaluate:
        eval_checkpoint_dir = checkpoint_dir + '___eval'
        Path(eval_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    return config, checkpoint_dir, eval_checkpoint_dir

def load_dataset(config, checkpoint_dir, key='training'):
    dataset_config = config[key]['dataset']
    batch_size = config[key]['batch_size']
    seed = config['seed']

    if dataset_config['type'] == 'Kaldi':
        dataset = KaldiDatasetLoader(seed, dataset_config)
    else:
        dataset = AudioDatasetLoader(seed, dataset_config)
    gens, nb_categories = dataset.load(batch_size, checkpoint_dir)

    # Add data augmentation generator on top of generators
    if 'data_augment' in config[key]:
        data_augment_config = config[key]['data_augment']
        sample_frequency = config[key]['dataset']['sample_frequency']
        for i in range(len(gens)):
            gens[i] = AudioAugmentationGenerator(gens[i],
                                                 data_augment_config,
                                                 sample_frequency)

    print("Number of training batches:", len(gens[0]))
    print("Number of val batches:", len(gens[1]))
    if len(gens) >= 3:
        print("Number of test batches:", len(gens[2]))

    # Determine input shape
    # Usually 20480 (1.28s at 16kHz on LibriSpeech) => nb_timesteps = 128
    frame_length = config[key]['dataset']['frames']['length']
    input_shape = (frame_length, 1)

    return gens, input_shape, nb_categories

def create_encoder(config):
    encoder_type = config['encoder']['type']

    if encoder_type in ['wav2vec2', 'vq-wav2vec']:
        encoder = None
        return encoder

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
    else:
        raise Exception('Config: encoder {} is not supported.'.format(encoder_type))

    return encoder

def create_model(model_config, encoder, input_shape):
    model_type = model_config['type']
    weight_regularizer = model_config.get('weight_regularizer', 0.0)

    if model_type == 'multitask':
        modules = model_config['modules']
        return MultiTaskModel(encoder, input_shape, modules)
    elif model_type == 'wav2vec2':
        config = Wav2Vec2Config()
        return Wav2Vec2Model(config)
    elif model_type == 'vq-wav2vec':
        config = VQWav2VecConfig()
        return VQWav2VecModel(config)

    encoder_output_shape = encoder.compute_output_shape(input_shape)
    encoded_dim = encoder_output_shape[1]

    if model_type == 'CPC':
        nb_timesteps = encoder_output_shape[0]
        nb_timesteps_to_predict = model_config['nb_timesteps_to_predict']
        context_dim = model_config['context_dim']
        bidirectional = model_config.get('bidirectional', False)
        model = CPCModel(encoder,
                         encoded_dim,
                         context_dim,
                         nb_timesteps,
                         nb_timesteps_to_predict,
                         bidirectional,
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
    else:
        raise Exception('Config: model {} is not supported.'.format(model_type))

    return model

def load_model(config, input_shape):
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        encoder = create_encoder(config)
        model = create_model(config['model'], encoder, input_shape)
    
    # Compile and print model
    learning_rate = config['training']['learning_rate']
    model.compile(Adam(learning_rate=learning_rate))
    summary_for_shape(model, input_shape)

    return model
