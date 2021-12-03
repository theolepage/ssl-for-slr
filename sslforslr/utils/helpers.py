import copy
from pathlib import Path
import random
import os

import ruamel.yaml
from dacite import from_dict

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD

from sslforslr.configs import Config
from sslforslr.models.cpc import CPCModel, CPCModelConfig
from sslforslr.models.lim import LIMModel, LIMModelConfig
from sslforslr.models.simclr import SimCLRModel, SimCLRModelConfig
from sslforslr.models.moco import MoCoModel, MoCoModelConfig
from sslforslr.models.encoders import CPCEncoder, CPCEncoderConfig
from sslforslr.models.encoders import SincEncoder, SincEncoderConfig
from sslforslr.models.encoders import Wav2SpkEncoder, Wav2SpkEncoderConfig
from sslforslr.models.encoders import XVectorEncoder, XVectorEncoderConfig
from sslforslr.models.encoders import ThinResNet34Encoder, ThinResNet34EncoderConfig
from sslforslr.dataset.AudioDatasetLoader import AudioDatasetLoader

REGISTERED_MODELS = [
    CPCModelConfig,
    LIMModelConfig,
    SimCLRModelConfig,
    MoCoModelConfig
]

REGISTERED_ENCODERS = [
    CPCEncoderConfig,
    SincEncoderConfig,
    XVectorEncoderConfig,
    ThinResNet34EncoderConfig,
    Wav2SpkEncoderConfig
]

def summary_for_shape(model, input_shape):
    x = Input(shape=input_shape)

    model_copy = copy.deepcopy(model)

    model_ = Model(inputs=x, outputs=model_copy.call(x))
    return model_.summary()

def get_sub_config(data, key, registered):
    registered_dict = {c.__NAME__:c for c in registered}
    
    type_ = data[key]['type']
    if type_ not in registered_dict:
        raise (
            Exception('{} {} not supported'
                .format(key.capitalize(), type_))
        )

    return from_dict(registered_dict[type_], data[key])

def load_config(path):
    data = ruamel.yaml.safe_load(open(path, 'r'))
    config = from_dict(Config, data)

    config.encoder = get_sub_config(data, 'encoder', REGISTERED_ENCODERS)
    config.model = get_sub_config(data, 'model', REGISTERED_MODELS)
    
    # Set seed
    seed = config.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create checkpoint dir
    checkpoint_dir = './checkpoints/' + config.name
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    return config, checkpoint_dir

def load_dataset(config):
    dataset = AudioDatasetLoader(config.dataset)
    gens = dataset.load(config.training.batch_size)
    return gens, dataset.get_input_shape(), dataset.nb_classes

def create_encoder(config):
    if config.encoder.__NAME__ == CPCEncoderConfig.__NAME__:
        encoder = CPCEncoder(config.encoder)
    
    elif config.encoder.__NAME__ == SincEncoderConfig.__NAME__:
        encoder = SincEncoder(
            config.dataset.sample_frequency,
            config.encoder)
    
    elif config.encoder.__NAME__ == Wav2SpkEncoderConfig.__NAME__:
        encoder = Wav2SpkEncoder(config.encoder)
    
    elif config.encoder.__NAME__ == XVectorEncoderConfig.__NAME__:
        encoder = XVectorEncoder(config.encoder)
    
    elif config.encoder.__NAME__ == ThinResNet34EncoderConfig.__NAME__:
        encoder = ThinResNet34Encoder(config.encoder)
    
    else:
        raise Exception('Encoder type not supported')

    return encoder

def create_model(config, input_shape):
    encoder = create_encoder(config)
    encoder_output_shape = encoder.compute_output_shape(input_shape)

    if config.model.__NAME__ == CPCModelConfig.__NAME__:
        nb_timesteps = encoder_output_shape[0]
        encoded_dim = encoder_output_shape[1]
        model = CPCModel(encoder, encoded_dim, nb_timesteps, config.model)
    
    elif config.model.__NAME__ == LIMModelConfig.__NAME__:
        nb_timesteps = encoder_output_shape[0]
        model = LIMModel(encoder, nb_timesteps, config.model)
    
    elif config.model.__NAME__ == SimCLRModelConfig.__NAME__:
        model = SimCLRModel(encoder, config.model)
    
    elif config.model.__NAME__ == MoCoModelConfig.__NAME__:
        encoder_k = create_encoder(config)
        model = MoCoModel(encoder, encoder_k, config.model)
    
    else:
        raise Exception('Model type not supported')

    return model

def load_model(config, input_shape):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = create_model(config, input_shape)
    
    # Create optimizer
    opt_type = config.training.optimizer
    if opt_type == 'Adam':
        optimizer = Adam(config.training.learning_rate)
    elif opt_type == 'SGD':
        optimizer = SGD(config.training.learning_rate)
    else:
        raise Exception('Optimizer {} not supported'.format(opt_type))
    
    # Compile and print model
    model.compile(optimizer)
    summary_for_shape(model, input_shape)

    return model
