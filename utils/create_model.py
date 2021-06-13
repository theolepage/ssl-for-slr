from models.CPC import CPCModel
from models.LIM import LIMModel
from models.wave2vec2.Wave2Vec2 import Wave2Vec2Model
from models.wave2vec2.Wave2Vec2Config import Wave2Vec2Config
from models.vqwave2vec.VQWave2Vec import VQWave2VecModel
from models.vqwave2vec.VQWave2VecConfig import VQWave2VecConfig

def create_model(model_config, encoder, input_shape):
    model_type = model_config['type']
    weight_regularizer = model_config.get('weight_regularizer', 0.0)

    if model_type == 'wave2vec2':
        config = Wave2Vec2Config()
        model = Wave2Vec2Model(config)
        return model
    elif model_type == 'vq-wave2vec':
        config = VQWave2VecConfig()
        model = VQWave2VecModel(config)
        return model

    encoder_output_shape = encoder.compute_output_shape(input_shape)
    nb_timesteps = encoder_output_shape[0]
    encoded_dim = encoder_output_shape[1]

    if model_type == 'CPC':
        nb_timesteps_to_predict = model_config['nb_timesteps_to_predict']
        bidirectional = model_config.get('bidirectional', False)
        model = CPCModel(encoder,
                         encoded_dim,
                         nb_timesteps,
                         nb_timesteps_to_predict,
                         bidirectional,
                         weight_regularizer)
    elif model_type == 'LIM':
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