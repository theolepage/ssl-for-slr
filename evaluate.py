import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import tensorflow as tf

from sslforslr.utils.helpers import load_config, load_model
from sslforslr.utils.evaluate import speaker_verification_evaluate

def evaluate(config_path):
    # Load model
    config, checkpoint_dir = load_config(config_path)
    model = load_model(config)

    # Load pre-trained weights
    last_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if last_checkpoint_path:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model.load_weights(last_checkpoint_path)
    else:
        raise Exception('%s has not been trained.' % config['name'])

    eer, min_dcf = speaker_verification_evaluate(model, config)
    print('EER (%):', eer)
    print('minDCF (p=0.01):', min_dcf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    evaluate(args.config)