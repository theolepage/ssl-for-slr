import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import pickle
import tensorflow as tf

from sslforslr.utils.helpers import load_config, load_dataset, load_model
from sslforslr.utils.evaluate import extract_embeddings, evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    config, checkpoint_dir = load_config(args.config)
    (train_gen, val_gen), input_shape = load_dataset(config)
    model = load_model(config, input_shape)

    # Load pre-trained weights
    last_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if last_checkpoint_path:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model.load_weights(last_checkpoint_path)
    else:
        raise Exception('%s has not been trained.' % config['name'])

    # Exract and save embeddings
    embeddings = extract_embeddings(model, config.dataset.test, config.dataset)
    with open(checkpoint_dir + '/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Show metrics on speaker verification
    eer, min_dcf_001, _, _ = evaluate(embeddings, config.dataset.trials)
    print('EER (%):', eer)
    print('minDCF (p=0.01):', min_dcf_001)