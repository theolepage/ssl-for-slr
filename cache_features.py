import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import kaldiio
from tqdm import tqdm

from sslforslr.utils.helpers import load_config
from sslforslr.dataset.utils import load_wav, extract_mfcc
from sslforslr.dataset.AudioAugmentation import AudioAugmentation

def cache_features(config_path, output_path, nb_aug=2, limit_length=10*16000):
    output_ark = output_path + '/feats.ark'
    output_scp = output_path + '/feats.scp'

    config, checkpoint_dir = load_config(config_path)
    
    # Load augmentation module
    augment = None
    augment_config = config['dataset'].get('augment', None)
    if augment_config and augment_config.get('enabled', True):
        augment = AudioAugmentation(augment_config)

    # Prepare progress bar
    nb_feats = sum(1 for line in open(config['dataset']['train']))
    nb_feats = nb_feats * (1 + nb_aug)
    pbar = tqdm(total=nb_feats)

    for line in open(config['dataset']['train']):
        utterance_id, file = line.rstrip().split()
        data = load_wav(file, None)
        data = data[:limit_length]

        for i in range(nb_aug):
            data_ = augment(data) if augment else data
            data_ = extract_mfcc(data_)
            utterance_id_ = utterance_id + '_aug_' + str(i)
            kaldiio.save_ark(output_ark,
                             { utterance_id_: data_ },
                             scp=output_scp,
                             append=True)

        data = extract_mfcc(data)
        kaldiio.save_ark(output_ark,
                         { utterance_id: data },
                         scp=output_scp,
                         append=True)

        pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    parser.add_argument('output_path', help='Path to save scp and ark files.')
    args = parser.parse_args()

    cache_features(args.config, args.output_path)