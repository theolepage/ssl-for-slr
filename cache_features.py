import argparse
import numpy as np
import kaldiio
from tqdm import tqdm

from shutil import copyfile

def create_features(data_path):
    feats = {}
    wav_scp = kaldiio.load_scp(data_path + '/wav.scp')
    for utterance_id in tqdm(wav_scp):
        sr, data = wav_scp[utterance_id]
        data = data.astype(np.float32)
        feats[utterance_id] = data

    kaldiio.save_ark(data_path + '/feats.ark', feats, scp=data_path + '/feats.scp')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to data folder containing wav.scp file (Kaldi format).')
    args = parser.parse_args()

    #create_features(args.data_path)
    copyfile(args.data_path + '/wav.scp', args.data_path + '/feats.scp')
