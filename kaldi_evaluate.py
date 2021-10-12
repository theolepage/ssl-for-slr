import argparse
import numpy as np
import kaldiio
from tqdm import tqdm

from scipy.spatial.distance import cosine

def extract_embeddings(trials_path, embeddings_path, scores_path):
    out = open(scores_path, "w")
    d = kaldiio.load_scp(embeddings_path)
    for line in tqdm(open(trials_path)):
        info = line.rstrip().split(' ')
        a = d[info[0]]
        b = d[info[1]]
        target = (info[2] == 'target')

        # Compute cosine distance
        dist = 1 - cosine(a, b)

        # Write to output file
        out.write(info[0] + ' ' + info[1] + ' ' + str(dist) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('trials_path', help='Path to test set trials list.')
    parser.add_argument('embeddings_path', help='Path to scp file containing embeddings for speakers in the test set.')
    parser.add_argument('scores_path', help='Path to output file containing scores.')
    args = parser.parse_args()

    extract_embeddings(args.trials_path, args.embeddings_path, args.scores_path)