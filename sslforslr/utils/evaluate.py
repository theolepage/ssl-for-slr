from tqdm import tqdm
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve

def extract_embeddings(model, wav_list_path, frame_length):
    embeddings = {}
    for line in tqdm(open(wav_list_path)):
        utterance_id, file = line.rstrip().split()
        
        sample, sr = sf.read(file)
        data = sample.reshape((len(sample), 1))

        assert len(data) >= frame_length
        offset = np.random.randint(0, len(data) - frame_length + 1)
        data = data[offset:offset+frame_length]
        
        feats = model(np.expand_dims(data, axis=0))
        embeddings[utterance_id] = feats

    return embeddings

def score_trials(trials_path, embeddings):
    scores, labels = [], []
    for line in tqdm(open(trials_path)):
        a, b, target = line.rstrip().split(' ')

        score = 1 - cosine(embeddings[a], embeddings[b])
        label = 1 if (target == 'target') else 0

        scores.append(score)
        labels.append(label)

    return scores, trials

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr    
    idxE = np.nanargmin(np.abs(fnr - fpr))
    eer  = max(fpr[idxE], fnr[idxE]) * 100
    return eer

def speaker_verification_evaluate(model, config):
    test_list_path = config['dataset']['test']
    trials_path = config['dataset']['trials']
    frame_length = config['dataset']['frame_length']

    embeddings = extract_embeddings(model, test_list_path, frame_length)
    scores, labels = score_trials(trials_path, embeddings)
    eer = compute_eer(scores, labels)
    return eer