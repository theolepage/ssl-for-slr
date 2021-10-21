from operator import itemgetter
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve

from sslforslr.dataset.utils import load_wav

def extract_embeddings(model, wav_list_path, frame_length):
    embeddings = {}
    for line in open(wav_list_path):
        utterance_id, file = line.rstrip().split()
        data = load_wav(file, frame_length)
        feats = model(np.expand_dims(data, axis=0))
        embeddings[utterance_id] = feats

    return embeddings

def score_trials(trials_path, embeddings):
    scores, labels = [], []
    for line in open(trials_path):
        a, b, target = line.rstrip().split(' ')

        score = 1 - cosine(embeddings[a], embeddings[b])
        label = 1 if (target == 'target') else 0

        scores.append(score)
        labels.append(label)

    return scores, labels

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr    
    idxE = np.nanargmin(np.abs(fnr - fpr))
    eer  = max(fpr[idxE], fnr[idxE]) * 100
    return eer

def compute_error_rates(scores, labels):
      # Sort scores from smallest to largest.
      # Scores are the thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      labels = [labels[i] for i in sorted_indexes]
      
      # Determine false negative rates and false positive rates for each threshold.
      fnrs = []
      fprs = []
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i -1] + labels[i])
              fprs.append(fprs[i -1] + 1 - labels[i])

      fnrs_norm = sum(labels)
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      fprs_norm = len(labels) - fnrs_norm
      fprs = [1 - x / float(fprs_norm) for x in fprs]

      return fnrs, fprs

def compute_min_dcf(fnrs, fprs, p_target=0.01, c_miss=1, c_fa=1):
    # Equations are from Section 3 of
    # NIST 2016 Speaker Recognition Evaluation Plan

    # Equation (2)
    min_c_det = float("inf")
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
    
    # Equations (3) and (4)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf

def speaker_verification_evaluate(model, config, round_val=5):
    test_list_path = config['dataset']['test']
    trials_path = config['dataset']['trials']
    frame_length = config['dataset']['frame_length']

    embeddings = extract_embeddings(model, test_list_path, frame_length)
    scores, labels = score_trials(trials_path, embeddings)

    eer = round(compute_eer(scores, labels), round_val)
    fnrs, fprs = compute_error_rates(scores, labels)
    min_dcf = round(compute_min_dcf(fnrs, fprs), round_val)

    return eer, min_dcf