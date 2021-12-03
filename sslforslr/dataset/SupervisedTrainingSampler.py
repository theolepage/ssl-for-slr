import numpy as np

class SupervisedTrainingSampler:

    def __init__(self, labels, batch_size, nb_labels_per_spk):
        self.labels = labels
        self.batch_size = batch_size
        self.nb_labels_per_spk = nb_labels_per_spk

    def __call__(self, epoch):
        rng = np.random.default_rng(seed=epoch)
        indices = rng.permutation(len(self.labels))

        # Create list of utterances for each speaker
        spk_to_utterances = {}
        for index in indices:
            speaker_label = self.labels[index]
            if speaker_label not in spk_to_utterances:
                spk_to_utterances[speaker_label] = []
            spk_to_utterances[speaker_label].append(index)
        speakers = list(spk_to_utterances.keys())
        speakers.sort()

        # Create training pairs (2 utterances per sample)
        x = []
        y = []
        for i, key in enumerate(speakers):
            utterances = spk_to_utterances[key]

            nb_utt = min(len(utterances), self.nb_labels_per_spk)
            nb_utt = nb_utt - nb_utt % 2

            indices = np.arange(nb_utt).reshape((-1, 2))
            for idx in indices:
                x.append([utterances[i] for i in idx])
                y.append(i)

        # Shuffle indices and avoid having two pairs
        # of the same speaker in the same batch
        x_ = []
        y_ = []
        for i in rng.permutation(len(y)):
            batch_start_i = len(y_) - len(y_) % self.batch_size
            if y[i] not in y_[batch_start_i:]:
                x_.append(x[i])
                y_.append(y[i])

        return x_
