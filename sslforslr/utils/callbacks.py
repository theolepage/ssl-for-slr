from tensorflow.keras.callbacks import Callback

from sslforslr.utils.evaluate import extract_embeddings, evaluate

class SVMetricsCallback(Callback):

    def __init__(self, config):
        super().__init__()

        self.config = config

    def on_epoch_end(self, epoch, logs):
        embeddings = extract_embeddings(self.model, self.config.dataset)

        eer, min_dcf_001, _, _ = evaluate(embeddings, self.config.dataset.trials)

        print('EER (%):', eer)
        print('minDCF (p=0.01):', min_dcf_001)

        logs.update({
            'test_eer': eer,
            'test_min_dcf_001': min_dcf_001
        })