from tf.keras.callbacks import Callback

from sslforslr.utils.evaluate import speaker_verification_evaluate

class SVMetricsCallback(Callback):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def on_epoch_end(self, epoch, logs):
        eer = speaker_verification_evaluate(self.model, self.config)
        logs['EER'] = eer