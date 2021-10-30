from tensorflow.keras.callbacks import Callback

from sslforslr.utils.evaluate import speaker_verification_evaluate

class SVMetricsCallback(Callback):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def on_epoch_end(self, epoch, logs):
        eer, min_dcf_001, min_dcf_005 = speaker_verification_evaluate(
            self.model,
            self.config
        )
        
        print('EER (%):', eer)
        print('minDCF (p=0.01):', min_dcf_001)
        print('minDCF (p=0.05):', min_dcf_005)

        logs.update({
            'test_eer': eer,
            'test_min_dcf_001': min_dcf_001,
            'test_min_dcf_005': min_dcf_005
        })