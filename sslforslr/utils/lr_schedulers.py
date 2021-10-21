from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class LinearDecay(LearningRateSchedule):
    
    def __init__(
        self,
        initial_lr=0.001,
        interval=5,
        decay_factor=0.05
    ):
        super().__init__()

        self.initial_lr = initial_lr
        self.interval = interval
        self.decay_factor = decay_factor

    def __call__(self, epoch):
        return self.initial_lr - (epoch // self.interval) * self.decay_factor