from tensorflow.keras.utils import Sequence
from audiomentations import Compose
from audiomentations import AddImpulseResponse
from audiomentations import AddBackgroundNoise
from audiomentations import FrequencyMask
from audiomentations import TimeMask
from audiomentations import ClippingDistortion

class AudioAugmentationGenerator(Sequence):
    '''
    Keras generator which adds audio augmentation transformations
    to an existing generator.
    '''

    def __init__(self, gen, config, sample_frequency):
        self.gen = gen
        self.config = config
        self.sample_frequency = sample_frequency
        self.transforms = self.init_transforms()

    def init_transforms(self):
        transforms = []
        for transform in self.config:
            transform_type = transform['type']
            probability = transform.get('p', 0.0)
            path = transform.get('path', "")

            if transform_type == 'add_ir':
                transforms.append(AddImpulseResponse(p=probability,
                                                     ir_path=path,
                                                     leave_length_unchanged=True))
            elif transform_type == 'add_noise':
                transforms.append(AddBackgroundNoise(p=probability,
                                                     sounds_path=path))
            elif transform_type == 'frequency_mask':
                transforms.append(FrequencyMask(p=probability))
            elif transform_type == 'time_mask':
                transforms.append(TimeMask(p=probability))
            elif transform_type == 'clipping':
                transforms.append(ClippingDistortion(p=probability))

        return Compose(transforms)

    def __len__(self):
        return len(self.gen)

    def __getitem__(self, i):
        X, y = self.gen[i]
        batch_size = X.shape[0]
        frame_length = X.shape[1]

        for i in range(batch_size):
            data = self.transforms(X[i].flatten(),
                                   sample_rate=self.sample_frequency)
            X[i] = data.reshape((frame_length, 1))

        return X, y