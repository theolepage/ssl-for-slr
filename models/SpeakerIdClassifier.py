import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

class SpeakerIdClassifier(Model):

    def __init__(self, nb_speakers):
        super(SpeakerIdClassifier, self).__init__()

        self.nb_speakers = nb_speakers

        self.flatten = Flatten()
        self.dense1 = Dense(units=256)
        self.dense2 = Dense(units=nb_speakers, activation='softmax')

    def call(self, X):
        X = self.flatten(X)
        X = self.dense1(X)
        X = self.dense2(X)
        return X