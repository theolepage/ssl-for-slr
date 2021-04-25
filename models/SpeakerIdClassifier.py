import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization

class SpeakerIdClassifier(Model):

    def __init__(self, nb_speakers):
        super(SpeakerIdClassifier, self).__init__()

        self.nb_speakers = nb_speakers

        self.dense1 = Dense(units=256, activation='relu')
        self.dense2 = Dense(units=nb_speakers, activation='softmax')

    def call(self, X):
        return self.dense2(self.dense1(X))