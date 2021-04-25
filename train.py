import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from models.CPC import CPCModel
from models.LIM import LIMModel
from models.SpeakerIdClassifier import SpeakerIdClassifier

from utils.LibriSpeech import LibriSpeechLoader
from utils.helpers import summary_for_shape

def train():
    sample_frequency = 16000

    frame_length = 20480   # 1.28s at 16kHz (LibriSpeech)
    frame_stride = 20480   # 1.28s at 16kHz (LibriSpeech)

    nb_timesteps = int(frame_length // 160)  # 128
    nb_timesteps_to_predict = 12
    nb_timesteps_for_context = nb_timesteps - nb_timesteps_to_predict

    encoded_dim = 512

    epochs = 30
    batch_size = 64

    max_frames_per_utterance = 1
    nb_speakers = 64
    max_utterances = 1000

    val_ratio=0.2
    test_ratio=0.1

    SEED = 1717

    # Set seed
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load dataset
    lb = LibriSpeechLoader(SEED, "D:/Datasets/LibriSpeech/train-clean-100/*",
                        frame_length=frame_length,
                        frame_stride=frame_stride,
                        max_frames_per_utterance=max_frames_per_utterance,
                        max_speakers=nb_speakers,
                        max_utterances=max_utterances,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio)
    train_gen, val_gen, test_gen = lb.load(batch_size)
    print("Number of training batches:", len(train_gen))
    print("Number of validation batches:", len(val_gen))

    # Create and compile model
    # model = CPCModel(batch_size, encoded_dim, nb_timesteps,
    #                  nb_timesteps_for_context,
    #                  nb_timesteps_to_predict)
    # model.compile(Adam(learning_rate=0.0001)) # 1e-4
    model = LIMModel(batch_size, encoded_dim, nb_timesteps, loss_fn='bce')
    model.compile(Adam(learning_rate=0.001)) # 1e-3
    # summary_for_shape(model.encoder, (frame_length, 1))

    # Setup callbacks
    checkpoint_path = './checkpoints/LIM-{epoch:04d}.ckpt'
    save_callback = ModelCheckpoint(filepath=checkpoint_path,
                                    monitor='loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='loss',
                                   patience=3)

    # Start training
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        callbacks=[save_callback, early_stopping])

    # Save training history
    hist_json_path = './checkpoints/history.json'
    with open(hist_json_path, mode='w') as file:
        pd.DataFrame(history.history).to_json(file)

if __name__ == "__main__":
    train()