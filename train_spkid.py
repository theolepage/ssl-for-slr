import argparse
from pathlib import Path
import pandas as pd

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from models.SpeakerIdClassifier import SpeakerIdClassifier
from utils.helpers import load_config

def train(config_path, nb_epochs, batch_size):
    config, encoder, dataset, checkpoint_dir = load_config(config_path,
                                                           create_checkpoint_dir=True)

    frame_length = config['dataset']['frame_length']
    nb_speakers = config['dataset']['nb_speakers']
    last_checkpoint_encoder_path = checkpoint_dir + '/' + config['name'] + '.ckpt'
    last_checkpoint_spkid_path = checkpoint_dir + '/' + config['name'] + '-spkid.ckpt'
    
    train_gen, val_gen, test_gen = dataset.load(batch_size)
    print("Number of training batches:", len(train_gen))
    print("Number of validation batches:", len(val_gen))

    # FIXME: load pre-trained encoder+classifier to resume training?

    # Load pre-trained encoder
    # Otherwise, train the encoder from scratch (supervised baseline)
    if Path(last_checkpoint_encoder_path).exists():
        encoder.trainable = False
        encoder.load_weights(last_checkpoint_encoder_path)

    # Create model: encoder + classifier
    inputs = Input((frame_length, 1))
    inputs_encoded = encoder(inputs)
    outputs = SpeakerIdClassifier(nb_speakers)(inputs_encoded)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Setup callbacks
    save_callback = ModelCheckpoint(filepath=last_checkpoint_spkid_path,
                                    monitor='loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='loss',
                                   patience=3)

    # Start training
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=nb_epochs,
                        callbacks=[save_callback, early_stopping])

    # Save training history
    hist_json_path = checkpoint_dir + '/history-spkid.json'
    with open(hist_json_path, mode='w') as file:
        pd.DataFrame(history.history).to_json(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    parser.add_argument('--nb_epochs', help='Number of training epochs.', default=10)
    parser.add_argument('--batch_size', help='Batch size.', default=64)
    args = parser.parse_args()

    train(args.config, args.nb_epochs, args.batch_size)