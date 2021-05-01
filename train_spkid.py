import argparse
from pathlib import Path
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from models.SpeakerIdClassifier import SpeakerIdClassifier
from utils.helpers import load_config

def train(config_path, batch_size, epochs, lr, epochs_ft, lr_ft, layers_ft, no_ft):
    config, encoder, dataset = load_config(config_path)

    checkpoint_dir = './checkpoints/' + config['name']
    last_checkpoint_path = checkpoint_dir + '/training'
    checkpoint_dir_spkid = './checkpoints/' + config['name'] + '_spkid'
    last_checkpoint_spkid_path = checkpoint_dir_spkid + '/training'
    frame_length = config['dataset']['frame_length']
    nb_speakers = config['dataset']['nb_speakers']
    
    # Create subfolder for saving checkpoints
    Path(checkpoint_dir_spkid).mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, test_gen = dataset.load(batch_size, checkpoint_dir)
    print("Number of training batches:", len(train_gen))
    print("Number of validation batches:", len(val_gen))

    # FIXME: load pre-trained encoder+classifier to resume training?

    # Load pre-trained encoder
    # Otherwise, train the encoder from scratch (supervised baseline)
    if Path(last_checkpoint_path).exists():
        encoder.trainable = False
        encoder.load_weights(last_checkpoint_path)

    # Create model: encoder + classifier
    inputs = Input((frame_length, 1))
    inputs_encoded = encoder(inputs)
    outputs = SpeakerIdClassifier(nb_speakers)(inputs_encoded)

    model_spkid = Model(inputs, outputs)
    model_spkid.compile(optimizer=Adam(learning_rate=lr),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    model_spkid.summary()

    # Setup callbacks
    save_callback = ModelCheckpoint(filepath=last_checkpoint_spkid_path,
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)

    # Start training
    history = model_spkid.fit(train_gen,
                              validation_data=val_gen,
                              epochs=epochs,
                              callbacks=[save_callback, early_stopping])

    # Start fine-tuning
    if Path(last_checkpoint_path).exists() and not no_ft:
        # Unfreeze some layers
        for layer in model_spkid.layers[-layers_ft:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True

        model_spkid.compile(optimizer=Adam(learning_rate=lr_ft),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        history_ft = model_spkid.fit(train_gen,
                                    validation_data=val_gen,
                                    epochs=epochs_ft,
                                    callbacks=[save_callback, early_stopping])

    # Save training history
    hist_path = checkpoint_dir_spkid + '/history.npy'
    np.save(hist_path, history.history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=64)
    parser.add_argument('--epochs', help='Number of epochs for training.', type=int, default=50)
    parser.add_argument('--lr', help='Learning rate for training.', type=float, default=0.01)
    parser.add_argument('--epochs_ft', help='Number of epochs for fine tuning.', type=int, default=50)
    parser.add_argument('--lr_ft', help='Learning rate for fine tuning.', type=float, default=0.0001)
    parser.add_argument('--layers_ft', help='Number of layers to unfreeze for fine tuning.', type=int, default=10)
    parser.add_argument('--no_ft', help='Disable fine tuning.')
    args = parser.parse_args()

    train(args.config, args.batch_size,
          args.epochs, args.lr,
          args.epochs_ft, args.lr_ft, args.layers_ft, args.no_ft)