import argparse
from pathlib import Path
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from utils.helpers import load_config

def train(config_path):
    config, model, dataset, checkpoint_dir = load_config(config_path,
                                                         create_checkpoint_dir=True)

    batch_size = config['training']['batch_size']
    nb_epochs = config['training']['epochs']
    last_checkpoint_path = checkpoint_dir + '/' + config['name'] + '.ckpt'
    
    train_gen, val_gen, test_gen = dataset.load(batch_size)
    print("Number of training batches:", len(train_gen))
    print("Number of validation batches:", len(val_gen))

    # Load weights
    if (Path(last_checkpoint_path).exists()):
        raise Exception('Train: model has already been trained.')
        # FIXME: model.load_weights(last_checkpoint_path)

    # Setup callbacks
    save_callback = ModelCheckpoint(filepath=last_checkpoint_path,
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
    hist_json_path = checkpoint_dir + '/history.json'
    with open(hist_json_path, mode='w') as file:
        pd.DataFrame(history.history).to_json(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train(args.config)