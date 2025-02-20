'''FUNCTIONS THAT ARE USED TO TRAIN THE MODEL'''

import argparse
import LoadData
import os
import tensorflow as tf
from datetime import datetime
import pickle
import optuna

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--model', '-m', type=str, default="SiameseModel", help='Model name')
    parser.add_argument('--data', '-d', type=str, default="./data/benchmark.db", help='Path to the data file')
    parser.add_argument('--output', '-o', type=str, default="./log/", help='Output directory for logs and models')
    parser.add_argument('--overwrite', '-w', action='store_true', help='Overwrite the existing data')
    parser.add_argument('--metrics', '-m', type=str, default='mse', help='Metrics to use for training')
    # TODO: Logging location and model saving location
    return parser.parse_args()

# Load data using tensorflow dataset
def LoadDataTensorflow(DB_FILE, batch_size = 32, split_size = (0.7, 0.1, 0.2), sqlite_batch = 1000, overwrite = False):
    print(f"===== Loading {DB_FILE} into Tensorflow Dataset =====")

    # Load the dataset
    train_set, val_set, test_set = LoadData.CreateTensorflowDataset(DB_FILE, batch_size, split_size, sqlite_batch, overwrite)

    print(f"===== Finished Tensorflow Dataset =====")
    return train_set, val_set, test_set

# TODO: Load data using pytorch dataset
def LoadDataPytorch(DB_File, batch_size = 32, split_size = (0.7, 0.1, 0.2), sqlite_batch = 1000):
    raise NotImplementedError("LoadDataPytorch is not implemented yet")
    print(f"Loading data from {DB_File} into pytorch dataset")
    print(f"Finished loading data into pytorch dataset")


# Trains the model using tensorflow
def TrainTensorflowModel(model, train_set, val_set, epochs, learning_rate):
    print(f"Training the model using tensorflow")
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_set, epochs=epochs, validation_data=val_set)

    print(f"Finished training the model using tensorflow")
    return model, history

# Check if path is valid
def CheckPath(path, library, model):
    # If path is a directory, give output file a default name
    if os.path.isdir(path):
        current_time = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        if library == 'tensorflow':
            path = os.path.join(path, f"{model}_{current_time}.keras")
        elif library == 'pytorch':
            path = os.path.join(path, f"{model}_{current_time}.pth")
        return path
    else:
        if os.path.exists(path):
            print(f"ERROR: {path} exists. Please provide a directory or a new file name")
            return False
        elif library == 'tensorflow' and path.endswith('.keras'):
            return path
        elif library == 'pytorch' and path.endswith('.pth'):
            return path
        else:
            return False

# Dump the model
def DumpModel(model, path, library, history = None):
    if library == 'tensorflow':
        # Save the model
        print(f"Saving the model to {path}")
        model.save(path)

        # Save the history
        dir = os.path.dirname(path)
        file = os.path.basename(path).split('.keras')[0]
        filename = os.path.join(dir, f"{file}.history")
        print(f"Saving the history to {filename}")

        with open(filename, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise NotImplementedError(f"Dumping for {library} is not implemented yet")

def getTestNonZeroData(DB_FILE, batch_size = 32, sqlite_batch = 1000):
    condition = "WHERE AlignmentScore != 0"

    dataset = LoadData.LoadDataset(DB_FILE, batch_size, sqlite_batch, condition=condition)
    return dataset

# Evaluate the model given the dataset
def EvaluateModel(model, dataset):
    print(f"===== Evaluating the model =====")
    evaluate = model.evaluate(dataset)

    return evaluate



