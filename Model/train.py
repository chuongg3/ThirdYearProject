import argparse
import LoadData
import os
import tensorflow as tf
from datetime import datetime
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--model', '-m', type=str, default="SiameseModel", help='Model name')
    parser.add_argument('--data', '-d', type=str, default="./data/benchmark.db", help='Path to the data file')
    # parser.add_argument('--library', '-lib', type=str, default="tensorflow", help='Library to use for training')
    parser.add_argument('--output', '-o', type=str, default="./log/", help='Output directory for logs and models')
    # TODO: Logging location and model saving location
    return parser.parse_args()

# Load data using tensorflow dataset
def LoadDataTensorflow(DB_FILE, batch_size = 32, split_size = (0.7, 0.1, 0.2), sqlite_batch = 1000):
    print(f"===== Loading {DB_FILE} into Tensorflow Dataset =====")

    # Load the dataset
    train_set, val_set, test_set = LoadData.CreateTensorflowDataset(DB_FILE, batch_size, split_size, sqlite_batch)

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

# Main Function
if __name__ == "__main__":
    args = parse_args()

    # ===== Loading the Model and Data =====
    Models = ['SiameseModel']   # Update this list with new models
    MODEL = args.model
    print(f"Attempting to load {MODEL}")
    if args.model == "SiameseModel":
        library = 'tensorflow'
        train_set, val_set, test_set = LoadDataTensorflow(args.data, args.batch_size)
        from models.SiameseModel import get_model
    else:
        print(f"ERROR: {args.model} is not a valid model")
        exit(1)

    # ===== Verify Path Validity =====
    PATH = CheckPath(args.output, library, MODEL)
    print(f"Checking path validity for {PATH}")
    if not PATH:
        print(f"ERROR: {PATH} is not a valid path")
        exit(1)

    # ===== Train the Tensorflow Model =====
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size} and learning rate {args.learning_rate}")
    if (library == 'tensorflow'):
        model = get_model(learning_rate=args.learning_rate)
        model, history = TrainTensorflowModel(model, train_set, val_set, args.epochs, args.learning_rate)
        
        # Dump the model and the history
        DumpModel(model, PATH, library, history.history)

        # Test the Model
        print("===== ALL TEST DATA =====")
        data = EvaluateModel(model, test_set)
        print(f"Test Loss: {data[0]}")
        print(f"Test MSE: {data[1]}")

        print("===== NON-ZERO TEST DATA =====")
        paths = LoadData.getTempDirectories(args.data)
        non_zero_data = getTestNonZeroData(paths[2], args.batch_size)
        non_zero_data = EvaluateModel(model, non_zero_data)
        print(f"Test Loss: {non_zero_data[0]}")
        print(f"Test MSE: {non_zero_data[1]}")

    else:
        raise NotImplementedError("Training for pytorch is not implemented yet")
