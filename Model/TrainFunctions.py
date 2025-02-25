'''FUNCTIONS THAT ARE USED TO TRAIN THE MODEL'''

import os
import pickle
import LoadData
from datetime import datetime
import time

# TODO: Load data using pytorch dataset
def LoadDataPytorch(DB_File, batch_size = 32, split_size = (0.7, 0.1, 0.2), sqlite_batch = 1000):
    raise NotImplementedError("LoadDataPytorch is not implemented yet")
    print(f"Loading data from {DB_File} into pytorch dataset")
    print(f"Finished loading data into pytorch dataset")


# Trains the model using tensorflow
def TrainTensorflowModel(model, train_set, val_set, epochs = 10, batch_size = 32):
    print(f"Training the model using tensorflow")

    # Fit the model
    startTime = time.time()
    history = model.fit(train_set, epochs=epochs, validation_data=val_set)
    totalTime = time.time() - startTime
    print(f"Time taken for the model to run: {time.strftime('%H:%M:%S', time.gmtime(totalTime))}")

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

# Dumps the histories into the same directory as where the model is being saved
# Used by HyperParameterTraining
def DumpHistory(histories, modelPath):
    # Save the history
    dir = os.path.dirname(modelPath)
    file = os.path.basename(modelPath).split('.keras')[0]
    filename = os.path.join(dir, f"{file}.history")
    print(f"Saving the history to {filename}")

    with open(filename, 'wb') as handle:
        pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Loads data with Non-Zero AlignmentScore from specified benchmark
def getTestNonZeroData(DB_FILE, sqlite_batch = 1000):
    condition = "WHERE AlignmentScore != 0"

    dataset = LoadData.LoadDataset(DB_FILE, sqlite_batch, condition=condition)
    return dataset

# Evaluates using all data and non-zero data
def EvaluateModel(model, dataPath, metrics, batch_size = 32):
    print(f"===== Evaluating Model's Performance =====")

    # Get path to test dataset
    paths = LoadData.getTempDirectories(dataPath)

    # Evaluate Model on All Data
    print("===== ALL TEST DATA =====")
    all_data = LoadData.LoadDataset(paths[2])
    all_eval = model.evaluate(all_data)
    print(f"Test Loss: {all_eval[0]}")
    for idx, metric in enumerate(metrics):
        print(f"Test {metric}: {all_eval[idx + 1]}")

    # Evaluate Model on Non-Zero Data
    print("===== NON-ZERO TEST DATA =====")
    non_zero_data = getTestNonZeroData(paths[2])
    non_zero_eval = model.evaluate(non_zero_data)
    print(f"Test Loss: {non_zero_eval[0]}")
    for idx, metric in enumerate(metrics):
        print(f"Test {metric}: {non_zero_eval[idx + 1]}")

    return all_eval, non_zero_eval

# Splits a string by comma and stripping them
# Used for splitting metrics parameter
def SplitString(input):
    return [x.strip() for x in input.split(',')]

