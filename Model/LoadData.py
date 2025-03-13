import os
import re
import time
import glob
import torch
import pickle
import sqlite3
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

_NEW_DATASET = True

""" ===== GENERAL FUNCTIONS ===== """

def connectToDB(DBLoc):
    if not os.path.exists(DBLoc):
        print(f"ERROR: {DBLoc} does not exist")
        return None
    try:
        print(f"Connecting to {DBLoc} .....")
        conn = sqlite3.connect(DBLoc, check_same_thread=False)
        print(f"Connected to {DBLoc}")
        return conn
    except sqlite3.Error as e:
        print(f"ERROR: Failed to open {DBLoc}")
        print(e)
        return None

def closeDB(conn):
    print("Closing connection .....")
    conn.close()

# Deserialize the Encoding column: converts binary data back into a Python object using pickle
def deserialize_encoding(binary_data):
    return pickle.loads(binary_data) if binary_data else None

def LoadAllEncodings(conn):
    query = f"""SELECT F1.Encoding AS Encoding1, F2.Encoding AS Encoding2, FunctionPairs.AlignmentScore
FROM FunctionPairs
JOIN Functions F1 ON
FunctionPairs.BenchmarkID = F1.BenchmarkID AND
FunctionPairs.Function1ID = F1.FunctionID
JOIN Functions F2 ON
FunctionPairs.BenchmarkID = F2.BenchmarkID AND
FunctionPairs.Function2ID = F2.FunctionID;"""

    df = pd.read_sql_query(query, conn)
    return df

def LoadAlignmentScore(conn):
    query = f"SELECT AlignmentScore FROM FunctionPairs"

    AlignmentScore = pd.read_sql_query(query, conn)
    return AlignmentScore

# Gets the size of the FunctionPairs table
def getDatasetSize(DB_File, condition = ""):
    query = f"""SELECT COUNT(ROWID) FROM FunctionPairs {condition}"""
    with sqlite3.connect(DB_File, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        return row[0]

# Returns a file name without the extension
def getFileName(DB_FILE):
    return ".".join(DB_FILE.split(".")[:-1])

# Get the temporary directories of the data to be generated
def getTempDirectories(DB_FILE):
    directory = os.path.dirname(DB_FILE)
    temp_directory = os.path.join(directory, ".temp")
    train_path = os.path.join(temp_directory, "train.db")
    val_path = os.path.join(temp_directory, "validation.db")
    test_path = os.path.join(temp_directory, "test.db")

    return train_path, val_path, test_path

# Convert a serialised python list into an NDArray
def decodeNP(element):
    element = pickle.loads(element)
    return np.array(element, dtype=np.float32)

# Convert Row to Three ND Arrays for each column
def convertRowToNDArray(row):
    columns = list(zip(*row))
    x1 = list(map(decodeNP, columns[0]))
    x2 = list(map(decodeNP, columns[1]))
    y = np.array(columns[2], dtype=np.float32)

    # Turn the list of arrays into NP Arrays
    x1 = np.vstack(x1)
    x2 = np.vstack(x2)

    return x1, x2, y

def calculateSampleWeights(number_weights):
    total = sum(number_weights)
    num_classes = len(number_weights)

    sample_weights = [total/(num_classes * num) for num in number_weights]
    return sample_weights

def calculateSampleWeight(AlignmentScores):
    # # Define conditions
    # conditions = [
    #     (AlignmentScores == 0),
    #     (0 < AlignmentScores) & (AlignmentScores <= 0.1),
    #     (0.1 < AlignmentScores) & (AlignmentScores <= 0.2),
    #     (0.2 < AlignmentScores) & (AlignmentScores <= 0.5),
    #     (0.2 < AlignmentScores) & (AlignmentScores <= 0.8),
    #     (0.8 < AlignmentScores) & (AlignmentScores < 1),
    #     (AlignmentScores == 1)
    # ]

    # # Define corresponding values
    # values = [0.01, 0.32, 5.71, 5.93, 28.01, 52.94, 92.13]

    # # Apply np.select
    # result = np.select(conditions, values, default=-1)  # Default for unmatched cases

    result = np.where(AlignmentScores == 0, 0.001, 1)
    return result


""" ===== Tensorflow Related Datasets ===== """

# Tensorflow dataset which loads only the necessary data
def TensorflowTrainingDataset(DB_FILE, batch_size = 1000, dataset = "Training", condition = "", zero_weight = 0.001, non_zero_weight = 1):
    query = f"""SELECT F1.Encoding AS Encoding1, F2.Encoding AS Encoding2, FunctionPairs.AlignmentScore
FROM FunctionPairs
JOIN Functions F1 ON
FunctionPairs.BenchmarkID = F1.BenchmarkID AND
FunctionPairs.Function1ID = F1.FunctionID
JOIN Functions F2 ON
FunctionPairs.BenchmarkID = F2.BenchmarkID AND
FunctionPairs.Function2ID = F2.FunctionID {condition}"""
    # Prints time taken to load dataset
    if not __debug__:
        totalTime = 0
        starttime = time.time()

    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(query)

        count = 0
        debugLoop = 500000

        print(f"Start of {dataset} dataset")
        while True:
            if not __debug__:
                starttime = time.time()

            # Fetch the rows
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            # Convert the rows into numpy arrays
            x1, x2, y = convertRowToNDArray(rows)
            if not __debug__:
                totalTime += (time.time() - starttime)
            sample_weight = calculateSampleWeight(y)
            yield (x1, x2), y, sample_weight
            count += len(rows)

        if not __debug__:
            # totalTime = time.time() - starttime
            print(f"Total time taken to load {count} so far: {time.strftime('%H:%M:%S', time.gmtime(totalTime))}")
        print(f"\nSize of dataset ({dataset}): {count}")

# Loads a dataset given the condition
def LoadSQLDataset(DB_FILE, sqlite_batch = 64, condition = "", zero_weight=0.001, non_zero_weight=1):
    dataset = tf.data.Dataset.from_generator(
        lambda: TensorflowTrainingDataset(DB_FILE, sqlite_batch, condition, condition, zero_weight=zero_weight, non_zero_weight=non_zero_weight),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create a tensorflow databset
def CreateTensorflowDataset(DB_FILE, split_size = (0.7, 0.1, 0.2), batch_size = 1000, overwrite = False, zero_weight = 0.001, non_zero_weight = 1):
    print(f"===== Loading {DB_FILE} into Tensorflow Dataset =====")
    from SplitDB import SplitDB

    # Get the temporary file directory
    train_path, val_path, test_path = getTempDirectories(DB_FILE)

    # Check if data exists
    data_exists =  (os.path.exists(train_path) and 
                    os.path.exists(val_path) and
                    os.path.exists(test_path))
    print(f"Data exists: {data_exists}")
    
    # Delete the original files if exists is true and overwrite is true
    if (data_exists and overwrite):
        os.remove(train_path)
        os.remove(val_path)
        os.remove(test_path)
    
    # Load the data given the directories
    if not data_exists or overwrite:
        SplitDB(DB_FILE, split_size)

    # Load the data into datasets
    train_set = tf.data.Dataset.from_generator(
        lambda: TensorflowTrainingDataset(train_path, batch_size, "Training", zero_weight=zero_weight, non_zero_weight=non_zero_weight),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    val_set = tf.data.Dataset.from_generator(
        lambda: TensorflowTrainingDataset(val_path, batch_size, "Validation", zero_weight=zero_weight, non_zero_weight=non_zero_weight),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    test_set = tf.data.Dataset.from_generator(
        lambda: TensorflowTrainingDataset(test_path, batch_size, "Testing", zero_weight=zero_weight, non_zero_weight=non_zero_weight),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    train_set = train_set.prefetch(tf.data.AUTOTUNE)
    val_set = val_set.prefetch(tf.data.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.AUTOTUNE)

    print(f"===== Finished Tensorflow Dataset =====")
    return train_set, val_set, test_set

def natural_keys(text):
    """Sort helper that converts numeric parts to integers."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

# Loads NPZ files with memory mapping and returns the arrays.
def load_npz_arrays(file_path, encodings=None, condition=None):
    print(f"[load_npz_arrays] Loading file {file_path} in process id: {os.getpid()}")
    with np.load(file_path, mmap_mode="r") as data:
        if _NEW_DATASET:
            assert encodings is not None
            print("Using new dataset")
            func1IDs = data["func1IDs"]
            enc1 = encodings[func1IDs]
            func2IDs = data["func2IDs"]
            enc2 = encodings[func2IDs]
        else:
            enc1 = data["Encoding1"]
            enc2 = data["Encoding2"]

        labels = data["AlignmentScore"]
    if condition is None:
        return enc1, enc2, labels
    else:
        indices = np.where(condition(labels))
        enc1 = enc1[indices]
        enc2 = enc2[indices]
        labels = labels[indices]
        return enc1, enc2, labels

# Dataset which loads data from numpy files
def NumpyDataset(DB_FILE, batch_size = 64, dataset = "Training", zero_weight = 0.001, non_zero_weight = 1, condition=None):
    print(f"[main] Processing batch in process id: {os.getpid()}")

    # Given the DB_FILE, find the filename to search for the numpy files
    filename = getFileName(DB_FILE)

    # Get list of all files with that extension
    numpyPaths = sorted(glob.glob(f"{filename}_*.npz"),
                        key=lambda x: natural_keys(os.path.basename(x)))
    print(f"{numpyPaths} files found ...")

    # Prints time taken to load dataset
    if not __debug__:
        totalTime = 0
        starttime = time.time()

    # Initialize accumulators for data from multiple files.
    leftover_enc1 = None
    leftover_enc2 = None
    leftover_labels = None

    count = 0

    # Use ThreadPoolExecutor to prefetch the next file.
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Create an iterator over the file paths.
        file_iter = iter(numpyPaths)

        encodings = None
        if _NEW_DATASET:
            print("Using new dataset")
            encodings_file = f'{filename}_encodings.npy'
            print(f"[NumpyDataset] Loading file {encodings_file} in process id: {os.getpid()}")
            encodings = np.load(encodings_file, mmap_mode="r")

        # Prefetch the first file.
        future = executor.submit(load_npz_arrays, next(file_iter), encodings, condition)

        # Loop through all files
        for file_path in file_iter:
            # Wait for the current file to be ready.
            enc1, enc2, labels = future.result()
            # Immediately schedule the next file.
            future = executor.submit(load_npz_arrays, file_path, encodings, condition)
            count += enc1.shape[0]

            # If there is leftover data from previous file, concatenate it.
            if leftover_enc1 is not None:
                enc1 = np.concatenate([leftover_enc1, enc1], axis=0)
                enc2 = np.concatenate([leftover_enc2, enc2], axis=0)
                labels = np.concatenate([leftover_labels, labels], axis=0)

            total_samples = enc1.shape[0]
            start = 0
            # Yield full batches using start/end indices.
            while start + batch_size <= total_samples:
                end = start + batch_size
                batch_enc1 = enc1[start:end]
                batch_enc2 = enc2[start:end]
                batch_labels = labels[start:end]
                sample_weight = calculateSampleWeight(batch_labels)
                yield (batch_enc1, batch_enc2), batch_labels, sample_weight
                start = end

            # Save leftover data from the current file.
            if start < total_samples:
                leftover_enc1 = enc1[start:]
                leftover_enc2 = enc2[start:]
                leftover_labels = labels[start:]
            else:
                leftover_enc1, leftover_enc2, leftover_labels = None, None, None

        # Process the final file that was prefetched.
        enc1, enc2, labels = future.result()
        count += enc1.shape[0]
        if leftover_enc1 is not None:
            enc1 = np.concatenate([leftover_enc1, enc1], axis=0)
            enc2 = np.concatenate([leftover_enc2, enc2], axis=0)
            labels = np.concatenate([leftover_labels, labels], axis=0)
        total_samples = enc1.shape[0]
        start = 0
        while start + batch_size <= total_samples:
            end = start + batch_size
            batch_enc1 = enc1[start:end]
            batch_enc2 = enc2[start:end]
            batch_labels = labels[start:end]
            sample_weight = calculateSampleWeight(batch_labels)
            yield (batch_enc1, batch_enc2), batch_labels, sample_weight
            start = end

        # After processing all files, yield any remaining data (if any).
        if leftover_enc1 is not None and leftover_enc1.shape[0] > 0:
            sample_weight = calculateSampleWeight(leftover_labels)
            yield (leftover_enc1, leftover_enc2), leftover_labels, sample_weight

    if not __debug__:
        totalTime = time.time() - starttime
        print(f"Total time taken to load {count} so far: {time.strftime('%H:%M:%S', time.gmtime(totalTime))}")
    print(f"\nSize of dataset ({dataset}): {count}")

# Loads a dataset given the condition
def LoadNumpyDataset(DB_FILE, sqlite_batch = 64, dataset_name = "Training", zero_weight = 0.001, non_zero_weight = 1, condition=None):
    dataset = tf.data.Dataset.from_generator(
        lambda: NumpyDataset(DB_FILE, sqlite_batch, dataset=dataset_name, zero_weight=zero_weight, non_zero_weight=non_zero_weight, condition=condition),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create a Numpy dataset for Tensorflow
def CreateNumpyDataset(DB_FILE, split_size = (0.7, 0.1, 0.2), batch_size = 64, overwrite = False, zero_weight = 0.001, non_zero_weight = 1):
    print(f"===== Loading {DB_FILE} into Numpy Dataset =====")
    from SplitDB import SplitDB

    # Get the temporary file directory
    train_path, val_path, test_path = getTempDirectories(DB_FILE)

    # Check if data exists
    data_exists =  (os.path.exists(train_path) and
                    os.path.exists(val_path) and
                    os.path.exists(test_path))
    print(f"Data exists: {data_exists}")

    # Delete the original files if exists is true and overwrite is true
    if (data_exists and overwrite):
        os.remove(train_path)
        os.remove(val_path)
        os.remove(test_path)

    # Load the data given the directories
    if not data_exists or overwrite:
        SplitDB(DB_FILE, split_size)

    # Load the data into datasets
    train_set = tf.data.Dataset.from_generator(
        lambda: NumpyDataset(train_path, batch_size, "Training", zero_weight=zero_weight, non_zero_weight=non_zero_weight),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    val_set = tf.data.Dataset.from_generator(
        lambda: NumpyDataset(val_path, batch_size, "Validation", zero_weight=zero_weight, non_zero_weight=non_zero_weight),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    test_set = tf.data.Dataset.from_generator(
        lambda: NumpyDataset(test_path, batch_size, "Testing", zero_weight=zero_weight, non_zero_weight=non_zero_weight),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    train_set = train_set.prefetch(tf.data.AUTOTUNE)
    val_set = val_set.prefetch(tf.data.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.AUTOTUNE)

    print(f"===== Finished Numpy Dataset =====")
    return train_set, val_set, test_set


""" ===== PyTorch Related Datasets ===== """


"""
Custom PyTorch dataset for encoding dataset
Columns in the dataset
   - Encoding 1: Python list of float
   - Encoding 2: Python list of float
   - Alignment Score: A float between 0 and 1 inclusive
"""
def EncodingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        Encoding1 = torch.tensor(row.iloc[0], dtype=torch.float32)
        Encoding2 = torch.tensor(row.iloc[1], dtype=torch.float32)
        AlignmentScore = row.iloc[2]
        return ((Encoding1, Encoding2), AlignmentScore)