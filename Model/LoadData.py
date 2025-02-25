import sqlite3
import pickle
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import tensorflow as tf
import time

""" ===== GENERAL FUNCTIONS ===== """

def connectToDB(DBLoc):
    if not os.path.exists(DBLoc):
        print(f"ERROR: {DBLoc} does not exist")
        return None
    try:
        print(f"Connecting to {DBLoc} .....")
        conn = sqlite3.connect(DBLoc)
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

""" ===== Tensorflow Related Datasets ===== """

# Tensorflow dataset which loads only the necessary data
def TensorflowTrainingDataset(DB_FILE, batch_size = 1000, dataset = "Training", condition = ""):
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
            yield (x1, x2), y
            count += len(rows)

        if not __debug__:
            # totalTime = time.time() - starttime
            print(f"Total time taken to load {count} so far: {time.strftime('%H:%M:%S', time.gmtime(totalTime))}")
        print(f"\nSize of dataset ({dataset}): {count}")

# Loads a dataset given the condition
def LoadDataset(DB_FILE, sqlite_batch = 1000, condition = ""):
    dataset = tf.data.Dataset.from_generator(
        lambda: TensorflowTrainingDataset(DB_FILE, sqlite_batch, condition, condition),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create a tensorflow databset
def CreateTensorflowDataset(DB_FILE, split_size = (0.7, 0.1, 0.2), batch_size = 1000, overwrite = False):
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
        lambda: TensorflowTrainingDataset(train_path, batch_size, "Training"),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    val_set = tf.data.Dataset.from_generator(
        lambda: TensorflowTrainingDataset(val_path, batch_size, "Validation"),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    test_set = tf.data.Dataset.from_generator(
        lambda: TensorflowTrainingDataset(test_path, batch_size, "Testing"),
        output_signature=(
            (tf.TensorSpec(shape=(None, 300), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 300), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32)
        )
    )

    train_set = train_set.prefetch(tf.data.AUTOTUNE)
    val_set = val_set.prefetch(tf.data.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.AUTOTUNE)

    print(f"===== Finished Tensorflow Dataset =====")
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