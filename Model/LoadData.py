import sqlite3
import pickle
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import tensorflow as tf

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

# Custom dataset for encoding dataset
# Columns in the dataset
#    - Encoding 1: Python list of float
#    - Encoding 2: Python list of float
#    - Alignment Score: A float between 0 and 1 includive
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

def LoadAlignmentScore(conn):
    query = f"SELECT AlignmentScore FROM FunctionPairs"

    AlignmentScore = pd.read_sql_query(query, conn)
    return AlignmentScore

def getDatasetSize(DB_File, condition):
    query = f"""SELECT COUNT(ROWID) FROM FunctionPairs WHERE {condition}"""
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

# Tensorflow dataset which loads only the necessary data
def TensorflowTrainingDataset(DB_FILE, batch_size = 1000, dataset = "Training"):
    query = f"""SELECT F1.Encoding AS Encoding1, F2.Encoding AS Encoding2, FunctionPairs.AlignmentScore
FROM FunctionPairs
JOIN Functions F1 ON
FunctionPairs.BenchmarkID = F1.BenchmarkID AND
FunctionPairs.Function1ID = F1.FunctionID
JOIN Functions F2 ON
FunctionPairs.BenchmarkID = F2.BenchmarkID AND
FunctionPairs.Function2ID = F2.FunctionID"""

    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        print(f"Start of {dataset} dataset")
        count = 0
        debugLoop = 10000
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                # Pickle the data where needed
                encoding1 = np.array(pickle.loads(row[0]), dtype=np.float32)
                encoding2 = np.array(pickle.loads(row[1]), dtype=np.float32)
                AlignmentScore = float(row[2])
                weight = (1/1000) if AlignmentScore == 0 else (999/1000)

                # Keeps track of the number of rows
                count += 1
                if (count % debugLoop == 0):
                    print(f"Dataset Index ({dataset}): {count}")

                # Outputs the results
                yield (encoding1, encoding2), AlignmentScore, weight
        print(f"Size of dataset ({dataset}): {count}")

# Create a tensorflow databset
def CreateTensorflowDataset(DB_FILE, batch_size = 32, split_size = (0.7, 0.1, 0.2), sqlite_batch = 1000, overwrite = False):
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
        lambda: TensorflowTrainingDataset(train_path, sqlite_batch, "Training"),
        output_signature=(
            (tf.TensorSpec(shape=(300,), dtype=tf.float32),
             tf.TensorSpec(shape=(300,), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    val_set = tf.data.Dataset.from_generator(
        lambda: TensorflowTrainingDataset(val_path, sqlite_batch, "Validation"),
        output_signature=(
            (tf.TensorSpec(shape=(300,), dtype=tf.float32),
             tf.TensorSpec(shape=(300,), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    test_set = tf.data.Dataset.from_generator(
        lambda: TensorflowTrainingDataset(test_path, sqlite_batch, "Testing"),
        output_signature=(
            (tf.TensorSpec(shape=(300,), dtype=tf.float32),
             tf.TensorSpec(shape=(300,), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    train_set = train_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_set = val_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_set = test_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_set, val_set, test_set
