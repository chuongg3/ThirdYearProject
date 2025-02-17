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

def FetchEncodingDataset(DB_FILE, condition, score = 0):
    query = f"""SELECT F1.Encoding AS Encoding1, F2.Encoding AS Encoding2, FunctionPairs.AlignmentScore
FROM FunctionPairs
JOIN Functions F1 ON
FunctionPairs.BenchmarkID = F1.BenchmarkID AND
FunctionPairs.Function1ID = F1.FunctionID
JOIN Functions F2 ON
FunctionPairs.BenchmarkID = F2.BenchmarkID AND
FunctionPairs.Function2ID = F2.FunctionID
WHERE {condition}"""

    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(query)

        print(f"Start of dataset (Score: {score})")
        count = 0
        batch_size = 1000  # Adjust the batch size as needed
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                encoding1 = np.array(pickle.loads(row[0]), dtype=np.float32)
                encoding2 = np.array(pickle.loads(row[1]), dtype=np.float32)
                AlignmentScore = float(row[2])
                weight = (1/1000) if AlignmentScore == 0 else (999/1000)

                count += 1
                if (count % 10000 == 0):
                    print(f"Dataset Index (Score: {score}): {count}")
                yield (encoding1, encoding2), AlignmentScore, weight
        print(f"Size of dataset (Score: {score}): {count}")

def CreateEncodingDataset(DB_FILE, score = 0, batch_size = 32, split_size = (0.7, 0.1, 0.2)):
    # Get the condition to filter out the SQL query
    if (score == 0):
        condition = "FunctionPairs.AlignmentScore = 0"
    elif (score == 1):
        condition = f"FunctionPairs.AlignmentScore = 1"
    else:
        condition = f"FunctionPairs.AlignmentScore > 0 AND FunctionPairs.AlignmentScore < 1"

    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: FetchEncodingDataset(DB_FILE, condition, score),
        output_signature=(
            (tf.TensorSpec(shape=(300,), dtype=tf.float32),
             tf.TensorSpec(shape=(300,), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    # Get the number of rows
    num_rows = getDatasetSize(DB_FILE, condition)
    train_size = int(split_size[0] * num_rows)
    val_size = int(split_size[1] * num_rows)
    test_size = num_rows - train_size - val_size

    # Shuffle the dataset
    # dataset = dataset.shuffle(10000)

    # Split the dataset into training, validation and testing
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    # Batch the datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

    