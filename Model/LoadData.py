import sqlite3
import pickle
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
import math

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