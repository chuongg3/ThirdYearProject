import time
import psutil
import pickle
import LoadData
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

# Returns a file name without the extension
# Assuming that all DB_FILE passed through are in the format of "name.db"
def getFileName(DB_FILE):
    return ".".join(DB_FILE.split(".")[:-1])

# Returns the percentage of available memory
def checkPercentageAvaiMemory(total_memory = psutil.virtual_memory().total):
    available_memory = psutil.virtual_memory().available
    return available_memory / total_memory

QUERY = f"""SELECT F1.Encoding AS Encoding1, F2.Encoding AS Encoding2, FunctionPairs.AlignmentScore
FROM FunctionPairs
JOIN Functions F1 ON
FunctionPairs.BenchmarkID = F1.BenchmarkID AND
FunctionPairs.Function1ID = F1.FunctionID
JOIN Functions F2 ON
FunctionPairs.BenchmarkID = F2.BenchmarkID AND
FunctionPairs.Function2ID = F2.FunctionID"""

def process_chunk(args):
    """
    Processes one chunk of data.

    Args:
      args: tuple containing (chunk_index, df) where df is a DataFrame read by pd.read_sql_query.

    Returns:
      A tuple (chunk_index, Encoding1_arr, Encoding2_arr, AlignmentScore_arr, num_rows)
    """
    while checkPercentageAvaiMemory() < 0.2:
        print("Memory is low. Sleeping ...")
        time.sleep(120)

    batch_index, df = args
    # Process each column:
    print("Processing batch", batch_index)
    Encoding1_arr = np.vstack(df["Encoding1"].map(lambda x: np.array(pickle.loads(x), dtype=np.float32)))
    Encoding2_arr = np.vstack(df["Encoding2"].map(lambda x: np.array(pickle.loads(x), dtype=np.float32)))
    AlignmentScore_arr = df["AlignmentScore"].astype(np.float32).values
    num_rows = df.shape[0]
    return (batch_index, Encoding1_arr, Encoding2_arr, AlignmentScore_arr, num_rows)

# Converts list of ND Arrays and to 2D ND Array and saves to a compressed .npz file.
def save_chunk(filename, enc1_list, enc2_list, score_list, chunk_index):
    """
    Merges lists of NumPy arrays and saves to a compressed .npz file.
    """
    Encoding1_arr = np.vstack(enc1_list)
    Encoding2_arr = np.vstack(enc2_list)
    AlignmentScore_arr = np.hstack(score_list)
    print(f"Saving chunk {chunk_index} with {Encoding1_arr.shape[0]} rows ...")
    np.savez_compressed(f"{filename}_{chunk_index}.npz",
                        Encoding1=Encoding1_arr,
                        Encoding2=Encoding2_arr,
                        AlignmentScore=AlignmentScore_arr)
    print(f"Saved chunk {chunk_index} with {Encoding1_arr.shape[0]} rows.")

# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Serialize Data")
    parser.add_argument("--db", '-d',  type=str, help="The database file to load data from", required=True)
    parser.add_argument("--batch_size", '-b', type=int, help="The batch size", default=10000)
    parser.add_argument("--workers", '-w', type=int, help="The number of workers to use", default=4)
    parser.add_argument("--chunk_size", '-c', type=int, help="The number of rows to save in each chunk", default=22000000)
    return parser.parse_args()

def main():
    args = parse_args()
    DB_FILE = args.db
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.workers
    CHUNK_SIZE = args.chunk_size

    # Get the file name
    filename = getFileName(DB_FILE)
    print(f"File name without extension: {filename}")

    # Open connection (main process reads sequentially)
    conn = LoadData.connectToDB(DB_FILE)

    # Prepare worker pool.
    """ ===== Main Process ===== """
    print("Starting main process...")
    pool = mp.Pool(NUM_WORKERS)

    # Use read_sql_query with chunksize to create an iterator.
    print("Reading SQL data...")
    chunk_gen = pd.read_sql_query(QUERY, conn, chunksize=BATCH_SIZE)

    enc1_list = []
    enc2_list = []
    score_list = []
    rows_accumulated = 0
    save_chunk_index = 0

    # Use imap to process chunks as they are read.
    # IMAP preserves order of results.
    print(f"Processing chunks with {NUM_WORKERS} workers...")
    for result in pool.imap(process_chunk, enumerate(chunk_gen)):
        # result: (chunk_index, Encoding1_arr, Encoding2_arr, AlignmentScore_arr, num_rows)
        enc1_list.append(result[1])
        enc2_list.append(result[2])
        score_list.append(result[3])
        rows_accumulated += result[-1]  # result[-1] is the number of rows

        # Save accumulated results to a chunk
        if rows_accumulated >= CHUNK_SIZE:
            save_chunk(filename, enc1_list, enc2_list, score_list, save_chunk_index)
            save_chunk_index += 1

            # Reset accumulators for next save chunk
            enc1_list = []
            enc2_list = []
            score_list = []
            rows_accumulated = 0

    # Save any remaining results
    if rows_accumulated > 0:
        save_chunk(filename, enc1_list, enc2_list, score_list, save_chunk_index)

    pool.close()
    pool.join()
    conn.close()

    print("Processing complete!")

if __name__ == "__main__":
    main()

