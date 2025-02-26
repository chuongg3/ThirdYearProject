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
      args: tuple containing (batch_index, df, skip)
            - batch_index: the index of the current batch
            - df: a DataFrame read by pd.read_sql_query
            - skip: boolean flag; if True, skip processing this entire chunk.

    Returns:
      A tuple (batch_index, Encoding1_arr, Encoding2_arr, AlignmentScore_arr, num_rows, status)
      - status is 'skipped' if skipped, 'processed' if processed.
    """
    while checkPercentageAvaiMemory() < 0.2:
        print("Memory is low. Sleeping ...")
        time.sleep(120)

    batch_index, df, skip = args
    total_rows = df.shape[0]
    if skip:
        print(f"Skipping entire batch {batch_index} ({total_rows} rows).")
        return (batch_index, None, None, None, total_rows, 'skipped')

    print("Processing batch: ", batch_index)
    Encoding1_arr = np.vstack(df["Encoding1"].map(lambda x: np.array(pickle.loads(x), dtype=np.float32)))
    Encoding2_arr = np.vstack(df["Encoding2"].map(lambda x: np.array(pickle.loads(x), dtype=np.float32)))
    AlignmentScore_arr = df["AlignmentScore"].astype(np.float32).values
    num_rows = df.shape[0]
    return (batch_index, Encoding1_arr, Encoding2_arr, AlignmentScore_arr, num_rows, 'processed')

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

# Task generator now yields a tuple (batch_index, df, skip_flag)
def task_generator(chunk_gen, rows_to_skip):
    """
    For each chunk from chunk_gen, yield (batch_index, df, skip)
    where skip is True if the entire chunk should be skipped.
    If the remaining rows to skip is greater than or equal to the chunk size,
    then mark skip as True; otherwise, mark skip as False.
    """
    for batch_index, df in enumerate(chunk_gen):
        if rows_to_skip >= df.shape[0]:
            skip = True
            rows_to_skip -= df.shape[0]
        else:
            skip = False
            # Once we reach a chunk that is not entirely skipped,
            # we set rows_to_skip to 0 for subsequent chunks.
            rows_to_skip = 0
        yield (batch_index, df, skip)

# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Serialize Data")
    parser.add_argument("--db", '-d',  type=str, help="The database file to load data from", required=True)
    parser.add_argument("--batch_size", '-b', type=int, help="The batch size", default=10000)
    parser.add_argument("--workers", '-w', type=int, help="The number of workers to use", default=4)
    parser.add_argument("--chunk_size", '-c', type=int, help="The number of rows to save in each chunk", default=22000000)
    parser.add_argument("--start_chunk", '-s', type=int, help="Start processing from this chunk index", default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    DB_FILE = args.db
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.workers
    CHUNK_SIZE = args.chunk_size
    START_CHUNK = args.start_chunk

    # Calculate how many rows to skip
    rows_to_skip = START_CHUNK * CHUNK_SIZE

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

    # Create a lazy iterator for all the tasks
    tasks = task_generator(chunk_gen, rows_to_skip)

    enc1_list = []
    enc2_list = []
    score_list = []
    rows_accumulated = 0
    save_chunk_index = START_CHUNK

    # Use imap to process chunks as they are read.
    # IMAP preserves order of results.
    for result in pool.imap(process_chunk, tasks):
        # result: (batch_index, Encoding1_arr, Encoding2_arr, AlignmentScore_arr, num_rows, status)
        status = result[-1]
        rows_accumulated += result[4]
        if status != 'skipped':
            enc1_list.append(result[1])
            enc2_list.append(result[2])
            score_list.append(result[3])

        # When the accumulated row count exceeds CHUNK_SIZE...
        if rows_accumulated >= CHUNK_SIZE:
            # CHANGED: Only save the chunk if we have any processed data
            if enc1_list:  # Meaning there is some processed (non-skipped) data.
                save_chunk(filename, enc1_list, enc2_list, score_list, save_chunk_index)
                save_chunk_index += 1
            else:
                print(f"Chunk {save_chunk_index} contains only skipped rows; not saving an empty file.")
            enc1_list = []
            enc2_list = []
            score_list = []
            rows_accumulated = 0

    # Save any remaining results
    if rows_accumulated > 0 and enc1_list:
        save_chunk(filename, enc1_list, enc2_list, score_list, save_chunk_index)

    pool.close()
    pool.join()
    conn.close()

    print("Processing complete!")

if __name__ == "__main__":
    main()

