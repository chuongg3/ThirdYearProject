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


QUERY = f"""SELECT F1.Encoding AS Encoding1, F2.Encoding AS Encoding2, FunctionPairs.AlignmentScore
FROM FunctionPairs
JOIN Functions F1 ON
FunctionPairs.BenchmarkID = F1.BenchmarkID AND
FunctionPairs.Function1ID = F1.FunctionID
JOIN Functions F2 ON
FunctionPairs.BenchmarkID = F2.BenchmarkID AND
FunctionPairs.Function2ID = F2.FunctionID"""

# Function to read data in chunks (each worker runs this)
def worker(DB_PATH, worker_id, task_queue, result_queue, batch_size):
    """ Fetches rows from SQLite and processes them. """
    conn = LoadData.connectToDB(DB_PATH)
    total_memory = psutil.virtual_memory().total
    while True:
        available_memory = psutil.virtual_memory().available
        while available_memory < total_memory * 0.2:  # If memory is low, sleep
            print(f"Worker {worker_id}: Memory low ({available_memory / 1024 / 1024:.2f} MB). Sleeping...")
            time.sleep(120)
            available_memory = psutil.virtual_memory().available

        start_row = task_queue.get()
        if start_row is None:  # Stop signal
            break

        # Fetch data in batches
        if not __debug__:
            print(f"Worker {worker_id}: Fetching data from {start_row}...")
        finalquery = f"{QUERY} LIMIT {batch_size} OFFSET {start_row}"

        # Use Pandas to load chunked data
        df = pd.read_sql_query(finalquery, conn)

        # Process data
        Encoding1_arr = np.vstack(df["Encoding1"].map(lambda x: np.array(pickle.loads(x), dtype=np.float32)))  # Convert list strings to NumPy arrays
        Encoding2_arr = np.vstack(df["Encoding2"].map(lambda x: np.array(pickle.loads(x), dtype=np.float32)))
        AlignmentScore_arr = df["AlignmentScore"].astype(np.float32).values  # Already numeric

        # Send results back to main process
        result_queue.put((Encoding1_arr, Encoding2_arr, AlignmentScore_arr, Encoding1_arr.shape[0]))

    conn.close()

# Function to save a chunk
def save_chunk(filename, enc1_list, enc2_list, score_list, chunk_index):
    Encoding1_arr = np.vstack(enc1_list)
    Encoding2_arr = np.vstack(enc2_list)
    AlignmentScore_arr = np.hstack(score_list)

    np.savez_compressed(
        f"{filename}_{chunk_index}.npz",
        Encoding1=Encoding1_arr,
        Encoding2=Encoding2_arr,
        AlignmentScore=AlignmentScore_arr
    )
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

    """ ===== Main Process ===== """
    print("Starting main process...")
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Get total rows count
    print("Getting total rows count...")
    total_rows = LoadData.getDatasetSize(DB_FILE)
    print(f"Total rows: {total_rows}")

    # Start worker processes
    print(f"Starting {NUM_WORKERS} workers...")
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(DB_FILE, i, task_queue, result_queue, BATCH_SIZE))
        p.start()
        workers.append(p)

    # Add tasks to queue (each worker gets different OFFSET)
    print("Adding tasks to queue...")
    for offset in range(0, total_rows, BATCH_SIZE):
        task_queue.put(offset)

    # Stop workers
    print("Stopping workers...")
    for _ in range(NUM_WORKERS):
        task_queue.put(None)

    # Collect results and write chunks
    enc1_list, enc2_list, score_list = [], [], []
    chunk_index, row_count = 0, 0

    # Process results
    print("Processing results...")
    while any(p.is_alive() for p in workers) or not result_queue.empty():
        if not result_queue.empty():
            Encoding1_arr, Encoding2_arr, AlignmentScore_arr, batch_rows = result_queue.get()

            # Store processed data
            enc1_list.append(Encoding1_arr)
            enc2_list.append(Encoding2_arr)
            score_list.append(AlignmentScore_arr)
            row_count += batch_rows

            # Save chunk if it reaches CHUNK_SIZE
            if row_count >= CHUNK_SIZE:
                save_chunk(filename, enc1_list, enc2_list, score_list, chunk_index)
                enc1_list, enc2_list, score_list = [], [], []  # Reset
                row_count = 0
                chunk_index += 1

    # Save any remaining data
    if row_count > 0:
        save_chunk(filename, enc1_list, enc2_list, score_list, chunk_index)

    # Clean up workers
    for p in workers:
        p.join()

    print("Processing complete!")

if __name__ == "__main__":
    main()

