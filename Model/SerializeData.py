#!/usr/bin/env python3
import argparse
import pickle
import sqlite3

import numpy as np

import LoadData

_QUERY_BENCH = 'SELECT ROWID, BenchmarkName FROM Benchmarks'
_QUERY_FUNC = 'SELECT BenchmarkID, FunctionID, FunctionName, Encoding FROM Functions'
_QUERY_PAIRS = 'SELECT BenchmarkID, Function1ID, Function2ID, AlignmentScore FROM FunctionPairs'

def serialize_chunk(basename: str, chunk_idx: int, func1IDs: np.array, func2IDs:np.array, scores: np.array):
    np.savez(f"{basename}_{chunk_idx}.npz",
            func1IDs=func1IDs, func2IDs=func2IDs,
            AlignmentScore=scores)
    print(f"Saved chunk {chunk_idx} with {scores.shape[0]} rows.")

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
    CHUNK_SIZE = args.chunk_size
    START_CHUNK = args.start_chunk

    # Calculate how many rows to skip
    rows_to_skip = START_CHUNK * CHUNK_SIZE

    # Get the file name
    basename = DB_FILE.rpartition('.')[0]
    print(f"File name without extension: {basename}")

    # Open connection
    conn = LoadData.connectToDB(DB_FILE)

    # Load all benchmark and function information from the db
    print("Reading Benchmark SQL data...")

    benchmarks = {}
    for benchmarkID, name in conn.execute(_QUERY_BENCH):
        benchmarks[benchmarkID] = name

    print("Reading Function SQL data...")

    functions = {}
    function_uniqueID = {}
    function_list = []
    function_encoding = []

    for benchmarkID, functionID, name, encoding_txt in conn.execute(_QUERY_FUNC):
        key = (benchmarkID, functionID)
        assert key not in function_uniqueID

        functions[functionID] = name

        function_uniqueID[key] = len(function_list)
        function_list.append(key)
        function_encoding.append(pickle.loads(encoding_txt))

    encodings_arr = np.array(function_encoding, dtype=np.float32)

    # Store all non-chunk information
    np.save(f'{basename}_encodings.npy', encodings_arr)
    with open(f'{basename}_function_list.txt', 'w', encoding='utf8') as fout:
        for benchmarkID, functionID in function_list:
            fout.write(f'{benchmarkID} : {benchmarks[benchmarkID]}\t{functionID} : {functions[functionID]}\n')

    # Load all function pair data in chunks
    print("Reading Pair SQL data...")

    func1_arr = np.empty(CHUNK_SIZE, dtype=np.int32)
    func2_arr = np.empty(CHUNK_SIZE, dtype=np.int32)
    scores_arr = np.empty(CHUNK_SIZE, dtype=np.float32)
    pair_idx = 0
    chunk_idx = START_CHUNK

    for benchmarkID, function1ID, function2ID, score in conn.execute(_QUERY_PAIRS):
        if rows_to_skip > 0:
            rows_to_skip -= 1
            continue

        key1 = (benchmarkID, function1ID)
        func1_arr[pair_idx] = function_uniqueID[key1]

        key2 = (benchmarkID, function2ID)
        func2_arr[pair_idx] = function_uniqueID[key2]

        scores_arr[pair_idx] = score
        pair_idx += 1

        if pair_idx == CHUNK_SIZE:
            serialize_chunk(basename, chunk_idx, func1_arr, func2_arr, scores_arr)

            # Initialise the next chunk
            func1_arr = np.empty(CHUNK_SIZE, dtype=np.int32)
            func2_arr = np.empty(CHUNK_SIZE, dtype=np.int32)
            scores_arr = np.empty(CHUNK_SIZE, dtype=np.float32)
            pair_idx = 0
            chunk_idx += 1

    if pair_idx > 0:
        # Serialise the last incomplete chunk
        func1_arr.resize(pair_idx)
        func2_arr.resize(pair_idx)
        scores_arr.resize(pair_idx)
        serialize_chunk(basename, chunk_idx, func1_arr, func2_arr, scores_arr)

    conn.close()

    print("Processing complete!")

if __name__ == "__main__":
    main()
