#!/bin/bash

# This script is used to run the encoding script on a benchmark bitcode file

set -x

BENCHMARK_LOC=$1
echo "Benchmark Directory: $BENCHMARK_LOC"

if [[ "$BENCHMARK_LOC" =~ build/_main_._all_._files_._linked_.bc$ ]]; then
    echo "Benchmark directory ends with build/_main_._all_._files_._linked_.bc. Exiting script."
    exit 1
fi

echo "Benchmark Directory: $BENCHMARK_LOC"


# Extract benchmark name
BENCHMARK_DIR=$(dirname "$BENCHMARK_LOC")
BENCHMARK=$(basename "$BENCHMARK_DIR")

# Encode function using ir2vec
ir2vec_bin=$2
EMBLocation="./Embedding/$BENCHMARK.emb"
$ir2vec_bin "$BENCHMARK_LOC" -fa -level=f -o "$EMBLocation"

# Database Location
database_loc=$3

# Run the cpp file with the right environment variables
export BENCH="$BENCHMARK"
export DBLOC="$database_loc"
export EMBLOC="$EMBLocation"
python3 ./GetEncoding.py

