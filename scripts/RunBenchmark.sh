#!/bin/bash

# Runs a benchmark given the directory and technique
# Usage: RunBenchmark.sh <benchmark_directory> <technique>

# First Parameter: Directory of benchmark
# Sets environment variable of benchmark name and runs the benchmark

set -x

LLVM_DIR="/home_alt/m19364tg/ThirdYearProject/build/bin"

# Check if the number of arguments is correct
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <benchmark_directory> <technique>"
    exit 1
fi

# Set the benchmark directory
BENCHMARK_LOC=$1
BENCHMARK_DIR=$(dirname "$BENCHMARK_LOC")
BENCHMARK=$(basename "$BENCHMARK_DIR")

echo "Benchmark Directory: $BENCHMARK_DIR"

# Set the current directory
CURRENT_DIR=$(pwd)
cd "$BENCHMARK_DIR" || { echo "Failed to change directory to $BENCHMARK_DIR"; exit 1; }

MERGE_TECHNIQUE=$2

# Run the benchmark
echo "Running benchmark: $BENCHMARK"
/usr/bin/make FROM_SCRATCH=false REPORT=true LLVM_DIR=$LLVM_DIR TECHNIQUE=$2 MERGE_TECHNIQUE="$MERGE_TECHNIQUE" BENCH="$BENCHMARK" 2>&1

# echo "Finished running benchmark: $BENCHMARK"

# Unset the benchmark directory
echo "Unsetting benchmark directory"
cd "$CURRENT_DIR" || { echo "Failed to change directory to $CURRENT_DIR"; exit 1; }
unset BENCHMARK
unset CURRENT_DIR
unset BENCHMARK_DIR
unset MERGE_TECHNIQUE

exit 0
