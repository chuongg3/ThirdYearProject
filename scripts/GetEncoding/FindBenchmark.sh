#!/bin/bash

# This script is used to find all benchmark bitcode files in the benchmark's base directory
# and calls a specified script on each bitcode file and saves a log file in the same location

# Technique Options:
# - f3m
# - f3m-adapt
# - hyfm
# - baseline

# set -x

search_dir=$1
script_loc=$2

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <benchmark_base_directory> <script_location> <ir2vec_binary> <database_location>"
    exit 1
fi

ir2vec=$3
dblocation=$4


find "$search_dir" -type f -name "_main_._all_._files_._linked_.bc" -exec sh -c '"$0" "$1" "$2" "$3"' "$script_loc" {} "$ir2vec" "$dblocation" \;