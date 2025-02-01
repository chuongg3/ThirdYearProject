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

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <benchmark_base_directory> <script_location> <technique>"
    exit 1
fi

# Validate technique argument
echo "Technique Argument: $3"
technique=$3
if [[ "$technique" != "f3m" && "$technique" != "f3m-adapt" && "$technique" != "hyfm" && "$technique" != "baseline" ]]; then
    echo "Invalid technique: $technique"
    echo "Valid options are: f3m, f3m-adapt, hyfm, baseline"
    exit 1
fi

# find "$search_dir" -type f -name "_main_._all_._files_._linked_.bc" -exec sh -c '. "$0" "$(dirname "$1")" "$2" | tee ./log/script_log.txt' "$script_loc" {} "$technique" \;
find "$search_dir" -type f -name "_main_._all_._files_._linked_.bc" -exec sh -c '. "$0" "$1" "$2" | tee ./log/script_log.txt' "$script_loc" {} "$technique" \;