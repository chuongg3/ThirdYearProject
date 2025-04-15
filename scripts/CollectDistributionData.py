import re
import os
import pandas
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Collect distribution data")
    parser.add_argument('--log_dir', '-ld', type=str, default="../f3m_exp/logs/", help='Path to the log files')
    parser.add_argument('--output', '-o', type=str, default="./MatchInfo.csv", help='Output directory for logs and models')
    return parser.parse_args()

re_filename = re.compile(r'build\.(.*)\.(TECHNIQUE=.*)\..*\..*\.log')
re_num_func = re.compile(r'Number of Functions: (\d+)')
re_merge = re.compile(r'@.* \+ @.* <= .* Valid: (\d) BinSizes: (\d+) \+ (\d+) <= (\d+) .* Profitable: (\d)')

if __name__ == "__main__":
    metrics = []
    args = parse_args()

    # Script Arguments
    LOG_DIR = args.log_dir
    OUTPUT = args.output

    # Get list of files in LOG_DIR
    files = os.listdir(LOG_DIR)
    files.sort()
    print(f"Found {len(files)} files in {LOG_DIR}")

    for file in files:
        # Extract each file's name
        print(f'File: {file}')
        filename_match = re.search(re_filename, file)
        if not filename_match:
            continue
        print(f'Benchmark: {filename_match.group(1)}')
        print(f'Flags: {filename_match.group(2)}')

        # Extract log information
        total_functions = 0
        merge_attempts = 0
        profitable_merges = 0
        bin_size_diff = 0

        # Read the file
        filepath = os.path.join(LOG_DIR, file)
        with open(filepath, 'r') as log_file:
            for line in log_file:
                # Find the number of functions
                num_func_match = re.search(re_num_func, line)
                if num_func_match:
                    total_functions = int(num_func_match.group(1))
                    print(f'Total Functions: {total_functions}')

                # Find the merge attempts
                merge_match = re.search(re_merge, line)
                if merge_match:
                    # Update the number of merge attempted
                    merge_attempts += 1

                    # Check if the merge was profitable
                    if int(merge_match.group(5)) == 1:
                        profitable_merges += 1

                    # Get the binary sizes
                    bin_size_1 = int(merge_match.group(2))
                    bin_size_2 = int(merge_match.group(3))
                    merged_bin_size = int(merge_match.group(4))

                    # Calculate the difference in binary sizes
                    bin_size_diff += merged_bin_size - (bin_size_1 + bin_size_2)
        
        print(f'Merge Attempts: {merge_attempts}')
        print(f'Profitable Merges: {profitable_merges}')
        print(f'Bin Size Difference: {bin_size_diff}')

        metrics.append([filename_match.group(1), filename_match.group(2), total_functions, merge_attempts, profitable_merges, bin_size_diff])
        print('===================================')

    # Save the data to a CSV file
    columns = ['Benchmark', 'Flags', 'TotalFunctions', 'MergeAttempts', 'ProfitableMerges', 'BinSizeDifference']
    data = pandas.DataFrame(metrics, columns=columns)
    data.to_csv(OUTPUT, index=False)
    print(f'Data saved to {OUTPUT}')