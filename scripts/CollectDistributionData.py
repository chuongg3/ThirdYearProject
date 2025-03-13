import re
import os
import sqlite3
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Collect distribution data")
    parser.add_argument('--log_dir', '-ld', type=str, default="../f3m_exp/logs/", help='Path to the data file')
    parser.add_argument('--output', '-o', type=str, default="./MatchInfo.db", help='Output directory for logs and models')
    return parser.parse_args()

digits_pattern = re.compile(r"\d+")

# Open Database
def openDatabase(DBFILE):
    # Only make changes to new database
    if not os.path.exists(DBFILE):
        print(f"Creating new database: {DBFILE}")
        conn = sqlite3.connect(DBFILE)
        return conn
    else:
        print(f"ERROR: {DBFILE} already exists")
        exit()

# Initialise the database
def initialiseDatabase(conn):
    cursor = conn.cursor()

    # Create Benchmarks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Benchmarks (
        BenchmarkName TEXT NOT NULL
    )
    ''')

    # Create Flags table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Flags (
        FLAG TEXT NOT NULL
    )
    ''')

    # Create MatchInfo table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS MatchInfo (
        BenchmarkID INTEGER NOT NULL,
        FlagID INTEGER NOT NULL,
        Valid INTEGER NOT NULL,
        BinSize1 INTEGER NOT NULL,
        BinSize2 INTEGER NOT NULL,
        MergedBinSize INTEGER NOT NULL,
        Profitable INTEGER NOT NULL,
        Distance REAL NOT NULL,
        TotalTime INTEGER NOT NULL,
        RankingTime INTEGER NOT NULL,
        AlignTime INTEGER NOT NULL,
        CodegenTime INTEGER NOT NULL,
        VerifyTime INTEGER NOT NULL,
        UpdateTime INTEGER NOT NULL,
        FOREIGN KEY (BenchmarkID) REFERENCES Benchmarks(ROWID),
        FOREIGN KEY (FlagID) REFERENCES Flags(ROWID)
    )
    ''')

# Inserts a benchmark name into Benchmarks table
# If the name already exists, it will return the ROWID of the existing row
# If the name doesn't exist, it will insert it and return the ROWID of the new row
def insertBenchmark(conn, BenchmarkName):
    print(f"Inserting Benchmark: {BenchmarkName}")
    cursor = conn.cursor()

    # Check if benchmark already exists
    cursor.execute("SELECT ROWID FROM Benchmarks WHERE BenchmarkName == ?", (BenchmarkName,))
    result = cursor.fetchone()

    # Benchmark exists
    if result:
        return result[0]
    # Benchmark doesn't exist
    else:
        cursor.execute("INSERT INTO Benchmarks (BenchmarkName) VALUES (?)", (BenchmarkName,))
        conn.commit()

        # Return the ROWID of the newly inserted row
        return cursor.lastrowid

# Inserts Flag into Flags table
# If the name already exists, it will return the ROWID of the existing row
# If the name doesn't exist, it will insert it and return the ROWID of the new row
def insertFlag(conn, Flag):
    print(f"Inserting Flag: {Flag}")
    cursor = conn.cursor()

    # Check if flag exists
    cursor.execute("SELECT ROWID FROM Flags WHERE FLAG == ?", (Flag,))
    result = cursor.fetchone()

    if result:
        return result[0]
    else:
        cursor.execute("INSERT INTO Flags (FLAG) VALUES (?)", (Flag,))
        conn.commit()
        return cursor.lastrowid

def insertMatchInfo(conn, BenchmarkID, FlagID, data):
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO MatchInfo (BenchmarkID, FlagID, Valid, BinSize1, BinSize2, MergedBinSize, Profitable, Distance, TotalTime, RankingTime, AlignTime, CodegenTime, VerifyTime, UpdateTime)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ''',
    (BenchmarkID, FlagID, data['valid'], data['bin_size_1'], data['bin_size_2'], data['bin_size_merged'], data['profitable'], data['distance'], data['total_time'], data['ranking_time'], data['align_time'], data['codegen_time'], data['verify_time'], data['update_time']), )
    conn.commit()

# Main Function
if __name__ == "__main__":
    args = parse_args()

    # Script Arguments
    LOG_DIR = args.log_dir
    OUTPUT = args.output

    # Connect to the database
    conn = openDatabase(OUTPUT)
    initialiseDatabase(conn)

    # Get list of files in LOG_DIR
    files = os.listdir(LOG_DIR)
    print(f"Found {len(files)} files in {LOG_DIR}")

    # Regular Expression Pattern for the log file
    d_pat = r"(\d*)"  # At least one digit
    sci_pat = r"([\d\.e\-\+]+)"  # Scientific notation pattern that handles both e+ and e-
    pattern = fr"Valid:\s+{d_pat}.*?BinSizes:\s+{d_pat}\s+\+\s+{d_pat}\s+<=\s+{d_pat}.*?Profitable:\s+{d_pat}.*?Distance:\s+{sci_pat}.*?TotalTime:\s+{d_pat}.*?RankingTime:\s+{d_pat}.*?AlignTime:\s+{d_pat}.*?CodegenTime:\s+{d_pat}.*?VerifyTime:\s+{d_pat}.*?UpdateTime:\s+{d_pat}"
    file_pattern = r'build\.(.*)\.(TECHNIQUE=.*)\.\d*\-\d*\-\d*.*'

    for file in files:
        # Only read log files
        if not file.endswith('.log'):
            continue
        print(f"Processing File: {file}")

        # Get the build conditions
        file_match = re.search(file_pattern, file)
        print(f"BENCHMARK: {file_match.group(1)}")
        print(f"FLAGS: {file_match.group(2)}")

        # Insert the benchmark and flag into the database
        benchmark_id = insertBenchmark(conn, file_match.group(1))
        flag_id = insertFlag(conn, file_match.group(2))

        # Read the file
        with open(os.path.join(LOG_DIR, file), 'r') as file:
            count = 0
            for line in file:
                match = re.search(pattern, line)
                if match:
                    data = {
                        'valid': int(match.group(1)),
                        'bin_size_1': int(match.group(2)),
                        'bin_size_2': int(match.group(3)),
                        'bin_size_merged': int(match.group(4)),
                        'profitable': int(match.group(5)),
                        'distance': float(match.group(6)),
                        'total_time': int(match.group(7)),
                        'ranking_time': int(match.group(8)),
                        'align_time': int(match.group(9)),
                        'codegen_time': int(match.group(10)),
                        'verify_time': int(match.group(11)),
                        'update_time': int(match.group(12))
                    }
                    if not __debug__:
                        print(data)
                    count += 1
                    insertMatchInfo(conn, benchmark_id, flag_id, data)
            print(f"TOTAL: {count}")
        
    conn.close()