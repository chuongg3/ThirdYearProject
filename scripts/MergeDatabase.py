import sys
import sqlite3
import argparse
import os

# Open the database location
def openNewDatabase(DBLocation = "./NewDB.db"):
    try:
        conn = sqlite3.connect(DBLocation)
        return conn
    except sqlite3.Error as e:
        print(f"ERROR: Failed to open {DBLocation}")
        print(e)
        return None

# Initialise database with tables
def initialiseTable(conn):
    benchmarkTable = '''CREATE TABLE IF NOT EXISTS Benchmarks (
    BenchmarkName TEXT NOT NULL PRIMARY KEY
);'''

    functionTable = '''CREATE TABLE IF NOT EXISTS Functions (
    BenchmarkID INTEGER NOT NULL,
    FunctionID INTEGER NOT NULL,
    FunctionName TEXT,
    FingerprintSize REAL,
    EstimatedSize REAL,
    Encoding BLOB,
    PRIMARY KEY (BenchmarkID, FunctionID),
    FOREIGN KEY (BenchmarkID) REFERENCES Benchmarks(ROWID)
);'''

    functionPairTable = '''CREATE TABLE IF NOT EXISTS FunctionPairs (
    BenchmarkID INTEGER NOT NULL,
    Function1ID INTEGER NOT NULL,
    Function2ID INTEGER NOT NULL,
    Technique TEXT NOT NULL,
    AlignmentScore REAL,
    FrequencyDistance REAL,
    MinHashDistance REAL,
    MergeSuccessful BOOLEAN,
    MergedEstimatedSize INTEGER,
    MergedLLVMIR TEXT,
    MergedEncoding TEXT,
    PRIMARY KEY (BenchmarkID, Function1ID, Function2ID, Technique),
    FOREIGN KEY (BenchmarkID) REFERENCES Benchmarks(ROWID),
    FOREIGN KEY (Function1ID) REFERENCES Functions(FunctionID),
    FOREIGN KEY (Function2ID) REFERENCES Functions(FunctionID)
);'''

    try:
        c = conn.cursor()
        c.execute(benchmarkTable)
        c.execute(functionTable)
        c.execute(functionPairTable)
        conn.commit()
        return True
    except sqlite3.Error as e:
        print("ERROR: Failed to initialise tables")
        print(e)
        return False

# TODO: CHECK THIS FUNCTION AGAIN
def attachDatabase(conn, DBLocation):
    # Verify that the DBLocation exists
    if not os.path.isfile(DBLocation):
        print(f"Database file {DBLocation} does not exist.")
        return False

    # Attach the database
    try:
        c = conn.cursor()
        c.execute(f"ATTACH DATABASE '{DBLocation}' AS DB2")
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"ERROR: Failed to attach {DBLocation}")
        print(e)
        return False

# Detach the database
def detachDatabase(conn):
    try:
        c = conn.cursor()
        c.execute("DETACH DATABASE DB2")
        conn.commit()
        return True
    except sqlite3.Error as e:
        print("ERROR: Failed to detach database")
        print(e)
        return False

def getHighestBenchmarkID(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT MAX(ROWID) FROM Benchmarks")
        result = c.fetchone()
        max_benchmark_id = result[0] if result[0] is not None else 0
        return max_benchmark_id
    
    except sqlite3.Error as e:
        print("ERROR: Failed to get highest benchmark ID")
        print(e)
        return None

# Insert the attached database information into the new databasex``
def InsertAttachedDatabase(conn, increment):
    InsertBenchmarks = f"INSERT INTO Benchmarks(ROWID, BenchmarkName) SELECT (ROWID + {increment}), BenchmarkName FROM DB2.Benchmarks"
    InsertFunctions = f"INSERT INTO Functions SELECT (BenchmarkID + {increment}), FunctionID, FunctionName, FingerprintSize, EstimatedSize, Encoding FROM DB2.Functions"
    InsertFuncionPairs = f"INSERT INTO FunctionPairs SELECT (BenchmarkID + {increment}), Function1ID, Function2ID, Technique, AlignmentScore, FrequencyDistance, MinHashDistance, MergeSuccessful, MergedEstimatedSize, MergedLLVMIR, MergedEncoding FROM DB2.FunctionPairs"

    try:
        c = conn.cursor()
        c.execute(InsertBenchmarks)
        c.execute(InsertFunctions)
        c.execute(InsertFuncionPairs)
        conn.commit()
        return True
    except sqlite3.Error as e:
        print("ERROR: Failed to insert attached database")
        print(e)
        return False



def main():
    parser = argparse.ArgumentParser(description="Handle variable arguments with an optional flag")

    # Positional arguments (variable amount)
    parser.add_argument("values", nargs="+", help="List of values")

    # Optional flag (only recognized when explicitly passed)
    parser.add_argument("-o", "--output", type=str, default = "./NewDB.db", help="Optional argument with a flag")

    # Parse arguments
    args = parser.parse_args()

    # Create the new database
    NewDatabaseLocation = args.output
    conn = openNewDatabase(NewDatabaseLocation)
    if conn is None:
        return

    # Initialise the tables
    if not initialiseTable(conn):
        return
    
    # Loop through all databases
    arguments = args.values
    for DBLocation in arguments:
        print(f"Processing: {DBLocation}")

        # Attach the the database
        if not attachDatabase(conn, DBLocation):
            continue
        
        # Retrieve the highest benchmark ID
        maxBenchID = getHighestBenchmarkID(conn)
        if (maxBenchID is None):
            continue
        
        # Insert attached database to new database
        if not InsertAttachedDatabase(conn, maxBenchID):
            continue
        
        # Detach the database
        if not detachDatabase(conn):
            continue

    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()