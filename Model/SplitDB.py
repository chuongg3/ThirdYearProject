import sqlite3
import os

# Connect to the database
def connectDB(DB_FILE):
    if not os.path.isfile(DB_FILE):
        raise FileNotFoundError(f"The database file {DB_FILE} does not exist.")
    return sqlite3.connect(DB_FILE, check_same_thread=False)

# Generate temporary table for each alignment score
def generateTempTable(conn, score):
    # Different filtering conditions for each alignment score
    if score != 0 and score != 1:
        condition = f"AlignmentScore > 0 AND AlignmentScore < 1"
        name = "NonZero"
    else:
        condition = f"AlignmentScore = {score}"
        name = score

    # Create and insert relevant data into temporary table
    select_query = f"""SELECT * FROM FunctionPairs WHERE {condition}"""
    create_table_query = f"""CREATE TEMP TABLE FunctionPairs_{name} AS {select_query}"""    

    cursor = conn.cursor()
    cursor.execute(create_table_query)

# Initialise database with tables
def initialiseTable(conn, DBName):
    benchmarkTable = f'''CREATE TABLE IF NOT EXISTS {DBName}.Benchmarks (
    BenchmarkName TEXT NOT NULL PRIMARY KEY
);'''

    functionTable = f'''CREATE TABLE IF NOT EXISTS {DBName}.Functions (
    BenchmarkID INTEGER NOT NULL,
    FunctionID INTEGER NOT NULL,
    FunctionName TEXT,
    FingerprintSize REAL,
    EstimatedSize REAL,
    Encoding BLOB,
    PRIMARY KEY (BenchmarkID, FunctionID),
    FOREIGN KEY (BenchmarkID) REFERENCES Benchmarks(ROWID)
);'''

    functionPairTable = f'''CREATE TABLE IF NOT EXISTS {DBName}.FunctionPairs (
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

# Attach to a database
def attachDatabase(conn, DBLocation, DBName):
    print(f"Attaching {DBLocation} as {DBName}")

    # Verify that the DBLocation exists
    if not os.path.isfile(DBLocation):
        print(f"Database file {DBLocation} does not exist.")
        return False

    # Attach the database
    try:
        c = conn.cursor()
        c.execute(f"ATTACH DATABASE '{DBLocation}' AS {DBName}")
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"ERROR: Failed to attach {DBLocation}")
        print(e)
        return False

# Detach the database
def detachDatabase(conn, DBName):
    try:
        c = conn.cursor()
        c.execute(f"DETACH DATABASE {DBName}")
        conn.commit()
        return True
    except sqlite3.Error as e:
        print("ERROR: Failed to detach database")
        print(e)
        return False

# Insert a shuffled ROWID column into the specified table
def insertShuffledROWIDColumn(conn, table_name):
    print(f"Inserting shuffled ROWID into {table_name}")

    # Add a column to store the shuffled ROWID
    print(f"Adding SHUFFLED_ID column to {table_name}")
    cursor = conn.cursor()
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN SHUFFLED_ID INTEGER")

    # Update the shuffled ROWID
    print("Updating the table with shuffled ROWID")
    query = f"""WITH shuffled AS (
                SELECT ROWID, 
                ROW_NUMBER() OVER () AS num
                FROM (SELECT ROWID FROM {table_name} ORDER BY RANDOM()))
                UPDATE {table_name}
                SET SHUFFLED_ID = (SELECT num FROM shuffled WHERE shuffled.ROWID = {table_name}.ROWID);"""
    cursor.execute(query)

# Splits the function pairs into training, validation, and testing sets using randomised ROWID
def splitFunctionPairsSHUFFLEDID(conn, src_table_name, table_size, split = (0.7, 0.1, 0.2)):
    print(f"Splitting {src_table_name} into TEST, TRAIN, and VAL")

    threshold1 = split[0] * table_size
    threshold2 = split[1] * table_size + threshold1

    columns = f"BenchmarkID, Function1ID, Function2ID, Technique, AlignmentScore, FrequencyDistance, MinHashDistance, MergeSuccessful, MergedEstimatedSize, MergedLLVMIR, MergedEncoding"
    

    train_query = f"INSERT INTO TRAIN.FunctionPairs SELECT {columns} FROM {src_table_name} WHERE SHUFFLED_ID < {threshold1}"
    validation_query = f"INSERT INTO VAL.FunctionPairs SELECT {columns} FROM {src_table_name} WHERE SHUFFLED_ID >= {threshold1} AND SHUFFLED_ID < {threshold2}"
    test_query = f"INSERT INTO TEST.FunctionPairs SELECT {columns} FROM {src_table_name} WHERE SHUFFLED_ID >= {threshold2}"

    cursor = conn.cursor()
    cursor.execute(train_query)
    cursor.execute(validation_query)
    cursor.execute(test_query)
    conn.commit()

    return True

# Copies the necessary Functions rows into the DBName database
def copyFunctionDetails(conn, DBName):
    print(f"Copying function details into {DBName}")

    function_query = f"""INSERT INTO {DBName}.Functions SELECT f.*
FROM Functions f
JOIN (SELECT BenchmarkID, Function1ID AS FunctionID FROM {DBName}.FunctionPairs
      UNION 
      SELECT BenchmarkID, Function2ID AS FunctionID FROM {DBName}.FunctionPairs) AS fp
ON f.BenchmarkID = fp.BenchmarkID AND f.FunctionID = fp.FunctionID"""
    
    cursor = conn.cursor()
    cursor.execute(function_query)
    conn.commit()

    return True

# Copies the necessary Benchmarks rows into the DBName database
def copyBenchmarkDetails(conn, DBName):
    print(f"Copying benchmark details into {DBName}")

    benchmark_query = f"""INSERT INTO {DBName}.Benchmarks(ROWID, BenchmarkName) 
    SELECT ROWID, BenchmarkName FROM Benchmarks b
    JOIN (SELECT DISTINCT(BenchmarkID) FROM {DBName}.FunctionPairs) AS fp
     ON b.ROWID = fp.BenchmarkID"""
    
    cursor = conn.cursor()
    cursor.execute(benchmark_query)
    conn.commit()

    return True

# Generates a temporary directory to store the new databases
def generateTempDirectory(basedir):
    print(f"Creating temporary directory")
    # Generate the temporary directory
    temp_dir = os.path.join(basedir, '.temp')
    try:
        os.makedirs(temp_dir)
    except FileExistsError:
        pass

    # Create the databases
    fd = os.open(os.path.join(temp_dir, 'train.db'), os.O_CREAT | os.O_WRONLY)
    os.close(fd)
    fd = os.open(os.path.join(temp_dir, 'validation.db'), os.O_CREAT | os.O_WRONLY)
    os.close(fd)
    fd = os.open(os.path.join(temp_dir, 'test.db'), os.O_CREAT | os.O_WRONLY)
    os.close(fd)

    return temp_dir

# Print the schema of the database, including any attached or temporary databases
def print_schema(conn):
    cursor = conn.cursor()

    # Query to get the schema of the main database
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    schema_statements = cursor.fetchall()

    # Print the schema of the main database
    print("Schema of main database:")
    for statement in schema_statements:
        if statement[0]:  # Ensure the statement is not None
            print(statement[0] + ';')

    # Query to get the names of attached databases
    cursor.execute("PRAGMA database_list;")
    databases = cursor.fetchall()

    # Print the schema of each attached database
    for db in databases:
        db_name = db[1]
        if db_name != 'main':  # Skip the main database
            print(f"\nSchema of {db_name} database:")
            cursor.execute(f"SELECT sql FROM {db_name}.sqlite_master WHERE type='table'")
            schema_statements = cursor.fetchall()
            for statement in schema_statements:
                if statement[0]:  # Ensure the statement is not None
                    print(statement[0] + ';')

# Get the size of a table
def getSizeOfTable(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    result = cursor.fetchone()
    return result[0]

# Split the database into training, validation, and testing sets
def SplitDB(DB_FILE, split = (0.7, 0.1, 0.2)):
    # Connect to source database
    print(f"Connecting to database")
    dirname = os.path.dirname(DB_FILE)
    conn = connectDB(DB_FILE)

    # Generate temporary tables for each alignment score
    print(f"Generating Temporary AlignmentScore Tables")
    generateTempTable(conn, 0)
    generateTempTable(conn, 1)
    generateTempTable(conn, -1)

    # Print the table size
    print(f"FunctionPairs_0: {getSizeOfTable(conn, 'temp.FunctionPairs_0')}")
    print(f"FunctionPairs_1: {getSizeOfTable(conn, 'temp.FunctionPairs_1')}")
    print(f"FunctionPairs_NonZero: {getSizeOfTable(conn, 'temp.FunctionPairs_NonZero')}")

    # Create a temporary directory to store the databases
    temp_dir = generateTempDirectory(dirname)

    # Attach the train, validation and test databases
    print(f"Attaching databases")
    attachDatabase(conn, os.path.join(temp_dir, 'train.db'), 'TRAIN')
    attachDatabase(conn, os.path.join(temp_dir, 'validation.db'), 'VAL')
    attachDatabase(conn, os.path.join(temp_dir, 'test.db'), 'TEST')

    # Create tables for each database
    print(f"Initialising tables")
    initialiseTable(conn, 'TRAIN')
    initialiseTable(conn, 'VAL')
    initialiseTable(conn, 'TEST')

    # Insert the random numbers for each table
    temp_tables = ['temp.FunctionPairs_0', 'temp.FunctionPairs_1', 'temp.FunctionPairs_NonZero']
    for table in temp_tables:
        insertShuffledROWIDColumn(conn, table)

    # Print temporary table size
    print(f"FunctionPairs_0: {getSizeOfTable(conn, 'temp.FunctionPairs_0')}")
    print(f"FunctionPairs_1: {getSizeOfTable(conn, 'temp.FunctionPairs_1')}")
    print(f"FunctionPairs_NonZero: {getSizeOfTable(conn, 'temp.FunctionPairs_NonZero')}")

    # Split the data into training, validation, and testing sets
    for table in temp_tables:
        table_size = getSizeOfTable(conn, table)
        splitFunctionPairsSHUFFLEDID(conn, table, table_size, split)

    # Print the size of the training, validation, and testing sets
    print(f"TRAIN: {getSizeOfTable(conn, 'TRAIN.FunctionPairs')}")
    print(f"VAL: {getSizeOfTable(conn, 'VAL.FunctionPairs')}")
    print(f"TEST: {getSizeOfTable(conn, 'TEST.FunctionPairs')}")

    # Insert the Functions Details
    copyFunctionDetails(conn, 'TRAIN')
    copyFunctionDetails(conn, 'VAL')
    copyFunctionDetails(conn, 'TEST')

    # Insert the benchmark Details
    copyBenchmarkDetails(conn, 'TRAIN')
    copyBenchmarkDetails(conn, 'VAL')
    copyBenchmarkDetails(conn, 'TEST')

    # Detach the train, validation and test databases
    detachDatabase(conn, 'TRAIN')
    detachDatabase(conn, 'VAL')
    detachDatabase(conn, 'TEST')

    # Close the connection
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split the database into training, validation, and testing sets")
    parser.add_argument("--DB_FILE", type=str, default="./data/benchmark-cp.db", help="The database file to split")
    # parser.add_argument("-s", "--split", type=float, nargs=3, default=(0.7, 0.1, 0.2), help="The split ratio for training, validation, and testing sets")
    args = parser.parse_args()

    SplitDB(args.DB_FILE)
