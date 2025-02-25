import ir2vec
import sqlite3
import pickle
import os

# def EncodeBitcodeFile(bitcodeFile):
#     print("Embedding bitcode")
#     try:
#         initObj = ir2vec.initEmbedding(bitcodeFile, "fa", "p", 100, "./a.emb")
#         print("Finished embedding bitcode")
#     except Exception as e:
#         print(f"Error during embedding: {e}")
#         return None
#     FunctionMap = initObj.getFunctionVector()
#     print("Finished getting function map")
#     return FunctionMap

# Open the encoding file and process it, load each item into a map
def getFunctionMapString(embeddingFile) :
    print("Generating Function Map")
    encoding_map = {}
    with open(embeddingFile, 'r') as file:
        for line in file:
            # Split the line into function name and vector
            FunctionName, Vector = line.strip().split("\t=\t")
            # print("Function Name: " + FunctionName[11:])

            # Convert the vector into a serialized form
            Vector =  ",".join(Vector.strip().split())
            # print("Vector: " + Vector)

            # Add the vector to the dictionary
            encoding_map[FunctionName[11:]] = Vector
    print("Finished Generating Function Map")
    return encoding_map

# Open the encoding file and process it, load each item into a map
def getFunctionMapVector(embeddingFile) :
    print("Generating Function Map")
    encoding_map = {}
    with open(embeddingFile, 'r') as file:
        for line in file:
            # Split the line into function name and vector
            FunctionName, Vector = line.strip().split("\t=\t")
            # print("Function Name: " + FunctionName[11:])
            
            # Convert the vector into a list of floats
            Vector = list(map(float, Vector.strip().split()))
            # print("Vector: " + str(Vector))

            # Add the vector to the dictionary
            encoding_map[FunctionName[11:]] = Vector
    print("Finished Generating Function Map")
    return encoding_map

# Open the encoding file and process it, load each item into a map
def getFunctionMapBlob(embeddingFile) :
    print("Generating Function Map")
    encoding_map = {}
    with open(embeddingFile, 'r') as file:
        for line in file:
            # Split the line into function name and vector
            FunctionName, Vector = line.strip().split("\t=\t")
            # print("Function Name: " + FunctionName[11:])
            
            # Convert the vector into a list of floats
            Vector = list(map(float, Vector.strip().split()))
            # print("Vector: " + str(Vector))

            # Add the vector to the dictionary
            encoding_map[FunctionName[11:]] = pickle.dumps(Vector)
    print("Finished Generating Function Map")
    return encoding_map

# Open the sqlite database given its location
def openDatabase(DBLocation):
    try:
        conn = sqlite3.connect(DBLocation)
        print(f"Opened database successfully at {DBLocation}")
        return conn
    except sqlite3.Error as e:
        print(f"Error opening database: {e}")
        return None

# Retrieve the benchmark ID given its name
def getBenchmarkID(conn, BenchmarkName):
    if conn is None:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT ROWID FROM Benchmarks WHERE BenchmarkName = ?", (BenchmarkName,))
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            print(f"No benchmark found with name: {BenchmarkName}")
            return None
    except sqlite3.Error as e:
        print(f"Error querying database: {e}")
        return None

# Retrieves all functionIDs and Names for a given benchmark
def getFunctionsIDandName(conn, benchmarkID):
    if conn is None:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT FunctionID, FunctionName FROM Functions WHERE BenchmarkID = ?", (benchmarkID,))
        rows = cursor.fetchall()
        if rows:
            return rows
        else:
            print(f"No functions found for benchmarkID: {benchmarkID}")
            return None
    except sqlite3.Error as e:
        print(f"Error querying database: {e}")

# Updates the function vector with the benchmark
def updateFunctionVector(conn, BenchmarkID, FunctionID, vector):
    if conn is None:
        return None
    try:
        print("Inserting FunctionID: " + str(FunctionID))
        cursor = conn.cursor()
        cursor.execute("UPDATE Functions SET Encoding = ? WHERE BenchmarkID = ? AND FunctionID = ?", (vector, BenchmarkID, FunctionID))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error updating database: {e}")

def insertFunctionVector(conn, BenchmarkID, row, FunctionMap):
    print("Inserting Function Vector")
    for FunctionID, FunctionName in row:
        # Process the function name
        functionName = FunctionName[1:].strip('"')
        updateFunctionVector(conn, BenchmarkID, FunctionID, FunctionMap[functionName])
    print("Finished Inserting Function Vector")



def main():
    # BitcodeLocation = "/home/chuongg3/Projects/TYP/f3m_exp/benchmarks/spec2017/605.mcf_s/_main_._all_._files_._linked_.bc"
    # EncodeBitcodeFile(BitcodeLocation)

    # ===== FOR LOCAL RUNNING =====
    # BENCH = "605.mcf_s"
    # EmbeddingLocation = "/home/chuongg3/Projects/TYP/Files/mcf.emb"
    # DatabaseLocation = "/home/chuongg3/Projects/ThirdYearProject/scripts/log/database-mcf.db"

    # ===== Get Environment Variable =====
    BENCH = os.getenv("BENCH")
    EmbeddingLocation = os.getenv("EMBLOC")
    DatabaseLocation = os.getenv("DBLOC")

    # Get the function mapping
    FunctionMap = getFunctionMapBlob(EmbeddingLocation)
    # print(FunctionMap.keys())
    print("Number of Functions: " +str (len(FunctionMap.keys())))

    # Open the database
    conn = openDatabase(DatabaseLocation)
    BenchmarkID = getBenchmarkID(conn, BENCH)
    print("Benchmark ID: " + str(BenchmarkID))

    # If benchmark does not exist, end
    if BenchmarkID == None:
        print("ERROR: Unable to get benchmark ID")
        return
    
    # Get the function IDs and Names
    rows = getFunctionsIDandName(conn, BenchmarkID)
    if rows == None:
        print("ERROR: Unable to get function IDs and Names")
        return
    
    # Insert the function vectors
    insertFunctionVector(conn, BenchmarkID, rows, FunctionMap)

if __name__ == "__main__":
    main()