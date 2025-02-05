CREATE TABLE IF NOT EXISTS Benchmarks (
    BenchmarkName TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Functions (
    BenchmarkID INTEGER NOT NULL,
    FunctionID INTEGER NOT NULL,
    FunctionName TEXT,
    FingerprintSize REAL,
    EstimatedSize REAL,
    Encoding BLOB,
    PRIMARY KEY (BenchmarkID, FunctionID),
    FOREIGN KEY (BenchmarkID) REFERENCES Benchmarks(ROWID)
);

CREATE TABLE IF NOT EXISTS FunctionPairs (
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
);