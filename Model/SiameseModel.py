import LoadData
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import pickle

# Parse any command line arguments
def ParseArguments():
    parser = argparse.ArgumentParser(description="Handle variable arguments with an optional flag")

    # Optional flag (only recognized when explicitly passed)
    parser.add_argument("-o", "--output", type=str, default = "./data/model.keras", help="Location to save model")
    parser.add_argument("-d", "--data", type=str, default = "./data/benchmarks.db", help="Input data files")

    # Parse arguments
    args = parser.parse_args()
    return args

# Split the dataset into three sets based on the alignment score
def SplitDataByAlignmentScore(data):
    print("Splitting data by alignment score")

    df_zero = data[(data['AlignmentScore'] == 0)]
    df_one = data[(data['AlignmentScore'] == 1)]
    df_non_zero = data[(data['AlignmentScore'] != 0) & (data['AlignmentScore'] != 1)]

    print(f"Number of zero scores: {df_zero.shape[0]}")
    print(f"Number of one scores: {df_one.shape[0]}")
    print(f"Number of non-zero scores: {df_non_zero.shape[0]}")
    print(f"Total number of scores: {df_zero.shape[0] + df_one.shape[0] + df_non_zero.shape[0]}")
    print(f"Actual number of scores: {data.shape[0]}")

    return df_zero, df_one, df_non_zero

# Split the data into training, validation and testing sets
def SplitTrainingValidationTesting(data, test_percentage = 0.2, val_percentage = 0.1, seed = 42):
    print("Splitting data into training, validation and testing sets")
    validation_split = val_percentage / (1 - test_percentage)

    train_set, test_set = train_test_split(data, test_size=test_percentage, random_state=seed)
    train_set, val_set = train_test_split(train_set, test_size=validation_split, random_state=seed)

    print(f"Original Set: {data.shape[0]}")
    print(f"Training Set: {train_set.shape[0]} | Validation Set: {val_set.shape[0]} | Testing Set: {test_set.shape[0]}")

    return train_set, val_set, test_set

# Converts a pandas series of python list into a 2D numpy array
def ConvertSeriesToNDArray(series):
    return np.stack(series.to_numpy())

# Saves the training history and model
def SaveHistoryAndModel(history, model, history_loc, model_loc):
    with open(history_loc, 'wb') as f:
        pickle.dump(history, f)

    model.save(model_loc)

# Create the siamese model
def CreateSiameseModel():
    from tensorflow.keras.layers import Input, Dense, Lambda
    from tensorflow.keras.models import Model
    import tensorflow.keras.backend as K

    # Define the base network for feature extraction
    def create_base_network(input_shape):
        inputs = Input(shape=input_shape)
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(64, activation='sigmoid')(x)  # Feature vector, maybe change this?
        return Model(inputs, outputs)

    # Define input shape
    input_shape = (300,)

    # Create the base network
    base_network = create_base_network(input_shape)

    # Siamese network inputs
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Generate embeddings
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    # Compute L1 distance
    def l1_distance(vectors):
        x, y = vectors
        return K.abs(x - y)

    distance = Lambda(l1_distance)([embedding_a, embedding_b])

    # Output layer for similarity score (0 to 1 range)
    output = Dense(1, activation='sigmoid')(distance)

    # Define the Siamese model
    siamese_model = Model(inputs=[input_a, input_b], outputs=output)

    # Compile the model
    siamese_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])

    # Model summary
    siamese_model.summary()

    return siamese_model


def main():
    args = ParseArguments()

    # Load Data and Deserialise
    DBLoc = args.data
    conn = LoadData.connectToDB(DBLoc)
    data = LoadData.LoadAllEncodings(conn)

    # Convert the binary data back into Python objects
    data['Encoding1'] = data['Encoding1'].apply(LoadData.deserialize_encoding)
    data['Encoding2'] = data['Encoding2'].apply(LoadData.deserialize_encoding)

    # Split the data by its Alignment Score
    df_zero, df_one, df_non_zero = SplitDataByAlignmentScore(data)

    # Split the data into training, validation and testing
    train_zero, val_zero, test_zero = SplitTrainingValidationTesting(df_zero)
    train_one, val_one, test_one = SplitTrainingValidationTesting(df_one)
    train_non_zero, val_non_zero, test_non_zero = SplitTrainingValidationTesting(df_non_zero)

    # Calculate weightings to give to training samples
    zero_ratio = train_zero.shape[0] / (train_non_zero.shape[0] + train_zero.shape[0] + train_one.shape[0])
    print(f"Zero ratio: {zero_ratio}")
    print(f"Non-zero ratio: {1 - zero_ratio}")

    # Combine the data into their sets
    train_set = pd.concat([train_zero, train_one, train_non_zero])
    validation_set = pd.concat([val_zero, val_one, val_non_zero])
    test_set = pd.concat([test_zero, test_one, test_non_zero])
    print(f"Training Set Shape: {train_set.shape}")
    print(f"Validation Set Shape: {validation_set.shape}")
    print(f"Testing Set Shape: {test_set.shape}")

    # Give the zero scores a lower weight as there are a lot more samples
    sample_weights = np.where(train_set['AlignmentScore'] == 0, 1 - zero_ratio, zero_ratio)

    # Convert the training data into numpy arrays
    Encoding1_Train = ConvertSeriesToNDArray(train_set['Encoding1'])
    Encoding2_Train = ConvertSeriesToNDArray(train_set['Encoding2'])
    AlignmentScore_Train = train_set['AlignmentScore'].to_numpy(dtype=float)

    # Convert the validation data into numpy arrays
    Encoding1_Val = ConvertSeriesToNDArray(validation_set['Encoding1'])
    Encoding2_Val = ConvertSeriesToNDArray(validation_set['Encoding2'])
    AlignmentScore_Val = validation_set['AlignmentScore'].to_numpy(dtype=float)

    # Convert the testing data into numpy arrays
    Encoding1_Test = ConvertSeriesToNDArray(test_set['Encoding1'])
    Encoding2_Test = ConvertSeriesToNDArray(test_set['Encoding2'])
    AlignmentScore_Test = test_set['AlignmentScore'].to_numpy(dtype=float)
    
    # Siamese Model
    siamese_model = CreateSiameseModel()
    history = siamese_model.fit([Encoding1_Train, Encoding2_Train], AlignmentScore_Train, sample_weight=sample_weights,batch_size=32, epochs=10, validation_data=([Encoding1_Val, Encoding2_Val], AlignmentScore_Val))

    # Keep the history and save the model
    SaveHistoryAndModel(history.history, siamese_model, "./data/history.pkl", args.output)

    # Use the model to predict value
    predictions = siamese_model.predict([Encoding1_Test, Encoding2_Test])

    # Evaluate the model
    siamese_model.evaluate([Encoding1_Test, Encoding2_Test], AlignmentScore_Test)


    
if __name__ == "__main__":
    main()