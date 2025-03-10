import time
import optuna
import pickle

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Input, Dense, Dropout, Dot, Activation, Normalization, BatchNormalization, Add, Multiply

from TrainFunctions import DumpHistory
from LoadData import CreateTensorflowDataset, CreateNumpyDataset

# Shared encoder with multiple Dense layers
def encoder(input_shape = (300,), dropout = 0.3, units = [512, 128]):
    inputs = Input(shape=input_shape)

    x = Dense(units[0], activation="relu")(inputs)
    x = Dropout(dropout)(x)

    x = Dense(units[1], activation="relu")(x)

    normalizer = BatchNormalization(axis=-1)
    x = normalizer(x)

    return Model(inputs, x)

def get_model(loss="mean_squared_error", optimizer="adam", learning_rate=0.001, metrics = ['mse', 'mae', 'mape'], dropout=0.25, units=[512, 128]):
    print(" ===== Creating Dot Product Siamese Model =====")

    # Define input shape
    input_shape = (300,)

    # Create the base network
    shared_encoder = encoder(input_shape=input_shape, dropout=dropout, units=units)

    # Inputs for two function embeddings
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    # Generate embeddings
    encoded1 = shared_encoder(input1)
    encoded2 = shared_encoder(input2)

    # Compute dot product similarity
    dot_product = Dot(axes=1, normalize=True)([encoded1, encoded2])

    # Normalize with sigmoid to ensure output is in [0,1]
    similarity_score = Dense(1, activation='sigmoid')(dot_product)

    # Create model
    model = Model(inputs=[input1, input2], outputs=similarity_score)

    # Compile model with regression loss
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    optimizer_instance.learning_rate = learning_rate
    model.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics)

    # Print model summary
    model.summary()

    return model

'''
===== HYPERPARAMETER TRAINING =====
This section is for hyperparameter training using optuna
Hyper-Parameters:
 - Epoch
 - Dropout Rate
 - Learning Rate
 - Optimizer
 - Batch Size

Best model will be saved to best_model.keras
===================================
'''

# Hyperparameter training
def HyperParameterTraining(DATA_PATH, metrics = ['mse', 'mae', 'mape'], n_trials = 5, bestModelPath = "./BestModel.keras", zero_weight=0.001, non_zero_weight=1):
    histories = []
    bestModelScore = float("inf")
    print(f"Hyperparameter Training with {n_trials} trials")

    # Define the objective function
    def objective(trial):
        nonlocal DATA_PATH
        nonlocal histories
        nonlocal bestModelScore

        # Get a range of values
        epochs = trial.suggest_int('epochs', 2, 10)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [64, 100])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
        optimizer = trial.suggest_categorical('optimizer', ['adam'])
        units = [
            trial.suggest_categorical('units_1', [512, 1024]),
            trial.suggest_categorical('units_2', [64, 128])
        ]

        # Early Stopper to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        print(f"""Testing Parameters:
        Epochs: {epochs}
        Dropout: {dropout}
        Batch Size: {batch_size}
        Learning Rate: {learning_rate}
        Optimizer: {optimizer}
        Units: {units}
        """)

        # Load the dataset
        # training_set, validation_set, test_set = CreateTensorflowDataset(DATA_PATH, batch_size=batch_size, overwrite=False, zero_weight=zero_weight, non_zero_weight=non_zero_weight)
        training_set, validation_set, test_set = CreateNumpyDataset(DATA_PATH, batch_size=batch_size, overwrite=False, zero_weight=zero_weight, non_zero_weight=non_zero_weight)

        # Create the model
        model = get_model(loss="mean_squared_error", optimizer=optimizer, learning_rate=learning_rate, metrics=metrics, dropout=dropout, units=units)
        startTime = time.time()
        history = model.fit(training_set, epochs=epochs, validation_data=validation_set, callbacks=[early_stopping])
        totalTime = time.time() - startTime
        print(f"Time taken for the model to run: {time.strftime('%H:%M:%S', time.gmtime(totalTime))}")

        # Get the validation loss
        val_loss = min(history.history['val_loss'])

        # Save the model if lowest validation loss
        if val_loss < bestModelScore:
            bestModelScore = val_loss
            model.save(bestModelPath)  # Save the best model

        # Save the history for the model
        information = dict()
        information['epochs'] = epochs
        information['dropout'] = dropout
        information['batch_size'] = batch_size
        information['learning_rate'] = learning_rate
        information['optimizer'] = optimizer
        information['history'] = history.history

        histories.append(information)

        return val_loss

    # Create the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Pickle the histories for all models
    DumpHistory(histories, bestModelPath)

    # Load the best model from file (no need to retrain)
    best_model = tf.keras.models.load_model(bestModelPath)

    # Print the best hyperparameters
    if study.best_params:
        print("Best hyperparameters found:", study.best_params)
    else:
        print("No valid hyperparameters found!")

    return study.best_params, best_model
