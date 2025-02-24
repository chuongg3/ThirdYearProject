import time
import optuna
import pickle

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Input, Dense, Dropout, Layer


from TrainFunctions import DumpHistory
from LoadData import CreateTensorflowDataset



# Define the base network for feature extraction
def create_base_network(input_shape, dropout = 0.25, units = [256, 128]):
    inputs = Input(shape=input_shape)
    x = Dense(units[0], activation='relu')(inputs)
    x = Dropout(dropout)(x)
    outputs = Dense(units[1], activation='relu')(x)
    return Model(inputs, outputs)

@register_keras_serializable()
class L1Distance(Layer):
    def __init__(self, **kwargs):
        super(L1Distance, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return K.abs(x - y)

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Output shape is the same as the input shapes

def get_model(loss="mean_squared_error", optimizer="adam", learning_rate=0.001, metrics = ['mse', 'mae', 'mape'], dropout=0.25, units=[256, 128]):
    # Define input shape
    input_shape = (300,)

    # Create the base network
    base_network = create_base_network(input_shape, dropout, units)

    # Siamese network inputs
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Generate embeddings
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    distance = L1Distance()([embedding_a, embedding_b])

    # Output layer for similarity score (0 to 1 range)
    output = Dense(1, activation='sigmoid')(distance)

    # Define the Siamese model
    siamese_model = Model(inputs=[input_a, input_b], outputs=output)

    # Compile the model
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    optimizer_instance.learning_rate = learning_rate
    siamese_model.compile(loss=loss, optimizer=optimizer_instance, metrics=metrics)

    # Model summary
    siamese_model.summary()

    return siamese_model

'''
===== HYPERPARAMETER TRAINING =====
This section is for hyperparameter training using optuna
Hyper-Parameters:
 - Epoch
 - Dropout Rate
 - Learning Rate
 - Number of Units
 - Optimizer
 - Batch Size

Best model will be saved to best_model.keras
===================================
'''

# Hyperparameter training
def HyperParameterTraining(DATA_PATH, metrics = ['mse', 'mae', 'mape'], n_trials = 2, bestModelPath = "./BestModel.keras"):
    histories = []
    bestModelScore = float("inf")

    # Define the objective function
    def objective(trial):
        nonlocal DATA_PATH
        nonlocal histories
        nonlocal bestModelScore

        # Get a range of values
        epochs = trial.suggest_int('epochs', 2, 15)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        batch_size = trial.suggest_int('batch_size', 32, 128, step=16)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
        # optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
        optimizer = trial.suggest_categorical('optimizer', ['adam'])
        units = [
            trial.suggest_int('units_1', 128, 512, step=64),
            trial.suggest_int('units_2', 64, 256, step=64)
        ]

        # Early Stopper to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Load the dataset
        training_set, validation_set, test_set = CreateTensorflowDataset(DATA_PATH, batch_size=batch_size, overwrite=False)

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
        information['units'] = units
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
