import time
import optuna
from TrainFunctions import DumpHistory
from LoadData import CreateNumpyDataset

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint


def simple_transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-head self-attention with dropout and skip connection.
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.Add()([inputs, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward block with dropout and skip connection.
    ff_output = layers.Dense(ff_dim, activation='relu')(x)
    ff_output = layers.Dense(inputs.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    return x

def get_model(loss="mean_squared_error", optimizer="sgd", learning_rate=0.0008483268515270536, metrics=['mse', 'mae', 'mape'], dropout=0.3757168539447645, head_size=64, num_heads=4, ff_dim=64):
    print(" ====== Creating Multi-Head Attention Model ====== ")    

    # Define two separate 300-dimensional inputs.
    input_vec1 = tf.keras.Input(shape=(300,), name="input_vec1")
    input_vec2 = tf.keras.Input(shape=(300,), name="input_vec2")
    
    # Encode each vector with a Dense layer (512 units) and ReLU activation.
    encode_dense = layers.Dense(512, activation='relu')
    vec1_encoded = encode_dense(input_vec1)
    vec2_encoded = encode_dense(input_vec2)
    
    # Reshape each encoded vector to add a sequence dimension.
    vec1_reshaped = layers.Reshape((1, 512))(vec1_encoded)
    vec2_reshaped = layers.Reshape((1, 512))(vec2_encoded)
    
    # Concatenate the two reshaped vectors to form a sequence of 2 tokens.
    sequence = layers.Concatenate(axis=1)([vec1_reshaped, vec2_reshaped])
    
    # Pass the sequence through the transformer encoder.
    x = simple_transformer_encoder(sequence, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
    
    # Apply global average pooling to aggregate the token representations.
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final dense layer with sigmoid activation for output in [0, 1].
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # Build and compile the model.
    model = Model(inputs=[input_vec1, input_vec2], outputs=output)
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    optimizer_instance.learning_rate = learning_rate
    model.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics)
    
    # Summarise the model and plot
    model.summary()
    plot_model(model, to_file='MultiHeadAttentionModel.png', show_shapes=True, show_layer_names=True)

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
 - Head Size
 - Number of Heads
 - Feed Forward Dimension

Best model will be saved to best_model.keras
===================================
'''

# Hyperparameter training
def HyperParameterTraining(DATA_PATH, metrics = ['mse', 'mae', 'mape'], n_trials = 3, bestModelPath = "./BestModel.keras", zero_weight=0.001, non_zero_weight=1):
    histories = []
    bestModelScore = float("inf")
    print(f"Hyperparameter Training with {n_trials} trials")
    print(f"Zero Weight: {zero_weight}")
    print(f"Non-Zero Weight: {non_zero_weight}")

    # Define the objective function
    def objective(trial):
        nonlocal DATA_PATH
        nonlocal histories
        nonlocal bestModelScore

        # Get a range of values
        epochs = trial.suggest_int('epochs', 2, 7)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [16, 32])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
        optimizer = trial.suggest_categorical('optimizer', ['adam'])
        head_size = trial.suggest_categorical('head_size', [32, 64, 128, 256])
        num_heads = trial.suggest_int('num_heads', 1, 8)
        ff_dim = trial.suggest_categorical('ff_dim', [64, 128, 256])

        # Early Stopper to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Model Checkpoint to save the best model
        checkpoint_callback = ModelCheckpoint(filepath='Transformer_{epochs}_{dropout}_{batch_size}_{learning_rate}_{optimizer}_{head_size}_{num_heads}_{ff_dim}.keras', save_best_only=True, monitor='val_loss', mode='min', save_weights_only=False)

        print(f"""Testing Parameters:
        Epochs: {epochs}
        Dropout: {dropout}
        Batch Size: {batch_size}
        Learning Rate: {learning_rate}
        Optimizer: {optimizer}
        Head Size: {head_size}
        Number of Heads: {num_heads}
        Feed Forward Dimension: {ff_dim}""")

        # Load the dataset
        # training_set, validation_set, test_set = CreateTensorflowDataset(DATA_PATH, batch_size=batch_size, overwrite=False, zero_weight=zero_weight, non_zero_weight=non_zero_weight)
        training_set, validation_set, test_set = CreateNumpyDataset(DATA_PATH, batch_size=batch_size, overwrite=False, zero_weight=zero_weight, non_zero_weight=non_zero_weight)

        # Create the model
        model = get_model(loss="mean_squared_error", optimizer=optimizer, learning_rate=learning_rate, metrics=metrics, dropout=dropout, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim)
        startTime = time.time()
        history = model.fit(training_set, epochs=epochs, validation_data=validation_set, callbacks=[early_stopping, checkpoint_callback])
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