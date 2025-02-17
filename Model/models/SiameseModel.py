from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import tensorflow as tf

# Define the base network for feature extraction
def create_base_network(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    # outputs = Dense(64, activation='sigmoid')(x)  # Feature vector
    outputs = Dropout(0.25)(x)
    return Model(inputs, outputs)


def get_model(loss="mean_squared_error", optimizer="adam", learning_rate=0.001, metrics = ['mse']):
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
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    optimizer_instance.learning_rate = learning_rate
    siamese_model.compile(loss=loss, optimizer=optimizer_instance, metrics=metrics)

    # Model summary
    siamese_model.summary()

    return siamese_model