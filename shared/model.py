import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

from shared import model_config

def create_model():
    """
    Creates the shared Keras (TensorFlow) model architecture.
    This is the "API Contract" that both client and server MUST use.
    """
    
    # 1. Define the Input Layer
    inputs = Input(shape=(model_config.NUM_FEATURES,))
    
    # 2. Define the Hidden Layers
    x = Dense(64, activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    
    # 3. Define the Output Layer
    outputs = Dense(1, activation="sigmoid")(x)

    # 4. Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # This compile step is a placeholder.
    # The *real* optimizer (DP-SGD) will be applied
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model