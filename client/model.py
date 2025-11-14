# Filepath: FED_IIDS/client/model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import config # Import your settings

def create_model():
    """
    Creates the shared Keras model architecture.
    This file is the "contract" between client and server.
    It reads the number of features from the config file.
    """
    inputs = Input(shape=(config.NUM_FEATURES,))
    
    # A simple but effective Deep Neural Network (DNN)
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    
    # Binary classification output: 0=Normal, 1=Attack
    outputs = Dense(1, activation='sigmoid')
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    # A simple check to print the model summary
    print(f"Building model with {config.NUM_FEATURES} input features.")
    model = create_model()
    model.summary()