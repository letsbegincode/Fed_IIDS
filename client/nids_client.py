# FED_IIDS/client/nids_client.py
#
# === FINAL CORRECTED VERSION ===
# This version fixes the 'TypeError' by adding the
# required 'config' argument to the get_parameters and evaluate functions.

import flwr as fl
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import numpy as np

import config
import model

class NIDSClient(fl.client.NumPyClient):
    """
    The main Flower client class for our NIDS.
    """
    def __init__(self, client_id, x_train, y_train, x_test, y_test):
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Create the model using the shared model.py
        self.model = model.create_model()

    # --- THIS IS THE FIX ---
    # The Flower library requires the 'config' argument in the signature
    # for all three of its main methods.
    #
    # OLD (BUGGY) CODE:
    # def get_parameters(self):
    #
    # NEW (CORRECT) CODE:
    def get_parameters(self, config):
        """Returns the current local model weights."""
        print(f"[{self.client_id}] get_parameters")
        return self.model.get_weights()
    # ---------------------

    def fit(self, parameters, config_server):
        """
        Train the local model using the new parameters from the server.
        """
        print(f"[{self.client_id}] fit")
        
        # 1. Update local model with server's parameters
        self.model.set_weights(parameters)

        # 2. Get local training config from the server's message
        local_epochs = config_server.get("local_epochs", config.DEFAULT_LOCAL_EPOCHS)
        batch_size = config_server.get("batch_size", config.DEFAULT_BATCH_SIZE)

        # 3. --- Path A: Apply Differential Privacy ---
        # We must create a *new* DP optimizer for each 'fit' call
        # to correctly track the privacy budget.
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=config.DP_L2_NORM_CLIP,
            noise_multiplier=config.DP_NOISE_MULTIPLIER,
            num_microbatches=config.DP_MICROBATCHES,
            learning_rate=config.DP_LEARNING_RATE
        )
        
        # 4. Re-compile the model with the DP optimizer
        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # 5. Train the model
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=local_epochs,
            batch_size=batch_size,
            validation_split=0.1,  # Use 10% of train data for validation
            verbose=2
        )

        # 6. Return the updated local model weights and sample count
        return self.model.get_weights(), len(self.x_train), {}

    # --- THIS IS THE OTHER FIX ---
    # The 'evaluate' function also requires the 'config' argument.
    #
    # OLD (BUGGY) CODE:
    # def evaluate(self, parameters):
    #
    # NEW (CORRECT) CODE:
    def evaluate(self, parameters, config):
        """
        Evaluate the provided parameters (from the global model) on
        the local test set.
        """
        print(f"[{self.client_id}] evaluate")
        
        # 1. Update local model with server's parameters
        self.model.set_weights(parameters)
        
        # 2. Re-compile (needed after 'fit' changed the optimizer)
        self.model.compile(
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # 3. Evaluate on the *local test set*
        loss, accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=0
        )
        
        print(f"[{self.client_id}] Evaluate Loss: {loss}, Accuracy: {accuracy}")

        # 4. Return results to the server
        return loss, len(self.x_test), {"accuracy": accuracy}