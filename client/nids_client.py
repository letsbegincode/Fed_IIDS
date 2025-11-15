import flwr as fl
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import numpy as np

from . import config
from shared import model

# --- UPDATED: Professional Minimal Logger ---
class MinimalProgressCallback(tf.keras.callbacks.Callback):
    """
    A custom, minimal Keras callback for clean federated logs.
    Shows in-place batch progress and a summary at the end of each epoch.
    """
    def __init__(self, client_id, epochs):
        super().__init__()
        self.client_id = client_id
        self.epochs = epochs
        self.steps = 0 # Total batches in this epoch
        self.current_epoch = 0 # Current epoch number

    def on_train_begin(self, logs=None):
        # Get the total number of batches
        self.steps = self.params.get('steps', 0)
        print(f"  [{self.client_id}] Starting local training ({self.steps} batches per epoch)...")

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        print(f"  [{self.client_id}] Epoch {self.current_epoch}/{self.epochs}")

    def on_batch_end(self, batch, logs=None):
        batch_num = batch + 1
        loss = logs.get('loss', 0.0)
        # Use carriage return \r to update the line in-place
        # This shows "live" progress for the current epoch
        print(f"\r    > Batch {batch_num}/{self.steps} - loss: {loss:.4f}", end='', flush=True)

    def on_epoch_end(self, epoch, logs=None):
        # Print a newline to move off the \r line from on_batch_end
        print() 
        logs = logs or {}
        # Print the final validation stats for the epoch
        print(f"    > Epoch complete. -> val_loss: {logs.get('val_loss'):.4f} - val_acc: {logs.get('val_accuracy'):.4f}")

    def on_train_end(self, logs=None):
        print(f"  [{self.client_id}] Local training complete.")
# --- End of Updated Callback ---


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
        
        self.model = model.create_model() 

    def get_parameters(self, config):
        """Returns the current local model weights."""
        print(f"[{self.client_id}] get_parameters")
        return self.model.get_weights()

    def fit(self, parameters, config_server):
        """
        Train the local model using the new parameters from the server.
        """
        print(f"[{self.client_id}] fit")
        
        self.model.set_weights(parameters)

        local_epochs = config_server.get("local_epochs", config.DEFAULT_LOCAL_EPOCHS)
        
        dp_batch_size = config.DP_MICROBATCHES
        
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=config.DP_L2_NORM_CLIP,
            noise_multiplier=config.DP_NOISE_MULTIPLIER,
            num_microbatches=dp_batch_size,
            learning_rate=config.DP_LEARNING_RATE
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            ),
            metrics=["accuracy"]
        )

        # 1. Create an instance of our new minimal callback
        progress_callback = MinimalProgressCallback(
            client_id=self.client_id, 
            epochs=local_epochs
        )

        # 2. Train the model
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=local_epochs,
            batch_size=dp_batch_size, 
            validation_data=(self.x_test, self.y_test),
            # 3. Set verbose=0 (silent) and pass our custom callback
            verbose=0,
            callbacks=[progress_callback] 
        )

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the provided parameters (from the global model) on
        the local test set.
        """
        print(f"[{self.client_id}] evaluate")
        
        self.model.set_weights(parameters)
        
        self.model.compile(
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        loss, accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=0
        )
        
        # Added .4f formatting for a cleaner log
        print(f"[{self.client_id}] Evaluate Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return loss, len(self.x_test), {"accuracy": accuracy}