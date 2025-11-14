# Filepath: FED_IIDS/client/nids_client.py

import flwr as fl
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.keras import DPKerasAdamOptimizer
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Import your custom modules
import config
import model
from data_loader import load_data

class NidsClient(fl.client.NumPyClient):
    """
    This is your main Flower Client. It handles all client-side logic:
    - Loading local data
    - Implementing Differential Privacy (Path A)
    """
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = model.create_model()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_data(self.client_id)

    def get_parameters(self, config_server):
        """Flower API: Get model parameters."""
        print(f"[Client {self.client_id}] get_parameters")
        return self.model.get_weights()

    def fit(self, parameters, config_server):
        """Flower API: Train the model."""
        print(f"[Client {self.client_id}] fit (training)")
        
        # 1. Set model parameters from server
        self.model.set_weights(parameters)
        
        # 2. Get training config from server (or use default)
        local_epochs = config_server.get("local_epochs", config.DEFAULT_LOCAL_EPOCHS)
        
        # 3. --- PATH A: DP-SGD IMPLEMENTATION ---
        # We must re-compile the model *every time* for DP.
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=config.DP_L2_NORM_CLIP,
            noise_multiplier=config.DP_NOISE_MULTIPLIER,
            num_microbatches=config.DP_MICROBATCHES,
            learning_rate=config.DP_LEARNING_RATE
        )
        # DP-SGD requires a special loss function with Reduction.NONE
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE 
        )
        
        # 4. Compile and fit using the DP optimizer.
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=local_epochs,
            batch_size=config.DEFAULT_BATCH_SIZE, # This MUST be static
            verbose=2
        )
        
        print(f"[Client {self.client_id}] Training complete.")
        
        # 5. Return the new (private) parameters to the server
        return self.model.get_weights(), len(self.x_train), {"loss": history.history['loss'][-1]}

    def evaluate(self, parameters, config_server):
        """Flower API: Evaluate the model."""
        print(f"[Client {self.client_id}] evaluate (testing)")
        
        # 1. Set model parameters from server
        self.model.set_weights(parameters)
        
        # 2. Compile for standard evaluation (NO DP)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        # 3. Evaluate on local test set
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # 4. Calculate metrics manually for reliability
        y_pred_probs = self.model.predict(self.x_test, verbose=0)
        y_pred_binary = (y_pred_probs > 0.5).astype(int)
        f1 = f1_score(self.y_test, y_pred_binary)
        
        print(f"[Client {self.client_id}] Evaluate: Loss={loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
        
        # 4. Return results to server
        return float(loss), len(self.x_test), {"accuracy": float(acc), "f1_score": float(f1)}