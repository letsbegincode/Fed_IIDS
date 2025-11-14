import flwr as fl
import tensorflow as tf
import numpy as np
import os
import sys
from sklearn.metrics import f1_score

# --- Path Setup ---
# This is tricky. The server needs to import 'model' and 'config'
# from the 'client' subfolder.
CLIENT_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'client')
if CLIENT_FOLDER_PATH not in sys.path:
    sys.path.append(CLIENT_FOLDER_PATH)

try:
    import model # The shared model.py
    import config # To get NUM_FEATURES
except ImportError:
    print("Error: Could not import 'model' or 'config' from 'client' folder.")
    print(f"Please make sure '{CLIENT_FOLDER_PATH}' exists and contains model.py and config.py")
    sys.exit(1)

# --- Global Test Set Loading ---
# This is the UNBIASED test set that no client has seen.
X_test_global = None
y_test_global = None

def load_global_test_set():
    """Loads the global test set from the client/data/ folder."""
    global X_test_global, y_test_global
    
    # The server looks *inside* the client/data folder for the test set
    test_file = os.path.join(CLIENT_FOLDER_PATH, "data", "global_test_set.npz")
    
    if not os.path.exists(test_file):
        print("-----------------------------------------------------------------")
        print(f"FATAL: 'global_test_set.npz' not found at {test_file}")
        print("The server needs this file to perform unbiased evaluation.")
        print("Please download it from Google Drive and place it in the 'client/data' folder.")
        print("-----------------------------------------------------------------")
        return False
    
    with np.load(test_file) as data:
        X_test_global = data['X']
        y_test_global = data['y']
        
    print(f"Loaded global test set: {len(y_test_global)} samples.")
    return True

def get_server_evaluation_fn():
    """
    Returns a function that will be used by the server to
    evaluate the *global* model on the *global* test set.
    """
    
    if X_test_global is None or y_test_global is None:
        print("Error: Global test set not loaded. Cannot create evaluation function.")
        return None

    def evaluate(server_round: int, parameters: fl.common.Parameters, config):
        """Standard evaluation function for the server."""
        
        # 1. Create a new instance of the model
        server_model = model.create_model()
        
        # 2. Set the weights to the new global parameters
        server_model.set_weights(fl.common.parameters_to_weights(parameters))
        
        # 3. Compile the model
        server_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        # 4. Evaluate on the global test set
        loss, accuracy = server_model.evaluate(X_test_global, y_test_global, verbose=0)
        
        # 5. Get F1 Score
        y_pred_probs = server_model.predict(X_test_global, verbose=0)
        y_pred_binary = (y_pred_probs > 0.5).astype(int)
        f1 = f1_score(y_test_global, y_pred_binary)
        
        print(f"\n--- Global Model Eval (Round {server_round}) ---")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}\n")
        
        return loss, {"accuracy": accuracy, "f1_score": f1}

    return evaluate

def main():
    """Starts the Federated Learning server."""
    print("Starting server...")
    
    # 1. Load the global test set
    if not load_global_test_set():
        return # Stop if data is missing

    # 2. Define the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # Train on all available clients (2)
        fraction_evaluate=1.0,      # Test on all available clients
        min_fit_clients=2,          # Wait for 2 clients to be ready
        min_available_clients=2,    # Must have 2 clients online
        evaluate_fn=get_server_evaluation_fn(), # Our global test function
    )

    # 3. Start the server
    # It will listen on all network interfaces on port 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10), # Run for 10 rounds
        strategy=strategy
    )

if __name__ == "__main__":
    main()