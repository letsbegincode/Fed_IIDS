"""
NIDS Federated Learning Server

This server orchestrates the federated learning process.
It imports all its settings from 'server_config.py' and 'shared/'.
IT IS COMPLETELY INDEPENDENT OF THE /client FOLDER.
"""

# --- 1. Silence TensorFlow Warnings ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO and WARNING
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings("ignore", ".*The name tf.losses.sparse_softmax_cross_entropy is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.executing_e_agerly_outside_functions is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.ragged.RaggedTensorValue is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.engine.base_layer_utils is deprecated.*")
# --- End Warning Silence ---

import flwr as fl
import numpy as np
import sys
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple

# --- Path Modification to Import 'shared' ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


# Add the root directory to the Python path to allow 'shared' import
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# --- End Path Modification ---

# --- 2. Import Our Configurations ---
import server_config as sc      # From local ./server/ folder
from shared import model        # From ../shared/ folder
from shared import model_config # From ../shared/ folder

# --- 3. Global Test Set Loading ---
X_test_global = None
y_test_global = None

def load_global_test_set():
    """Loads the global test set from this server's internal data/ folder."""
    global X_test_global, y_test_global
    
    print("  > Loading global test set...", end="", flush=True)
    # --- [NEW] Path is now local to the server ---
    test_file = os.path.join(SCRIPT_DIR, "data", "global_test_set.npz")
    
    if not os.path.exists(test_file):
        print(" FAILED")
        print("-----------------------------------------------------------------")
        print(f"FATAL: 'global_test_set.npz' not found at {test_file}")
        print("The server needs this file to perform unbiased evaluation.")
        print("Please move 'global_test_set.npz' to 'Fed_IIDS/server/data/'")
        print("-----------------------------------------------------------------")
        return False
    
    with np.load(test_file) as data:
        X_test_global = data['X']
        y_test_global = data['y']
        
    print(f" DONE ({len(y_test_global)} samples).")
    return True

# --- 4. Server-Side Evaluation Function ---
def get_server_evaluation_fn():
    """
    Returns a function that the server uses to evaluate the
    global model on the held-out global test set.
    """
    if X_test_global is None or y_test_global is None:
        return None

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        """Standard evaluation function for the server."""
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        
        server_model = model.create_model() 
        server_model.set_weights(parameters)
        
        server_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        loss, accuracy = server_model.evaluate(X_test_global, y_test_global, verbose=0)
        
        y_pred_probs = server_model.predict(X_test_global, verbose=0)
        y_pred_binary = (y_pred_probs > 0.5).astype(int)
        f1 = f1_score(y_test_global, y_pred_binary)
        
        print("\n==================== [Global Model Evaluation] ====================")
        print(f"ROUND {server_round}")
        print(f"  Global Loss:     {loss:.4f}")
        print(f"  Global Accuracy: {accuracy:.4f}")
        print(f"  Global F1-Score: {f1:.4f}")
        print("=================================================================\n")
        
        return loss, {"accuracy": accuracy, "f1_score": f1}

    return evaluate

# --- 5. Client Configuration Function ---
def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """
    Return training configuration dictionary for each round.
    This is sent to *each* client selected for training.
    """
    config_dict = {
        "local_epochs": sc.LOCAL_EPOCHS_PER_ROUND, # From server_config.py
    }
    print(f"\n--- [Round {server_round} | Fit] ---")
    print(f"Sending config to clients: {config_dict}")
    return config_dict

# --- 6. Client-Side Metric Aggregation ---
def aggregate_evaluate_metrics(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    """
    Logs and aggregates the *client-side* evaluation metrics.
    """
    if not metrics:
        return {}
        
    total_samples = sum([num_examples for num_examples, _ in metrics])
    
    weighted_avg_acc = sum(
        [num_examples * m["accuracy"] for num_examples, m in metrics]
    ) / total_samples
    
    print(f"--- [Client-Side Evaluate Results] ---")
    print(f"Received local evaluations from {len(metrics)} clients.")
    print(f"  Weighted Avg. *Client-Side* Accuracy: {weighted_avg_acc:.4f}")
    
    return {"accuracy": weighted_avg_acc}


# --- 7. Main Server Execution ---
def main():
    """Starts the Federated Learning server."""
    print("\n=================================================================")
    print("           NIDS FEDERATED LEARNING SERVER - STARTING")
    print("=================================================================")
    
    if not load_global_test_set():
        return

    print("  > Configuring FedAvg strategy...")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           
        fraction_evaluate=1.0,      
        
        min_fit_clients=sc.MIN_CLIENTS,        
        min_evaluate_clients=sc.MIN_CLIENTS,   
        min_available_clients=sc.MIN_CLIENTS,  
        
        evaluate_fn=get_server_evaluation_fn(), 
        on_fit_config_fn=fit_config,            
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics, 
    )
    print("  > Strategy configured.")

    print("\n-----------------------------------------------------------------")
    print(f"Server configured for {sc.NUM_ROUNDS} rounds.")
    print(f"Waiting for {sc.MIN_CLIENTS} clients to connect...")
    print("-----------------------------------------------------------------")

    fl.server.start_server(
        server_address=sc.SERVER_ADDRESS, # From server_config.py
        config=fl.server.ServerConfig(num_rounds=sc.NUM_ROUNDS), # From server_config.py
        strategy=strategy
    )
    
    print("=================================================================")
    print("           FEDERATED LEARNING COMPLETE - SERVER SHUTDOWN")
    print("=================================================================")

if __name__ == "__main__":
    main()