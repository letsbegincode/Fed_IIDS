"""
NIDS Federated Learning Client Starter

This script:
1. Silences noisy TensorFlow/Keras warnings.
2. Parses the client ID ("hospital" or "factory").
3. Loads the correct, partitioned data.
4. Creates an instance of the NIDSClient.
5. Starts the client and connects to the server.
"""

# --- 1. Silence TensorFlow Warnings ---
# This must be at the top, before TensorFlow is imported
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress INFO and WARNING messages (show only ERROR+)
import tensorflow as tf
import logging

# Configure TensorFlow logger to hide deprecation/INFO messages that come
# through Python logging (these are separate from the C++ logs controlled by
# `TF_CPP_MIN_LOG_LEVEL`).
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('flwr').setLevel(logging.ERROR)
try:
    # For TF1-style logging calls inside TF2, set compat verbosity as well.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    pass
import warnings
# Suppress specific Keras/TF warnings
warnings.filterwarnings("ignore", ".*The name tf.losses.sparse_softmax_cross_entropy is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.train.SessionRunHook is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.logging.TaskLevelStatusMessage is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.control_flow_v2_enabled is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.executing_eagerly_outside_functions is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.ragged.RaggedTensorValue is deprecated.*")
import sys
# Suppress TensorFlow Probability warning
# Suppress TensorFlow Probability distutils deprecation
warnings.filterwarnings("ignore", ".*distutils Version classes are deprecated.*", category=DeprecationWarning, module="tensorflow_probability.*")
# As a fallback, ignore DeprecationWarning coming from installed packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Some TF internals emit deprecation warnings via the `warnings` module.
warnings.filterwarnings("ignore", ".*sparse_softmax_cross_entropy.*")
# --- End Warning Silence ---

import argparse
import flwr as fl
import time
import grpc
# If this script is executed from inside the `client/` directory (making
# `client` not importable as a top-level package), add the project root to
# `sys.path` so absolute imports like `from client import ...` still work.
if __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from client import config
from client import data_loader
from client import nids_client


def main():
    """
    Parses command-line arguments and starts the Flower client.
    """
    parser = argparse.ArgumentParser(description="Flower Client for NIDS")
    parser.add_argument(
        "--client-id",
        type=str,
        required=True,
        choices=["hospital", "factory"],
        help="Specify the Client ID ('hospital' or 'factory')."
    )
    args = parser.parse_args()
    client_id = args.client_id
    
    print(f"\n=================================================================")
    print(f"           STARTING CLIENT: {client_id.upper()}")
    print(f"=================================================================")

    # 2. Load the client-specific data
    try:
        print(f"  > Loading data...", end="", flush=True)
        x_train, y_train, x_test, y_test = data_loader.load_data(client_id)
        print(f" DONE ({len(x_train)} train, {len(x_test)} test samples).")
    
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load data: {e}")
        return

    # 3. Create an instance of the NIDSClient
    client = nids_client.NIDSClient(
        client_id=client_id,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    
    # 4. Start the Flower client
    print(f"  > Connecting to server at {config.SERVER_ADDRESS}...")

    # Retry loop with exponential backoff to handle transient connection
    # timeouts/drops (firewalls, load balancers, etc.).
    def start_client_with_retries(max_retries=None):
        attempt = 0
        delay = 5
        while True:
            attempt += 1
            try:
                fl.client.start_client(
                    server_address=config.SERVER_ADDRESS,
                    client=client.to_client(),
                )
                break
            except grpc.RpcError as rpc_err:
                # Print concise gRPC error code and details only
                try:
                    code = rpc_err.code()
                    details = rpc_err.details()
                    print(f"[Connection] gRPC error on attempt {attempt}: {code} - {details}")
                except Exception:
                    print(f"[Connection] gRPC error on attempt {attempt}: {rpc_err}")
            except Exception as e:
                print(f"[Connection] Error on attempt {attempt}: {e}")

            if (max_retries is not None) and (attempt >= max_retries):
                print(f"[Connection] Reached max retries ({max_retries}). Aborting.")
                raise

            print(f"[Connection] Retry in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * 2, 60)

    try:
        start_client_with_retries()
    except KeyboardInterrupt:
        print("\n[Connection] Interrupted by user. Exiting client.")
        sys.exit(0)
    print(f"\n[{client_id}] Client disconnected.")
    print("=================================================================")

if __name__ == "__main__":
    main()