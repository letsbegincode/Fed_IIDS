# FED_IIDS/client/run_client.py
#
# === FINAL CORRECTED VERSION ===
# This version fixes the 'AttributeError' by calling the
# correct function name: 'load_data' instead of 'load_client_data'.

import argparse
import flwr as fl
import config
import data_loader  # Imports data_loader.py
import nids_client  # Imports nids_client.py

def main():
    """
    Parses command-line arguments and starts the Flower client.
    """
    # 1. Parse command-line arguments
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
    
    print(f"Starting client with ID: {client_id}")

    # 2. Load the client-specific data
    try:
        # --- THIS IS THE FIX ---
        # Changed 'data_loader.load_client_data' to 'data_loader.load_data'
        x_train, y_train, x_test, y_test = data_loader.load_data(client_id)
        # ---------------------
        
        print(f"  Loaded {len(x_train)} training samples and {len(x_test)} test samples.")
    
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
    print("  Connecting to server...")
    fl.client.start_client(
        server_address=config.SERVER_ADDRESS,
        client=client
    )
    print("Client disconnected.")

if __name__ == "__main__":
    main()