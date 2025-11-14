# Filepath: FED_IIDS/client/run_client.py

import flwr as fl
import argparse
import sys
import os

# This makes sure Python can find your files (config, model, etc.)
# It adds the 'client' folder to the Python path.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    import config
    from nids_client import NidsClient
except ImportError:
    print("Error: Could not import custom modules. Make sure you are in the 'client' directory.")
    sys.exit(1)


def main():
    """Parses arguments and starts the Flower client."""
    
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Flower NIDS Client")
    parser.add_argument(
        "--client-id",
        required=True,
        type=str,
        help="The partition/ID for this client (e.g., 'hospital', 'factory'). " \
             "This must match a data file in the /data folder."
    )
    args = parser.parse_args()
    
    print(f"Starting client with ID: {args.client_id}")
    
    # 2. Instantiate the client
    try:
        client = NidsClient(args.client_id)
    except Exception as e:
        print(f"Error instantiating client: {e}")
        sys.exit(1)
        
    # 3. Start the client
    # It will connect to the server specified in config.py
    try:
        fl.client.start_client(
            server_address=config.SERVER_ADDRESS,
            client=client
        )
    except Exception as e:
        print(f"Error connecting to server at {config.SERVER_ADDRESS}: {e}")
        print("Please ensure the server is running and the IP in config.py is correct.")
        sys.exit(1)

if __name__ == "__main__":
    main()