"""
Configuration file for the NIDS client.
Imports the shared model contract and defines client-specific settings
like the server IP and DP parameters.
"""

# --- 1. Import Shared Model Config ---
# We get the model shape from the shared "API Contract"
from shared.model_config import NUM_FEATURES

# --- 2. Network Configuration ---
# This is the IP address of the server.
SERVER_ADDRESS = "127.0.0.1:8080" # Use "127.0.0.1" for local testing

# --- 3. Differential Privacy Config ---
DP_L2_NORM_CLIP = 1.0
DP_NOISE_MULTIPLIER = 0.5
DP_LEARNING_RATE = 0.001
DP_MICROBATCHES = 256 # This MUST match DEFAULT_BATCH_SIZE

# --- 4. Training Defaults ---
DEFAULT_LOCAL_EPOCHS = 5
DEFAULT_BATCH_SIZE = 256 # Must match DP_MICROBATCHES