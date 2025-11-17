"""
Configuration file for the NIDS Federated Learning Server.
All tweakable parameters are stored here.
"""

# --- Server Network Config ---
# "0.0.0.0" means listen on all available network interfaces.
# "8080" is the port.
SERVER_ADDRESS = "0.0.0.0:8080"


# --- Federated Learning Config ---
# Number of rounds to run
NUM_ROUNDS = 1

# Number of clients to wait for before starting
# This MUST match the number of clients you intend to run (e.g., hospital + factory = 2)
MIN_CLIENTS = 2

# Timeout (in seconds) for clients to complete training per round
# 7200 seconds = 2 hours
FIT_TIMEOUT = 7200


# --- Client Task Config ---
# Configuration to send to clients each round
# (e.g., how many epochs to train locally)
LOCAL_EPOCHS_PER_ROUND = 1