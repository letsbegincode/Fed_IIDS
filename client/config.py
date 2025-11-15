# ---------------------------------
# 1. NETWORK CONFIGURATION
# ---------------------------------
# This is the IP address and port of your "Server Person's" machine.
# Ask them what this is.
SERVER_ADDRESS = "127.0.0.1:8080" 

DP_MICROBATCHES = 256
DP_L2_NORM_CLIP = 1.0
DP_NOISE_MULTIPLIER = 0.5
DP_LEARNING_RATE = 0.001

# === Training Defaults ===
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_BATCH_SIZE = 256 

# === Model Parameters ===
NUM_FEATURES = 30