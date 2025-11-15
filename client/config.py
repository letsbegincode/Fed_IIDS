# ---------------------------------
# 1. NETWORK CONFIGURATION
# ---------------------------------
# This is the IP address and port of your "Server Person's" machine.
# Ask them what this is.
SERVER_ADDRESS = "10.2.48.245:8080" 

# ---------------------------------
# 2. MODEL "API CONTRACT" CONFIG
# ---------------------------------
NUM_FEATURES = 30 

# ---------------------------------
# 3. PATH A: DIFFERENTIAL PRIVACY CONFIG
# ---------------------------------
# These are the "privacy budget" parameters you control.
DP_L2_NORM_CLIP = 1.0
DP_NOISE_MULTIPLIER = 0.7  # A higher number means more noise (more privacy)
DP_MICROBATCHES = 256
DP_LEARNING_RATE = 0.001

# ---------------------------------
# 4. TRAINING CONFIG
# ---------------------------------
# These are default values. The server might override them.
DEFAULT_LOCAL_EPOCHS = 5
DEFAULT_BATCH_SIZE = DP_MICROBATCHES # For DP-SGD, batch_size must equal microbatches