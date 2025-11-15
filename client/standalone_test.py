import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import numpy as np
import sys
import os

# --- Setup Paths ---
if 'client' not in os.getcwd():
    sys.path.append(os.path.join(os.path.dirname(__file__)))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import data_loader
except ImportError as e:
    print(f"ERROR: Could not import 'data_loader.py'. {e}")
    sys.exit(1)

print("--- Starting STANDALONE Local DP-SGD Test ---")
print("This test has NO dependencies on config.py or model.py")

# --- 1. Hard-coded Config ---
NUM_FEATURES = 30
DP_BATCH_SIZE = 256       # This is the fix. We use 256 for BOTH.
DP_L2_NORM_CLIP = 1.0
DP_NOISE_MULTIPLIER = 0.5
DP_LEARNING_RATE = 0.001
LOCAL_EPOCHS = 2
CLIENT_ID = "hospital"

print("\n--- Hard-coded Configuration ---")
print(f"  NUM_FEATURES = {NUM_FEATURES}")
print(f"  DP_BATCH_SIZE (for fit AND optimizer) = {DP_BATCH_SIZE}")
print("---------------------------------")

# --- 2. Load Data ---
print(f"Loading data for client: {CLIENT_ID}...")
try:
    # We load both train and test data now
    x_train, y_train, x_test, y_test = data_loader.load_data(CLIENT_ID)
    print(f"Loaded {len(x_train)} training samples and {len(x_test)} test samples.")
except Exception as e:
    print(f"ERROR: Could not load data. {e}")
    sys.exit(1)

# --- 3. Sanity Check ---
print(f"Running sanity check on data shape...")
if x_train.shape[1] != NUM_FEATURES:
    print(f"\n--- FATAL ERROR ---")
    print(f"Data has {x_train.shape[1]} features, but we expected {NUM_FEATURES}.")
    sys.exit(1)
print(f"  Train data has {x_train.shape[0]} samples.")
if x_train.shape[0] % DP_BATCH_SIZE != 0:
    print(f"\n--- FATAL ERROR ---")
    print(f"Training data size ({x_train.shape[0]}) is not divisible by DP_BATCH_SIZE ({DP_BATCH_SIZE}).")
    print("This is required by tensorflow-privacy.")
    print(f"Check your 'data_loader.py' trimming logic.")
    sys.exit(1)
print("  Data features and batch size check passed.")


# --- 4. Hard-coded Model Creation ---
def create_standalone_model():
    """Creates the Keras model, using our hard-coded NUM_FEATURES."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_FEATURES,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

print("Creating Keras model...")
local_model = create_standalone_model()
local_model.summary()

# --- 5. Create the DP Optimizer ---
print("Setting up DP-SGD Optimizer...")
optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=DP_L2_NORM_CLIP,
    noise_multiplier=DP_NOISE_MULTIPLIER,
    num_microbatches=DP_BATCH_SIZE, # 256
    learning_rate=DP_LEARNING_RATE
)

# --- 6. Compile the model ---
print("Compiling model with DP Optimizer and Vector Loss...")
local_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    ),
    metrics=["accuracy"]
)

# --- 7. Run Local Training ---
print("\n--- Calling model.fit() ---")
print("Setting verbose=1 to show the progress bar.")
try:
    local_model.fit(
        x_train,
        y_train,
        epochs=LOCAL_EPOCHS,
        batch_size=DP_BATCH_SIZE, # 256
        # --- THIS IS THE FIX ---
        # We removed 'validation_split=0.1'
        # We now pass the pre-loaded test set.
        validation_data=(x_test, y_test),
        verbose=1 
    )
    print("\n--- ✅✅✅ SUCCESS! ✅✅✅ ---")
    print("Standalone local training test passed without errors.")
    
except Exception as e:
    print(f"\n--- TEST FAILED ---")
    print(f"Error: {e}")