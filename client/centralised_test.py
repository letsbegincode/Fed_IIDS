# --- 1. Silence TensorFlow Warnings ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO and WARNING
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings("ignore", ".*The name tf.losses.sparse_softmax_cross_entropy is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.train.SessionRunHook is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.logging.TaskLevelStatusMessage is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.control_flow_v2_enabled is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.executing_eagerly_outside_functions is deprecated.*")
warnings.filterwarnings("ignore", ".*The name tf.ragged.RaggedTensorValue is deprecated.*")
warnings.filterwarnings("ignore", ".*distutils Version classes are deprecated.*")
# --- End Warning Silence ---

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import numpy as np
import sys
import logging

# --- Setup Paths ---
# If this script is executed directly from inside `client/` (so the `client`
# package is not on `sys.path`), add the project root so absolute imports
# (`from client import ...`) work. This mirrors the approach used in
# `run_client.py` and lets users run either from project root or inside
# the `client/` folder.
if __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from client import data_loader

# Also tighten TensorFlow python-logged warnings (sometimes emitted despite
# the `TF_CPP_MIN_LOG_LEVEL` setting)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

print("--- Starting CENTRALIZED DP-SGD Test ---")
print("This test simulates a traditional, centralized model.")
print("It has NO dependencies on config.py or model.py")

# --- 1. Hard-coded Config ---
# These are identical to the federated run for a fair comparison
NUM_FEATURES = 30
DP_BATCH_SIZE = 256
DP_L2_NORM_CLIP = 1.0
DP_NOISE_MULTIPLIER = 0.5
DP_LEARNING_RATE = 0.001
LOCAL_EPOCHS = 2 # We use the same number of epochs as one FL round

print("\n--- Hard-coded Configuration ---")
print(f"  NUM_FEATURES = {NUM_FEATURES}")
print(f"  DP_BATCH_SIZE (for fit AND optimizer) = {DP_BATCH_SIZE}")
print("---------------------------------")

# --- 2. Load and Merge Data ---
print("Loading data for all clients...")
try:
    print("  > Loading 'hospital' data...")
    x_train_h, y_train_h, x_test_h, y_test_h = data_loader.load_data("hospital")
    print("  > Loading 'factory' data...")
    x_train_f, y_train_f, x_test_f, y_test_f = data_loader.load_data("factory")
    print("  > Data loaded.")

    print("\nMerging all datasets...")
    # Stack the training data
    x_train = np.vstack((x_train_h, x_train_f))
    y_train = np.hstack((y_train_h, y_train_f)) # labels are 1D

    # Stack the testing data
    x_test = np.vstack((x_test_h, x_test_f))
    y_test = np.hstack((y_test_h, y_test_f)) # labels are 1D
    
    print(f"  Total training samples: {len(x_train)}")
    print(f"  Total testing samples:  {len(x_test)}")

except Exception as e:
    print(f"ERROR: Could not load data. {e}")
    sys.exit(1)

# --- 3. Sanity Check ---
print(f"\nRunning sanity check on data shape...")
if x_train.shape[1] != NUM_FEATURES:
    print(f"\n--- FATAL ERROR ---")
    print(f"Data has {x_train.shape[1]} features, but we expected {NUM_FEATURES}.")
    sys.exit(1)
print(f"  Train data has {x_train.shape[0]} samples.")
if x_train.shape[0] % DP_BATCH_SIZE != 0:
    print(f"\n--- FATAL ERROR ---")
    print(f"Combined training data size ({x_train.shape[0]}) is not divisible by DP_BATCH_SIZE ({DP_BATCH_SIZE}).")
    print("This is required by tensorflow-privacy.")
    print(f"Check your 'data_loader.py' trimming logic.")
    sys.exit(1)
print("  Data features and batch size check passed.")


# --- 4. Hard-coded Model Creation ---
# This model is IDENTICAL to the one in shared/model.py,
# but hard-coded here for a standalone test.
def create_standalone_model():
    """Creates the Keras model, using our hard-coded NUM_FEATURES."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_FEATURES,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

print("\nCreating Keras model...")
local_model = create_standalone_model()
local_model.summary()

# --- 5. Create the DP Optimizer ---
print("\nSetting up DP-SGD Optimizer...")
optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=DP_L2_NORM_CLIP,
    noise_multiplier=DP_NOISE_MULTIPLIER,
    num_microbatches=DP_BATCH_SIZE, # 256
    learning_rate=DP_LEARNING_RATE
)

# --- 6. Compile the model ---
print("\nCompiling model with DP Optimizer and Vector Loss...")
local_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    ),
    metrics=["accuracy"]
)

# --- 7. Run Local Training ---
print("\n--- Calling model.fit() on COMBINED dataset ---")
print("Setting verbose=1 to show the progress bar...")
try:
    local_model.fit(
        x_train, # Merged training features
        y_train, # Merged training labels
        epochs=LOCAL_EPOCHS,
        batch_size=DP_BATCH_SIZE, # 256
        validation_data=(x_test, y_test), # Merged test data
        verbose=2
    )
    print("\n--- ✅✅✅ SUCCESS! ✅✅✅ ---")
    print("Centralized training test passed without errors.")
    
except Exception as e:
    print(f"\n--- TEST FAILED ---")
    print(f"Error: {e}")