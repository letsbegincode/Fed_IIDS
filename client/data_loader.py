import numpy as np
import os
import config

# Base path where client data files are stored
BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

def load_data(client_id: str):
    """
    Loads the pre-partitioned data for a specific client.
    
    Args:
        client_id (str): The ID of the client (e.g., 'hospital').
        
    Returns:
        A tuple of four numpy arrays: (x_train, y_train, x_test, y_test)
    """
    
    # 1. Define file paths
    train_file = os.path.join(BASE_DATA_PATH, f"client_{client_id}_train.npz")
    test_file = os.path.join(BASE_DATA_PATH, f"client_{client_id}_test.npz")

    # 2. Check if files exist (Raise an error if missing)
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"Data files not found for client {client_id} in "
            f"{BASE_DATA_PATH}. Please run the Colab preprocessing."
        )

    # 3. Load the data
    try:
        train_data = np.load(train_file, allow_pickle=True)
        x_train = train_data['X']
        y_train = train_data['y']
        
        test_data = np.load(test_file, allow_pickle=True)
        x_test = test_data['X']
        y_test = test_data['y']
    except Exception as e:
        raise IOError(f"Error loading .npz files for {client_id}: {e}")
        
    print(f"[Data] Client {client_id}: Loaded {len(x_train)} train samples and {len(x_test)} test samples.")

    # 4. CRITICAL: Trim training data for DP-SGD
    # The DPKerasAdamOptimizer requires that the number of samples
    # is an exact multiple of the number of microbatches.
    
    num_microbatches = config.DP_MICROBATCHES
    
    # Calculate how many full batches we can make
    num_complete_batches = (len(x_train) // num_microbatches) * num_microbatches
    
    if num_complete_batches == 0:
        raise ValueError(
            f"Not enough training samples ({len(x_train)}) to form even "
            f"one microbatch ({num_microbatches}). Please check config.py."
        )
        
    if num_complete_batches < len(x_train):
        print(f"[Data] Trimming training data from {len(x_train)} to {num_complete_batches} samples for DP-SGD.")
        x_train = x_train[:num_complete_batches]
        y_train = y_train[:num_complete_batches]


    # This returns FOUR separate items, which matches run_client.py
    return x_train, y_train, x_test, y_test
    # ---------------------