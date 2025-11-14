# Filepath: FED_IIDS/client/data_loader.py

import numpy as np
import os
import config # Import config for DP-SGD settings

def load_data(client_id: str):
    """
    Loads the preprocessed, partitioned data for a specific client
    from the .npz files inside the 'data/' folder.
    """
    data_dir = "data" # Assumes a 'data' folder in the same directory
    train_file = os.path.join(data_dir, f"client_{client_id}_train.npz")
    test_file = os.path.join(data_dir, f"client_{client_id}_test.npz")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Error: Data files not found for client {client_id}")
        print(f"Looked for: {train_file} and {test_file}")
        print("Please make sure the correct .npz files are in the 'data/' folder.")
        # Return dummy data to avoid a crash during testing
        dummy_x = np.random.rand(10, config.NUM_FEATURES) 
        dummy_y = np.random.randint(0, 2, 10)
        return (dummy_x, dummy_y), (dummy_x, dummy_y)

    with np.load(train_file) as train_data:
        X_train = train_data['X']
        y_train = train_data['y']
        
    with np.load(test_file) as test_data:
        X_test = test_data['X']
        y_test = test_data['y']
        
    print(f"[Data] Client {client_id}: Loaded {len(y_train)} train samples and {len(y_test)} test samples.")
    
    # --- Data Trimming for DP-SGD ---
    # DP-SGD requires the batch size to be static and not drop the remainder.
    n_train = len(y_train)
    n_train_rounded = (n_train // config.DEFAULT_BATCH_SIZE) * config.DEFAULT_BATCH_SIZE
    if n_train != n_train_rounded:
        print(f"[Data] Trimming training data from {n_train} to {n_train_rounded} samples for DP-SGD.")
        X_train = X_train[:n_train_rounded]
        y_train = y_train[:n_train_rounded]

    return (X_train, y_train), (X_test, y_test)