import numpy as np
import os
import sys

# Ensure we can import data_loader
if 'client' not in os.getcwd():
    sys.path.append(os.path.join(os.path.dirname(__file__)))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
try:
    import data_loader
except ImportError:
    print("ERROR: Cannot find 'data_loader.py'. Make sure it's in the 'client/' folder.")
    sys.exit(1)

print("--- Checking Data Structure ---")
try:
    x_train, y_train, x_test, y_test = data_loader.load_data("hospital")
    
    print("\n[SUCCESS] Data loaded.")
    print(f"  Train X shape: {x_train.shape}")
    print(f"  Train y shape: {y_train.shape}")
    print(f"  Test X shape:  {x_test.shape}")
    print(f"  Test y shape:  {y_test.shape}")
    
    print(f"\n[CRITICAL] Number of Features: {x_train.shape[1]}")
    
except Exception as e:
    print(f"\n[ERROR] Could not load data using 'data_loader.py': {e}")
    print("Please check the 'data_loader.py' script and 'client/data/' folder.")

print("-------------------------------")