import h5py
import numpy as np

file_path = 'datasets/canMH_raw.hdf5'

def print_keys(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
        if 'xml' in name:
             print(f"  Content (first 100 chars): {obj[0][:100]}")

try:
    with h5py.File(file_path, 'r') as f:
        print(f"Keys in {file_path}:")
        # Print top level keys
        print(list(f.keys()))
        
        # Check 'data' group if it exists (common in robomimic)
        if 'data' in f:
            print("\nKeys in 'data':")
            print(list(f['data'].keys())[:5]) # Print first 5 demos
            
            first_demo = list(f['data'].keys())[0]
            print(f"\nKeys in 'data/{first_demo}':")
            print(list(f['data'][first_demo].keys()))
            
            # Check for initial state info
            if 'states' in f['data'][first_demo]:
                print(f"\nInitial state shape: {f['data'][first_demo]['states'][0].shape}")
            
            # Check for environment metadata
            if 'env_args' in f['data']:
                 print(f"\nEnv args: {f['data']['env_args']}")

        # Check for global metadata
        if 'env_args' in f:
             print(f"\nGlobal Env args: {f['env_args']}")
             
except Exception as e:
    print(f"Error reading file: {e}")
