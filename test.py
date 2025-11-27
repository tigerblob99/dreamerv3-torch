import h5py
import json

with h5py.File("datasets/canMH_raw.hdf5", "r") as f:
    env_args = json.loads(f["data"].attrs["env_args"])
    print(json.dumps(env_args, indent=2))