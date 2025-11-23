import robosuite
import numpy as np
import h5py

# Create a dummy env to check state shape
env = robosuite.make(
    "PickPlaceCan",
    robots=["Panda"],
    use_camera_obs=False,
    has_renderer=False,
)
env.reset()
state = env.sim.get_state().flatten()
print(f"Env state shape: {state.shape}")

# Check dataset state shape
with h5py.File('datasets/canMH_raw.hdf5', 'r') as f:
    demo_key = list(f['data'].keys())[0]
    ds_state = f['data'][demo_key]['states'][0]
    print(f"Dataset state shape: {ds_state.shape}")
