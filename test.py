"""
Compare proprioception observations between NPZ files and environment.

This script loads a demo from an HDF5 dataset, replays it in the environment,
and compares the proprioceptive observations at each timestep to identify
any discrepancies between recorded and simulated observations.
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

from BC_MLP_eval import (
    _make_robomimic_can_env,
    _log_obs_snapshot,
    EvalConfig,
)


def compare_npz_vs_env_proprioception(
    npz_dir: pathlib.Path,
    hdf5_path: pathlib.Path,
    demo_key: str = "demo_0",
    output_dir: pathlib.Path = pathlib.Path("imgs/obs_comparison"),
):
    """
    Compare proprioception observations between NPZ file and environment replay.
    
    Args:
        npz_dir: Directory containing NPZ files with pre-processed observations
        hdf5_path: Path to HDF5 dataset with states for environment reset
        demo_key: Demo key to compare (e.g., "demo_0")
        output_dir: Directory to save comparison plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load NPZ episode
    npz_files = list(npz_dir.glob(f"{demo_key}-*.npz"))
    if not npz_files:
        npz_files = list(npz_dir.glob(f"{demo_key}*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ file found for {demo_key} in {npz_dir}")
    
    npz_file = npz_files[0]
    print(f"Loading NPZ from: {npz_file}")
    npz_data = dict(np.load(npz_file))
    
    # Load HDF5 for states
    print(f"Loading HDF5 from: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        if demo_key not in f["data"]:
            raise KeyError(f"Demo '{demo_key}' not found in HDF5")
        demo = f["data"][demo_key]
        states = np.array(demo["states"])
        actions = np.array(demo["actions"])
    
    # Create environment
    config = EvalConfig(
        policy_path=pathlib.Path("."),  # Dummy, not used
        checkpoint=pathlib.Path("."),   # Dummy, not used
        episodes=1,
        max_env_steps=500,
        camera_obs_keys=("agentview_image", "robot0_eye_in_hand_image"),
        mlp_obs_keys=("robot0_joint_pos", "robot0_joint_vel", "robot0_gripper_qpos", "robot0_gripper_qvel"),
        camera_image_key="image",
        clip_actions=False,
        render=False,
        device="cuda:0",
        seed=0,
        robosuite_task="PickPlaceCan",
        robosuite_robots=("Panda",),
        robosuite_controller="OSC_POSE",
        robosuite_reward_shaping=False,
        robosuite_control_freq=20,
        flip_camera_keys=("agentview_image", "robot0_eye_in_hand_image"),
        bc_hidden_units=1024,
        bc_hidden_layers=4,
        bc_activation="SiLU",
        bc_use_layernorm=False,
        video_dir=None,
        video_fps=20,
        video_camera=None,
        video_camera_keys=None,
        dataset_path=hdf5_path,
        dataset_obs_path=None,
        encoder_snapshot_path=None,
        replay_dataset_demo=False,
        replay_demo_key=None,
        replay_max_steps=None,
    )
    
    print("Creating environment...")
    env = _make_robomimic_can_env(config, (84, 84))
    
    # Proprioception keys to compare
    proprio_keys = [
        "robot0_joint_pos",
        "robot0_joint_vel", 
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
    ]
    
    # Also check trig keys if present in NPZ
    trig_keys = ["aux_robot0_joint_pos_sin", "aux_robot0_joint_pos_cos"]
    env_trig_keys = ["robot0_joint_pos_sin", "robot0_joint_pos_cos"]
    
    num_steps = min(len(actions), len(states) - 1)
    print(f"Comparing {num_steps} timesteps...")
    
    # Storage for comparison
    npz_obs = {k: [] for k in proprio_keys + trig_keys}
    env_obs = {k: [] for k in proprio_keys + env_trig_keys}
    
    # Reset to initial state
    env.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    
    # Collect observations at each step
    for t in range(num_steps + 1):
        # Get environment observation
        raw_obs = env._get_observations(force_update=True)
        
        for k in proprio_keys:
            if k in raw_obs:
                env_obs[k].append(np.array(raw_obs[k]))
        for k in env_trig_keys:
            if k in raw_obs:
                env_obs[k].append(np.array(raw_obs[k]))
        
        # Get NPZ observation for this timestep
        for k in proprio_keys:
            if k in npz_data and t < len(npz_data[k]):
                npz_obs[k].append(npz_data[k][t])
        for k in trig_keys:
            if k in npz_data and t < len(npz_data[k]):
                npz_obs[k].append(npz_data[k][t])
        
        # Step environment with recorded action (except last step)
        if t < num_steps:
            env.step(actions[t])
    
    env.close()
    
    # Convert to arrays
    for k in npz_obs:
        if npz_obs[k]:
            npz_obs[k] = np.array(npz_obs[k])
    for k in env_obs:
        if env_obs[k]:
            env_obs[k] = np.array(env_obs[k])
    
    # Plot comparisons
    print("\n=== Proprioception Comparison ===")
    
    for key in proprio_keys:
        if key not in npz_obs or len(npz_obs[key]) == 0:
            print(f"Skipping {key}: not in NPZ")
            continue
        if key not in env_obs or len(env_obs[key]) == 0:
            print(f"Skipping {key}: not in env obs")
            continue
        
        npz_arr = npz_obs[key]
        env_arr = env_obs[key]
        
        # Align lengths
        min_len = min(len(npz_arr), len(env_arr))
        npz_arr = npz_arr[:min_len]
        env_arr = env_arr[:min_len]
        
        diff = npz_arr - env_arr
        
        print(f"\n{key}:")
        print(f"  Shape: NPZ={npz_arr.shape}, Env={env_arr.shape}")
        print(f"  Diff - max: {np.abs(diff).max():.6f}, mean: {np.abs(diff).mean():.6f}, std: {diff.std():.6f}")
        
        # Plot
        dim = npz_arr.shape[1] if npz_arr.ndim > 1 else 1
        if npz_arr.ndim == 1:
            npz_arr = npz_arr[:, None]
            env_arr = env_arr[:, None]
            diff = diff[:, None]
        
        fig, axes = plt.subplots(dim, 3, figsize=(15, 2.5 * dim))
        if dim == 1:
            axes = axes[None, :]
        
        for i in range(dim):
            # NPZ vs Env overlay
            axes[i, 0].plot(npz_arr[:, i], label='NPZ', alpha=0.7)
            axes[i, 0].plot(env_arr[:, i], label='Env', alpha=0.7, linestyle='--')
            axes[i, 0].set_ylabel(f'Dim {i}')
            axes[i, 0].legend(loc='upper right', fontsize=8)
            if i == 0:
                axes[i, 0].set_title('NPZ vs Env')
            
            # Difference
            axes[i, 1].plot(diff[:, i], color='red', alpha=0.7)
            axes[i, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            if i == 0:
                axes[i, 1].set_title('Difference (NPZ - Env)')
            
            # Cumulative difference
            axes[i, 2].plot(np.cumsum(np.abs(diff[:, i])), color='purple', alpha=0.7)
            if i == 0:
                axes[i, 2].set_title('Cumulative |Diff|')
        
        axes[-1, 0].set_xlabel('Timestep')
        axes[-1, 1].set_xlabel('Timestep')
        axes[-1, 2].set_xlabel('Timestep')
        
        plt.suptitle(f'{key} - {demo_key}')
        plt.tight_layout()
        plt.savefig(output_dir / f"{demo_key}_{key}_comparison.png", dpi=150)
        plt.close(fig)
        print(f"  Saved plot to {output_dir / f'{demo_key}_{key}_comparison.png'}")
    
    # Compare trig keys
    for npz_key, env_key in zip(trig_keys, env_trig_keys):
        if npz_key not in npz_obs or len(npz_obs[npz_key]) == 0:
            continue
        if env_key not in env_obs or len(env_obs[env_key]) == 0:
            continue
        
        npz_arr = npz_obs[npz_key]
        env_arr = env_obs[env_key]
        
        min_len = min(len(npz_arr), len(env_arr))
        npz_arr = npz_arr[:min_len]
        env_arr = env_arr[:min_len]
        
        diff = npz_arr - env_arr
        
        print(f"\n{npz_key} (NPZ) vs {env_key} (Env):")
        print(f"  Shape: NPZ={npz_arr.shape}, Env={env_arr.shape}")
        print(f"  Diff - max: {np.abs(diff).max():.6f}, mean: {np.abs(diff).mean():.6f}")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    # Example usage - adjust paths as needed
    compare_npz_vs_env_proprioception(
        npz_dir=pathlib.Path("datasets/robomimic_data_MV/can_PH_train"),
        hdf5_path=pathlib.Path("datasets/canPH_raw.hdf5"),
        demo_key="demo_2",
        output_dir=pathlib.Path("imgs/obs_comparison"),
    )
