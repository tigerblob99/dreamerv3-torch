"""
Compare first-frame images from a converted NPZ demo against freshly rendered
environment observations initialized to the same joint state.

Usage:
    python test.py --npz_path /path/to/demo_0-1001.npz \
                   --task PickPlaceCan \
                   --camera_keys agentview_image robot0_eye_in_hand_image \
                   --flip_keys agentview_image robot0_eye_in_hand_image

Notes:
- The NPZ produced by convert_robomimic_to_dreamer.py does not store simulator
  states. This script approximates the initial state by setting robot joint
  positions/velocities (and gripper) from the NPZ low-dim keys if available.
  If those keys are missing or the env API differs, the script falls back to
  a plain reset.
- The script compares only the image tensor stacking along the channel axis.
  Ensure camera_keys order matches the converter order.
"""

import argparse
import os
import pathlib
from typing import Dict, Iterable, List

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import robosuite  # type: ignore
from envs.robosuite_env import _resolve_controller_loader, _ensure_composite_controller_config


def _stack_env_cameras(obs: Dict[str, np.ndarray], camera_keys: Iterable[str], flip_keys: Iterable[str]) -> np.ndarray:
    """Stack camera observations in the specified order, applying vertical flips where requested."""
    frames: List[np.ndarray] = []
    flip_set = set(flip_keys)
    for key in camera_keys:
        if key not in obs:
            raise KeyError(f"Camera observation '{key}' missing from environment output")
        frame = np.asarray(obs[key])
        if key in flip_set:
            frame = np.flip(frame, axis=0)  # vertical flip
        frames.append(frame)
    return np.concatenate(frames, axis=-1)


def _set_robot_state_from_npz(env, npz_obs: Dict[str, np.ndarray]) -> None:
    """Best-effort: set robot joint/gripper positions (and optionally velocities) from NPZ low-dim keys."""
    robot = env.robots[0]
    qpos = env.sim.data.qpos.copy()
    qvel = env.sim.data.qvel.copy()

    pos_key = "robot0_joint_pos"
    vel_key = "robot0_joint_vel"
    grip_pos_key = "robot0_gripper_qpos"
    grip_vel_key = "robot0_gripper_qvel"

    if hasattr(robot, "_ref_joint_pos_indexes") and pos_key in npz_obs:
        idxs = robot._ref_joint_pos_indexes  # type: ignore[attr-defined]
        if len(idxs) == npz_obs[pos_key].shape[-1]:
            qpos[idxs] = npz_obs[pos_key]
    if hasattr(robot, "_ref_joint_vel_indexes") and vel_key in npz_obs:
        idxs = robot._ref_joint_vel_indexes  # type: ignore[attr-defined]
        if len(idxs) == npz_obs[vel_key].shape[-1]:
            qvel[idxs] = npz_obs[vel_key]

    if hasattr(robot, "_ref_gripper_joint_pos_indexes") and grip_pos_key in npz_obs:
        idxs = robot._ref_gripper_joint_pos_indexes  # type: ignore[attr-defined]
        if len(idxs) == npz_obs[grip_pos_key].shape[-1]:
            qpos[idxs] = npz_obs[grip_pos_key]
    if hasattr(robot, "_ref_gripper_joint_vel_indexes") and grip_vel_key in npz_obs:
        idxs = robot._ref_gripper_joint_vel_indexes  # type: ignore[attr-defined]
        if len(idxs) == npz_obs[grip_vel_key].shape[-1]:
            qvel[idxs] = npz_obs[grip_vel_key]

    env.sim.data.qpos[:] = qpos
    env.sim.data.qvel[:] = qvel
    env.sim.forward()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=pathlib.Path, required=True, help="Path to a single NPZ demo file")
    parser.add_argument("--task", type=str, default="PickPlaceCan")
    parser.add_argument("--robots", nargs="+", default=["Panda"])
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--camera_keys", nargs="+", default=["agentview_image", "robot0_eye_in_hand_image"])
    parser.add_argument("--flip_keys", nargs="+", default=["agentview_image", "robot0_eye_in_hand_image"])
    parser.add_argument("--image_hw", type=int, nargs=2, default=[84, 84], help="Height width for cameras")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering (offscreen always on)")
    parser.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("imgs/env_vs_npz"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = args.npz_path.expanduser()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    ep = dict(np.load(npz_path))
    if "image" not in ep:
        raise KeyError("NPZ is missing 'image' key; cannot compare images.")

    # Pull first-frame low-dim for setting state
    npz_obs0 = {k: ep[k][0] for k in ep.keys() if not k.startswith("action") and not k.startswith("reward")}

    # Build env
    load_controller_config = _resolve_controller_loader(robosuite)
    controller_cfg = load_controller_config(default_controller=args.controller)
    controller_cfg = _ensure_composite_controller_config(controller_cfg, tuple(args.robots))

    env = robosuite.make(
        env_name=args.task,
        robots=args.robots,
        controller_configs=controller_cfg,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=[k.replace("_image", "") for k in args.camera_keys],
        camera_heights=args.image_hw[0],
        camera_widths=args.image_hw[1],
        has_renderer=args.render,
        has_offscreen_renderer=True,
        reward_shaping=False,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
    )

    env.reset()
    try:
        _set_robot_state_from_npz(env, npz_obs0)
        print("Applied NPZ joint/gripper state to env.")
    except Exception as exc:
        print(f"[warn] Could not apply NPZ state to env: {exc}")

    env_obs = env._get_observations(force_update=True)
    env_img = _stack_env_cameras(env_obs, args.camera_keys, args.flip_keys)

    npz_img = ep["image"][0]  # (H,W,3*K)

    if env_img.shape != npz_img.shape:
        print(f"[shape mismatch] env {env_img.shape} vs npz {npz_img.shape}")
    else:
        diff = env_img.astype(np.float32) - npz_img.astype(np.float32)
        print(
            f"Image diff stats | max: {np.abs(diff).max():.2f}, mean: {np.abs(diff).mean():.2f}, rmse: {np.sqrt((diff**2).mean()):.2f}"
        )

    # Save side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Tile multi-camera channels along width for visualization if needed
    def _tile(img: np.ndarray) -> np.ndarray:
        if img.shape[-1] % 3 != 0:
            return img
        cams = img.shape[-1] // 3
        slices = [img[..., i * 3 : (i + 1) * 3] for i in range(cams)]
        return np.concatenate(slices, axis=1)

    npz_vis = _tile(npz_img.astype(np.uint8))
    env_vis = _tile(env_img.astype(np.uint8))

    axes[0].imshow(npz_vis)
    axes[0].set_title("NPZ image (t=0)")
    axes[1].imshow(env_vis)
    axes[1].set_title("Env image (t=0)")
    if env_img.shape == npz_img.shape:
        diff_vis = np.clip(np.abs(env_img.astype(np.int16) - npz_img.astype(np.int16)), 0, 255).astype(np.uint8)
        axes[2].imshow(_tile(diff_vis))
        axes[2].set_title("Abs diff")
    else:
        axes[2].axis("off")
        axes[2].set_title("Shapes differ")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    out_path = out_dir / "env_vs_npz_image.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved comparison to {out_path}")

    env.close()


if __name__ == "__main__":
    main()
