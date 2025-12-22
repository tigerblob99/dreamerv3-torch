"""Quick BC sweep evaluation script for the robosuite Lift task.

Loads a checkpoint produced by ``BC_Sweep.py`` (encoder + action MLP),
rebuilds the networks, and rolls them out in the real environment using
the same ``evaluate_in_environment`` helper used during the sweep.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from types import SimpleNamespace
from typing import Iterable, Tuple

import torch
import ruamel.yaml as yaml


# Ensure project root is importable when running from evals/
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("WANDB_MODE", "disabled")  # avoid accidental online logging for quick tests

from BC_Sweep import (  # noqa: E402
    DEFAULT_CAMERA_KEYS,
    DEFAULT_CNN_KEYS,
    DEFAULT_MLP_KEYS,
    evaluate_in_environment,
)
from bc_mlp.BC_MLP_train import (  # noqa: E402
    _build_encoder,
    _build_ordered_obs_space,
    build_action_mlp,
)
from offline_train import _infer_spaces  # noqa: E402
import tools  # noqa: E402


def _as_tuple(value: Iterable[str] | Tuple[str, ...] | None, fallback: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return fallback
    return tuple(value)


def _load_env_block(name: str) -> dict:
    """Load a config block from configs.yaml for env overrides."""
    root_configs = _ROOT / "configs.yaml"
    local_configs = pathlib.Path(__file__).with_name("configs.yaml")
    cfg_path = root_configs if root_configs.exists() else local_configs
    if not cfg_path.exists():
        raise FileNotFoundError(f"configs.yaml not found at {cfg_path}")
    parser = yaml.YAML(typ="safe")
    configs = parser.load(cfg_path.read_text())
    if name not in configs:
        raise KeyError(f"Config block '{name}' not found in {cfg_path}")
    block = configs[name] or {}
    if not isinstance(block, dict):
        raise TypeError(f"Config block '{name}' must be a mapping.")
    return block


def _load_policy_from_checkpoint(checkpoint_path: pathlib.Path, *, device: str | None, offline_traindir: str | None):
    """Rebuild encoder + action MLP and load weights from a BC sweep checkpoint."""
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = raw.get("config")
    if cfg_dict is None:
        raise KeyError("Checkpoint missing 'config' (expected vars(run_config) from BC_Sweep).")
    config = SimpleNamespace(**cfg_dict)

    if offline_traindir:
        config.offline_traindir = offline_traindir
    config.device = device or getattr(config, "device", "cpu")
    config.size = tuple(getattr(config, "size", (84, 84)))
    config.bc_crop_height = int(getattr(config, "bc_crop_height", 0) or 0)
    config.bc_crop_width = int(getattr(config, "bc_crop_width", 0) or 0)
    config.bc_cnn_keys_order = _as_tuple(getattr(config, "bc_cnn_keys_order", None), tuple(DEFAULT_CNN_KEYS))
    config.bc_mlp_keys_order = _as_tuple(getattr(config, "bc_mlp_keys_order", None), tuple(DEFAULT_MLP_KEYS))
    config.camera_obs_keys = _as_tuple(getattr(config, "camera_obs_keys", None), tuple(DEFAULT_CAMERA_KEYS))
    config.mlp_obs_keys = _as_tuple(getattr(config, "mlp_obs_keys", None), tuple(DEFAULT_MLP_KEYS))
    config.flip_camera_keys = _as_tuple(getattr(config, "flip_camera_keys", None), tuple(DEFAULT_CAMERA_KEYS))

    tools.set_seed_everywhere(getattr(config, "seed", 0))
    dataset = tools.load_episodes(config.offline_traindir, limit=1)
    if not dataset:
        raise RuntimeError(f"No episodes found in offline_traindir={config.offline_traindir} to infer shapes.")
    first_episode = next(iter(dataset.values()))
    obs_space = _build_ordered_obs_space(config, first_episode)
    _, act_space = _infer_spaces(first_episode)

    encoder = _build_encoder(config, obs_space)
    policy = build_action_mlp(
        encoder,
        act_space,
        hidden_units=getattr(config, "bc_hidden_units", 1024),
        hidden_layers=getattr(config, "bc_hidden_layers", 2),
        act_name=getattr(config, "bc_activation", "SiLU"),
        norm=getattr(config, "bc_use_layernorm", False),
        device=config.device,
    )

    enc_state = raw.get("encoder_state_dict")
    pol_state = raw.get("action_mlp_state_dict")
    if enc_state is None or pol_state is None:
        raise KeyError("Checkpoint missing encoder_state_dict or action_mlp_state_dict.")
    missing_enc, unexpected_enc = encoder.load_state_dict(enc_state, strict=False)
    missing_pol, unexpected_pol = policy.load_state_dict(pol_state, strict=False)
    if missing_enc or unexpected_enc:
        print(f"[warn] Encoder state mismatch (missing={missing_enc}, unexpected={unexpected_enc})")
    if missing_pol or unexpected_pol:
        print(f"[warn] Policy state mismatch (missing={missing_pol}, unexpected={unexpected_pol})")
    encoder.eval().requires_grad_(False)
    policy.eval().requires_grad_(False)
    encoder.to(config.device)
    policy.to(config.device)
    return config, encoder, policy


def _build_env_cfg(config: SimpleNamespace, args) -> SimpleNamespace:
    camera_keys = _as_tuple(args.camera_obs_keys, getattr(config, "camera_obs_keys", tuple(DEFAULT_CAMERA_KEYS)))
    video_cam_keys = _as_tuple(args.video_camera_keys, camera_keys)
    return SimpleNamespace(
        robosuite_task=getattr(config, "robosuite_task", "PickPlaceCan"),
        robosuite_robots=_as_tuple(getattr(config, "robosuite_robots", ("Panda",)), ("Panda",)),
        robosuite_controller=getattr(config, "robosuite_controller", "OSC_POSE"),
        render=bool(getattr(config, "has_renderer", getattr(config, "render", False)) or args.render),
        robosuite_reward_shaping=bool(getattr(config, "robosuite_reward_shaping", False)),
        robosuite_control_freq=int(getattr(config, "robosuite_control_freq", 20)),
        max_env_steps=args.max_env_steps,
        camera_obs_keys=camera_keys,
        flip_camera_keys=_as_tuple(args.flip_camera_keys, getattr(config, "flip_camera_keys", camera_keys)),
        bc_cnn_keys_order=_as_tuple(getattr(config, "bc_cnn_keys_order", None), tuple(DEFAULT_CNN_KEYS)),
        bc_mlp_keys_order=_as_tuple(getattr(config, "bc_mlp_keys_order", None), tuple(DEFAULT_MLP_KEYS)),
        clip_actions=bool(getattr(config, "clip_actions", False) or args.clip_actions),
        episodes=args.episodes,
        num_envs=args.num_envs,
        seed=getattr(config, "seed", 0),
        video_dir=pathlib.Path(args.video_dir).expanduser(),
        video_fps=args.video_fps,
        video_top_k=args.video_top_k,
        video_camera=args.video_camera or (video_cam_keys[0] if video_cam_keys else None),
        video_camera_keys=video_cam_keys,
        disable_video=bool(args.no_video),
        controller_configs=getattr(config, "controller_configs", None),
        ignore_done=bool(getattr(config, "ignore_done", False)),
        has_offscreen_renderer=getattr(config, "has_offscreen_renderer", True),
        has_renderer=getattr(config, "has_renderer", getattr(config, "render", False)),
        camera_depths=bool(getattr(config, "camera_depths", False)),
        use_camera_obs=bool(getattr(config, "use_camera_obs", True)),
        bc_crop_height=getattr(config, "bc_crop_height", 0),
        bc_crop_width=getattr(config, "bc_crop_width", 0),
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a BC sweep checkpoint in the robosuite Lift task.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to BC_Sweep checkpoint (.pt)")
    parser.add_argument("--offline_traindir", type=str, default=None, help="Override offline_traindir for shape inference")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device override (e.g., cpu, cuda:0)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of rollouts to run")
    parser.add_argument("--max_env_steps", type=int, default=500, help="Max env steps per episode")
    parser.add_argument("--video_dir", type=str, default="videos/lift_eval", help="Directory to write rollout videos")
    parser.add_argument("--video_fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--video_camera", type=str, default="", help="Specific camera for video rendering (defaults to first camera key)")
    parser.add_argument("--video_camera_keys", nargs="+", default=None, help="Camera keys to stack for video rendering")
    parser.add_argument("--video_top_k", type=int, default=-1, help="Keep only top-K reward episode videos (-1 keeps all, 0 disables video)")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording/encoding during eval")
    parser.add_argument("--camera_obs_keys", nargs="+", default=None, help="Camera observation keys to feed into encoder")
    parser.add_argument("--flip_camera_keys", nargs="+", default=None, help="Camera keys to vertically flip before stacking")
    parser.add_argument("--clip_actions", action="store_true", help="Clip actions to [-1, 1] before env step")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering (slow)")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, help="Override image size (height width)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs to use during eval")
    parser.add_argument("--env_config", type=str, default=None, help="Config block name in configs.yaml to override env settings (e.g., lift_env_eval)")
    args = parser.parse_args()
    args.env_config_block = _load_env_block(args.env_config) if args.env_config else None
    return args


def main():
    args = _parse_args()
    env_overrides = args.env_config_block or {}
    config, encoder, policy = _load_policy_from_checkpoint(
        args.checkpoint, device=args.device, offline_traindir=args.offline_traindir
    )
    config.env_config = env_overrides or getattr(config, "env_config", None)
    for key in ("robosuite_task", "robosuite_controller", "robosuite_reward_shaping", "robosuite_control_freq", "has_renderer", "has_offscreen_renderer", "ignore_done", "camera_depths", "use_camera_obs", "bc_crop_height", "bc_crop_width"):
        if key in env_overrides:
            setattr(config, key, env_overrides[key])
    if "robosuite_robots" in env_overrides:
        robots = env_overrides["robosuite_robots"]
        config.robosuite_robots = tuple(robots) if isinstance(robots, (list, tuple)) else (robots,)
    if "controller_configs" in env_overrides:
        config.controller_configs = env_overrides["controller_configs"]
    if "camera_obs_keys" in env_overrides:
        config.camera_obs_keys = tuple(env_overrides["camera_obs_keys"])
    if "flip_camera_keys" in env_overrides:
        config.flip_camera_keys = tuple(env_overrides["flip_camera_keys"])
    if "bc_cnn_keys_order" in env_overrides:
        config.bc_cnn_keys_order = tuple(env_overrides["bc_cnn_keys_order"])
    if "bc_mlp_keys_order" in env_overrides:
        config.bc_mlp_keys_order = tuple(env_overrides["bc_mlp_keys_order"])
    if "camera_heights" in env_overrides and "camera_widths" in env_overrides:
        config.size = (int(env_overrides["camera_heights"]), int(env_overrides["camera_widths"]))
    if args.image_size:
        config.size = tuple(args.image_size)
    env_cfg = _build_env_cfg(config, args)
    metrics = evaluate_in_environment(config, encoder, policy, env_cfg, run=SimpleNamespace(log=lambda *a, **k: None))
    print("Finished evaluation.")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
