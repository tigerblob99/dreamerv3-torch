"""
python evals/joint_encoder_eval.py \
  --configs joint_train \
  --env_config can_env_eval \
  --offline_traindir ./datasets/robomimic_data_MV/can_PH_train \
  --checkpoint /workspace/dreamerv3-torch/dreamerv3-torch/logdir/joint_run_04/latest.pt \
  --eval_episodes 500 \
  --num_envs 20
"""

import argparse
import pathlib
import sys
from types import SimpleNamespace

import ruamel.yaml as yaml
import torch

# Hardcode repository root: parent of this evals directory holds configs.yaml
REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import tools
from models import WorldModel
from parallel import Parallel

# Import helpers from joint_train with a defensive fallback

from joint_train import _define_spaces, ActionMLP, EnvWorker, evaluate_online


# Optional env-config loader (used in joint_train and BC_Sweep)
try:
    from BC_Sweep import _load_env_block
except Exception:
    _load_env_block = None


def _parse_config():
    """Parse CLI args with config/env_config handling mirroring joint_train."""

    # Pre-parse to capture config block names
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--configs", nargs="+")
    pre_parser.add_argument(
        "--env_config",
        type=str,
        default=None,
        help="Env config block name from configs.yaml",
    )
    pre_args, remaining = pre_parser.parse_known_args()

    cfg_path = REPO_ROOT / "configs.yaml"
    configs = yaml.safe_load(cfg_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *(pre_args.configs or [])]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    # Merge env-config block (dict/list fields stored separately to be set after argparse)
    complex_defaults = {}
    if pre_args.env_config:
        print(f"Loading env config block: {pre_args.env_config}")
        env_defaults = {}
        if _load_env_block:
            try:
                env_defaults = _load_env_block(pre_args.env_config) or {}
            except FileNotFoundError:
                print("Warning: BC_Sweep._load_env_block could not find configs.yaml; using local fallback.")
                env_defaults = configs.get(pre_args.env_config, {}) or {}
            except Exception as exc:
                print(f"Warning: Error in _load_env_block: {exc}")
                env_defaults = configs.get(pre_args.env_config, {}) or {}
        else:
            env_defaults = configs.get(pre_args.env_config, {}) or {}

        for k, v in env_defaults.items():
            if isinstance(v, (dict, list)):
                complex_defaults[k] = v
            else:
                defaults[k] = v

    # Additional defaults required for evaluation
    defaults.setdefault("num_workers", 0)
    defaults.setdefault("bc_loss_scale", 1.0)
    defaults.setdefault("wm_loss_scale", 1.0)
    defaults.setdefault("batch_length", 64)
    defaults.setdefault("batch_size", 16)
    defaults.setdefault("save_every", 10000)
    defaults.setdefault("robosuite_task", "Lift")
    defaults.setdefault("robosuite_robots", ["Panda"])
    defaults.setdefault("robosuite_controller", "OSC_POSE")
    defaults.setdefault("robosuite_reward_shaping", False)
    defaults.setdefault("robosuite_control_freq", 20)
    defaults.setdefault("has_renderer", False)
    defaults.setdefault("has_offscreen_renderer", True)
    defaults.setdefault("use_camera_obs", True)
    defaults.setdefault("camera_depths", False)
    defaults.setdefault("ignore_done", False)
    defaults.setdefault("clip_actions", False)
    defaults.setdefault("num_envs", 1)
    defaults.setdefault("eval_episodes", 10)

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--env_config", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint produced by joint_train (contains wm/policy)",
    )
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        if key in complex_defaults:
            continue
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    config = parser.parse_args(remaining)
    for k, v in complex_defaults.items():
        setattr(config, k, v)
    return config


def _load_models(config):
    """Instantiate WorldModel + policy and load weights from checkpoint."""

    episodes = tools.load_episodes(config.offline_traindir, limit=config.dataset_size)
    if not episodes:
        raise RuntimeError(f"No episodes found in {config.offline_traindir} for space inference.")
    sample_ep = next(iter(episodes.values()))
    obs_space, act_space = _define_spaces(sample_ep, config)
    config.num_actions = act_space.shape[0]

    wm = WorldModel(obs_space, act_space, 0, config).to(config.device)
    feat_size = (
        config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        if config.dyn_discrete
        else config.dyn_stoch + config.dyn_deter
    )
    policy = ActionMLP(
        inp_dim=feat_size,
        shape=(config.num_actions,),
        layers=4,
        units=1024,
        act=config.act,
        norm=config.norm,
    ).to(config.device)

    ckpt_path = pathlib.Path(config.checkpoint).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device(config.device))
    if "wm" not in ckpt or "policy" not in ckpt:
        raise KeyError("Checkpoint must contain 'wm' and 'policy' keys from joint_train")
    wm.load_state_dict(ckpt["wm"], strict=False)
    policy.load_state_dict(ckpt["policy"], strict=False)
    wm.requires_grad_(False)
    policy.requires_grad_(False)
    return wm, policy


def _build_envs(config):
    crop_h = config.image_crop_height
    crop_w = config.image_crop_width
    image_hw = (crop_h, crop_w) if (crop_h > 0 and crop_w > 0) else (84, 84)

    env_cfg = SimpleNamespace(
        robosuite_task=getattr(config, "robosuite_task", "PickPlaceCan"),
        robosuite_robots=getattr(config, "robosuite_robots", ["Panda"]),
        robosuite_controller=getattr(config, "robosuite_controller", "OSC_POSE"),
        robosuite_reward_shaping=getattr(config, "robosuite_reward_shaping", False),
        robosuite_control_freq=getattr(config, "robosuite_control_freq", 20),
        max_env_steps=getattr(config, "max_env_steps", 500),
        ignore_done=getattr(config, "ignore_done", False),
        has_renderer=getattr(config, "has_renderer", False),
        has_offscreen_renderer=getattr(config, "has_offscreen_renderer", True),
        use_camera_obs=getattr(config, "use_camera_obs", True),
        camera_depths=getattr(config, "camera_depths", False),
        camera_obs_keys=tuple(getattr(config, "camera_obs_keys", ["agentview_image"])),
        seed=config.seed,
        render=getattr(config, "render", False),
    )
    if hasattr(config, "controller_configs"):
        env_cfg.controller_configs = config.controller_configs

    envs = []
    for _ in range(config.num_envs):
        worker = EnvWorker(env_cfg, image_hw)
        envs.append(Parallel(worker, "process"))
    return envs


def main():
    config = _parse_config()
    tools.set_seed_everywhere(config.seed)

    wm, policy = _load_models(config)
    config.eval_episodes = int(config.eval_episodes)

    envs = _build_envs(config)
    try:
        metrics = evaluate_online(wm, policy, config, step=0, run=None, envs=envs)
        print("Evaluation complete:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()