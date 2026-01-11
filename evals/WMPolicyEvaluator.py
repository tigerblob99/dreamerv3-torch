"""
"Closed-loop BC reconstruction error vs. open-loop WM reconstruction error"

# Example:
# python evals/WMPolicyEvaluator.py --configs joint_train --env_config can_env_eval_WMPolicy --offline_traindir ./datasets/robomimic_data_MV/can_PH_train --checkpoint /workspace/dreamerv3-torch/dreamerv3-torch/logdir/joint_run_04/latest.pt
"""

import argparse
import pathlib
import sys
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np

import ruamel.yaml as yaml
import torch
import torch.nn.functional as F

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import tools
from models import WorldModel
from joint_train import _define_spaces, ActionMLP
from joint_model import joint_model
from evals.EvalEnvWorker import EvalEnvWorker
from evals.EvalDataset import EvalDataset
from torch.utils.data import DataLoader
from parallel import Parallel


# Optional env-config loader (used in joint_train and BC_Sweep)
from BC_Sweep import _load_env_block



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
    defaults.setdefault("dataset_size", 1000)
    defaults.setdefault("warmup_len", 5)
    defaults.setdefault("horizon", 10)


    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--env_config", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint produced by joint_train (contains wm/policy)",
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to the raw HDF5 (used ONLY for 'env_state' resets)."
    )
    parser.add_argument(
        "--offline_evaldir",
        type=str,
        required=True,
        help="Path to the NPZ eval dir."
    )
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        if key in complex_defaults:
            continue
        if key == "offline_evaldir":
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
        layers=config.mlp_layers,
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


def main():
    config = _parse_config()
    tools.set_seed_everywhere(config.seed)

    # 1. Load the model
    wm, policy = _load_models(config)
    jm = joint_model(config, wm, policy)
    jm.eval()

    # 2. Load the dataset
    dataset = EvalDataset(
        hdf5_path=config.hdf5_path,
        eval_dir=config.offline_evaldir,
        config=config,
        warmup_len=config.warmup_len,
        horizon=config.horizon
    )
    Loader = DataLoader(dataset, batch_size=config.num_envs, num_workers=0, drop_last=True)

    # 3. Setup the environment
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

    crop_h = config.image_crop_height
    crop_w = config.image_crop_width
    image_hw = (crop_h, crop_w) if (crop_h > 0 and crop_w > 0) else (84, 84)
    
    envs = [Parallel(lambda: EvalEnvWorker(config, env_cfg, image_hw), "process")
           for _ in range(config.num_envs)
           ]
    
    # 4. Run evaluations
    real_env_errors = eval_closed_loop_bc_real_env(jm, Loader, envs, config)
    wm_open_loop_errors = eval_open_loop_wm(jm, Loader, config)
    wm_closed_loop_errors = eval_closed_loop_bc_wm(jm, Loader, config)

    # 5. Plotting
    assert len(real_env_errors) == len(wm_open_loop_errors) == len(wm_closed_loop_errors)
    plot_errors(real_env_errors, wm_open_loop_errors, "real_env_vs_wm_open_loop")
    plot_errors(real_env_errors, wm_closed_loop_errors, "real_env_vs_wm_closed_loop")
    plot_errors(wm_open_loop_errors, wm_closed_loop_errors, "wm_open_loop_vs_wm_closed_loop")


from tqdm import tqdm

def eval_closed_loop_bc_real_env(jm, Loader, envs, config):
    """
    1. Closed-loop BC reconstruction error in the real environment:
        - Sample a state from the demonstration data.
        - Rollout the BC policy in the real environment for a short horizon.
        - Compare the trajectory from the policy with the ground truth demonstration trajectory.
        - Measure the Mean Squared Error (MSE).
    """
    print("Evaluating closed-loop BC in real environment...")
    errors = []
    num_envs = len(envs)
    with torch.no_grad():
        for batch in tqdm(Loader):
            # 1. Set states in all envs in parallel
            set_state_promises = [envs[i].set_state(batch['env_state'][i]) for i in range(num_envs)]
            current_obs_list = [p() for p in set_state_promises]

            # 2. Warm up RSSM for the whole batch
            warmup_data = {k.replace('warmup_', ''): v.to(config.device) for k, v in batch.items() if k.startswith('warmup_')}
            rssm_state = jm._warmup_rssm(warmup_data)

            # 3. Rollout policy in parallel
            batch_policy_trajectory_obs = [[] for _ in range(num_envs)]
            prev_actions = torch.zeros((num_envs, config.num_actions), device=config.device)
            for t in range(config.horizon):
                # Collate observations from list of dicts to a dict of batched tensors
                obs_batch = {
                    key: torch.from_numpy(np.stack([obs[key] for obs in current_obs_list])).to(config.device)
                    for key in current_obs_list[0].keys()
                }
                # Ensure 'is_first' is a 1D tensor
                if 'is_first' in obs_batch:
                    obs_batch['is_first'] = obs_batch['is_first'].squeeze(-1)

                actions, rssm_state = jm.obs_2_act(obs_batch, rssm_state, action=prev_actions)
                
                # Step all environments in parallel
                step_promises = [envs[i].step(actions[i].cpu().numpy()) for i in range(num_envs)]
                step_results = [p() for p in step_promises]

                prev_actions = actions

                # Update current observations and store images
                current_obs_list = []
                for i in range(num_envs):
                    obs, reward, done, info, success = step_results[i]
                    current_obs_list.append(obs)
                    batch_policy_trajectory_obs[i].append(obs['image'])

            # 4. Calculate error for the batch
            # policy_images shape: (B, T, H, W, C)
            policy_images = np.stack([np.stack(traj) for traj in batch_policy_trajectory_obs])
            target_images = batch['target_image'].numpy()
            
            policy_images_scaled = policy_images / 255.0 - 0.5
            target_images_scaled = target_images / 255.0 - 0.5
            
            # Calculate MSE for each episode in the batch
            mse = ((policy_images_scaled - target_images_scaled) ** 2).mean(axis=(1, 2, 3, 4))
            errors.extend(mse.tolist())
        
    return errors


def eval_open_loop_wm(jm, Loader, config):
    """
    2. Open-loop reconstruction error in the world model:
        - Sample a state from the demonstration data.
        - Take the actions from the demonstration data.
        - Feed these actions into the world model.
        - Compare the world model's reconstructed states with the ground truth demonstration states.
        - Measure the MSE.
    """
    print("Evaluating open-loop WM...")
    errors = []
    with torch.no_grad():
        for batch in tqdm(Loader):
            # 1. Warm up RSSM
            warmup_data = {k.replace('warmup_', ''): v.to(config.device) for k, v in batch.items() if k.startswith('warmup_')}
            current_state = jm._warmup_rssm(warmup_data)

            # 2. Get target actions
            target_actions = batch['target_action'].to(config.device)

            # 3. Open-loop rollout in WM
            wm_rollout_obs = []
            # `target_actions` has shape (B, T, A), we need to iterate through T
            for t in range(config.horizon):
                # action has shape (B, A)
                action = target_actions[:, t]
                
                next_state = jm.wm.dynamics.img_step(current_state, action, sample=False)
                
                feats = jm.feats(next_state)
                preds = jm.get_preds(feats)
                reconstructed_image = preds["image"]
                
                wm_rollout_obs.append(reconstructed_image)
                current_state = next_state

            # 4. Calculate error
            # Stack along the time dimension
            wm_images = torch.stack(wm_rollout_obs, dim=1).cpu().numpy() # Shape: (B, T, C, H, W)
            target_images = batch['target_image'].numpy()
            target_images_scaled = target_images / 255.0

            # Assuming wm_images are in [0, 1] range from the model
            mse = ((wm_images - target_images_scaled) ** 2).mean(axis=(1, 2, 3, 4))
            errors.extend(mse.tolist())

    return errors


def eval_closed_loop_bc_wm(jm, Loader, config):
    """
    3. Closed-loop BC reconstruction error in the world model:
        - Sample a state from the demonstration data.
        - Start from this state in the world model.
        - For a short horizon:
            - Get an action from the BC policy based on the current world model state.
            - Feed this action into the world model to get the next state.
        - Compare the resulting trajectory of world model states with the ground truth demonstration trajectory.
        - Measure the MSE.
    """
    print("Evaluating closed-loop BC in WM...")
    errors = []
    with torch.no_grad():
        for batch in tqdm(Loader):
            # 1. Warm up RSSM
            warmup_data = {k.replace('warmup_', ''): v.to(config.device) for k, v in batch.items() if k.startswith('warmup_')}
            start_state = jm._warmup_rssm(warmup_data)
            
            # 2. Imagine rollout with policy - this is already batched
            feats, _, _ = jm.imagine(start_state, config.horizon) # feats: (T, B, D)

            # 3. Decode images from feats
            # Reshape from (T, B, D) to (T*B, D) for the heads
            T, B, D = feats.shape
            preds = jm.get_preds(feats.view(T * B, D))
            # Reshape back to (T, B, C, H, W) and then permute to (B, T, C, H, W)
            image_pred = preds["image"]
            reconstructed_images = image_pred.reshape(T, B, *image_pred.shape[1:]).permute(1, 0, 2, 3, 4)

            # 4. Calculate error
            wm_images = reconstructed_images.cpu().numpy()
            target_images = batch['target_image'].numpy()
            target_images_scaled = target_images / 255.0 - 0.5
            
            # Assuming wm_images are in [-0.5, 0.5] range from the model
            mse = ((wm_images - target_images_scaled) ** 2).mean(axis=(1, 2, 3, 4))
            errors.extend(mse.tolist())
            
    return errors


def plot_errors(errors1, errors2, title):
    """Generates and saves a scatter plot of two sets of errors."""
    plt.figure()
    plt.scatter(errors1, errors2)
    plt.xlabel("Error 1")
    plt.ylabel("Error 2")
    plt.title(title)
    plt.savefig(f"imgs/{title}.png")
    print(f"Saved plot to imgs/{title}.png")


if __name__ == "__main__":
    main()
