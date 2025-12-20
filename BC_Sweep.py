"""
python BC_Sweep.py \
  --configs bc_defaults bc_can_PH \
  --checkpoint /workspace/dreamerv3-torch/dreamerv3-torch/logdir/robomimic_offline_can_MH_cropped/latest.pt \
  --offline_traindir ./datasets/robomimic_data_MV/can_PH_train \
  --offline_evaldir ./datasets/robomimic_data_MV/can_PH_eval \
  --lr_grid 1e-4 1e-5 1e-3 \
  --weight_decay_grid 0.0 1e-5 1e-4 1e-3 \
  --sweep_name bc_frozen_encoder_cropped \
  --env_config lift_env_eval \
  --video_dir ./videos/bc_sweep_frozen_cropped \
  --save_dir ./checkpoints/bc_sweep_frozen_cropped \
  --env_episodes 10 \
  --env_max_steps 500 \
  --eval_epochs 200 400 600 800 1000 \
  --bc_crop_height 78 \
  --bc_crop_width 78 \
"""

import argparse
import os
import pathlib
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
import torch
import wandb

import tools
from bc_mlp.BC_MLP_train import (
    _load_config,
    BC_MLP,
    _flatten_episodes,
    _sample_batch,
    _build_ordered_obs_space,
    _build_encoder,
    build_action_mlp,
    _center_crop,
)
from offline_train import _infer_spaces
from bc_mlp.BC_MLP_eval import (
    _make_robomimic_env,
    _prepare_obs,
    _obs_to_torch,
    _extract_success,
    EpisodeVideoRecorder,
)

os.environ.setdefault("MUJOCO_GL", "osmesa")


DEFAULT_CNN_KEYS = ("image",)
DEFAULT_MLP_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "aux_robot0_joint_pos_sin",
    "aux_robot0_joint_pos_cos",
)
DEFAULT_CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def _load_env_block(name: str | None):
    """Load env config block from configs.yaml if provided."""
    if not name:
        return None
    cfg_path = pathlib.Path(__file__).resolve().parent / "configs.yaml"
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


def _build_scratch_policy(config):
    """Build a randomly initialized encoder + policy (no Dreamer checkpoint)."""
    if not config.offline_traindir:
        raise ValueError("--offline_traindir must be provided to infer observation shapes.")
    tools.set_seed_everywhere(config.seed)
    dataset = tools.load_episodes(config.offline_traindir, limit=1)
    if not dataset:
        raise RuntimeError(f"No episodes found in {config.offline_traindir}.")
    first_episode = next(iter(dataset.values()))
    obs_space = _build_ordered_obs_space(config, first_episode)
    _, act_space = _infer_spaces(first_episode)
    encoder = _build_encoder(config, obs_space)
    action_mlp = build_action_mlp(
        encoder,
        act_space,
        hidden_units=getattr(config, "bc_hidden_units", 1024),
        hidden_layers=getattr(config, "bc_hidden_layers", 4),
        act_name=getattr(config, "bc_activation", "SiLU"),
        norm=getattr(config, "bc_use_layernorm", False),
        device=config.device,
    )
    print(
        f"Built scratch encoder+MLP: hidden_layers={getattr(config, 'bc_hidden_layers', 4)}, "
        f"hidden_units={getattr(config, 'bc_hidden_units', 1024)}, action_dim={act_space.shape[0]}"
    )
    return encoder, action_mlp


def _prepare_datasets(config):
    """Load and split offline data once so every sweep run sees the same samples."""
    print("Preparing BC dataset once for the entire sweep...")
    train_eps = tools.load_episodes(config.offline_traindir, limit=config.dataset_size)
    eval_dir = getattr(config, "offline_evaldir", None)
    eval_eps = tools.load_episodes(eval_dir, limit=config.dataset_size) if eval_dir else {}
    if not train_eps:
        raise RuntimeError("No episodes for BC training.")

    if not eval_eps and config.bc_eval_split > 0:
        episode_items = list(train_eps.items())
        eval_cut = int(len(episode_items) * config.bc_eval_split)
        eval_eps = dict(episode_items[:eval_cut]) if eval_cut > 0 else {}
        train_eps = dict(episode_items[eval_cut:])

    train_samples = _flatten_episodes(train_eps)
    eval_samples = _flatten_episodes(eval_eps) if eval_eps else []
    eval_episodes_list = list(eval_eps.values()) if eval_eps else []
    print(f"Loaded {len(train_samples)} train samples and {len(eval_samples)} eval samples.")
    return train_samples, eval_samples, eval_episodes_list


def BC_MLP_train(
    config,
    encoder,
    action_mlp,
    train_samples,
    eval_samples,
    eval_episodes_list,
    train_encoder=True,
    run=None,
    checkpoint_epochs=None,
    checkpoint_cb=None,
):
    """Train encoder + policy head and return final eval loss."""
    log_fn = run.log if run else wandb.log
    if train_encoder:
        encoder.requires_grad_(True)
        encoder.train()
        trainable_params = list(encoder.parameters()) + list(action_mlp.parameters())
    else:
        encoder.requires_grad_(False)
        encoder.eval()
        trainable_params = list(action_mlp.parameters())
    action_mlp.train()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.bc_lr,
        weight_decay=config.bc_weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    steps_per_epoch = max(1, len(train_samples) // config.bc_batch_size)
    total_steps = config.bc_epochs * steps_per_epoch if config.bc_max_steps < 0 else config.bc_max_steps
    print(f"BC training: epochs={config.bc_epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")

    def evaluate():
        if not eval_samples:
            return None
        encoder.eval()
        action_mlp.eval()
        losses = []
        with torch.no_grad():
            batch_size = min(config.bc_batch_size, len(eval_samples))
            for start in range(0, len(eval_samples), batch_size):
                chunk = eval_samples[start : start + batch_size]
                obs_list = {k: [] for k in chunk[0][0].keys()}
                act_list = []
                for obs, act in chunk:
                    for k, v in obs.items():
                        obs_list[k].append(v)
                    act_list.append(act)
                for k in obs_list:
                    arr = np.asarray(obs_list[k])
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    obs_tensor = torch.as_tensor(arr, device=config.device).float()
                    if k == "image":
                        if getattr(config, "bc_crop_height", 0) and getattr(config, "bc_crop_width", 0):
                            obs_tensor = _center_crop(
                                obs_tensor,
                                int(getattr(config, "bc_crop_height", 0)),
                                int(getattr(config, "bc_crop_width", 0)),
                            )
                        obs_tensor = obs_tensor / 255.0
                    obs_list[k] = obs_tensor
                acts = torch.as_tensor(np.asarray(act_list), device=config.device).float()
                enc_in = {k: v for k, v in obs_list.items()}
                embedding = encoder(enc_in)
                pred = action_mlp(embedding)
                losses.append(torch.nn.functional.mse_loss(pred, acts).item())

        if eval_episodes_list:
            with torch.no_grad():
                ep_idx = np.random.randint(0, len(eval_episodes_list))
                ep = eval_episodes_list[ep_idx]
                ep_length = ep["action"].shape[0]
                # Skip dummy action at index 0 and align obs[t-1] with action[t].
                obs_list = {k: [] for k in ep.keys() if k not in ("action",) and not k.startswith("log_")}
                gt_acts = []
                for t in range(1, ep_length):
                    obs_idx = t - 1
                    for k in obs_list.keys():
                        obs_list[k].append(ep[k][obs_idx])
                    gt_acts.append(ep["action"][t])

                for k in obs_list:
                    arr = np.asarray(obs_list[k])
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    obs_tensor = torch.as_tensor(arr, device=config.device).float()
                    if k == "image":
                        if getattr(config, "bc_crop_height", 0) and getattr(config, "bc_crop_width", 0):
                            obs_tensor = _center_crop(
                                obs_tensor,
                                int(getattr(config, "bc_crop_height", 0)),
                                int(getattr(config, "bc_crop_width", 0)),
                            )
                        obs_tensor = obs_tensor / 255.0
                    obs_list[k] = obs_tensor

                enc_in = {k: v for k, v in obs_list.items()}
                pred_acts = action_mlp(encoder(enc_in)).cpu().numpy()
                gt_acts = np.array(gt_acts)

                action_dim = gt_acts.shape[1]
                fig, axes = plt.subplots(action_dim, 1, figsize=(10, 2 * action_dim), sharex=True)
                if action_dim == 1:
                    axes = [axes]
                for i in range(action_dim):
                    axes[i].plot(gt_acts[:, i], label="Ground Truth", color="black", alpha=0.6)
                    axes[i].plot(pred_acts[:, i], label="Predicted", color="red", linestyle="--", alpha=0.6)
                    axes[i].set_ylabel(f"Dim {i}")
                    if i == 0:
                        axes[i].legend()
                plt.suptitle(f"Episode {ep_idx} ({ep_length} steps)")
                plt.tight_layout()
                log_fn({"eval/action_plots": wandb.Image(fig)}, commit=False)
                plt.close(fig)

        action_mlp.train()
        encoder.train(mode=train_encoder)
        return float(np.mean(losses)) if losses else None

    checkpoint_set = set(checkpoint_epochs or [])
    global_step = 0
    final_eval_loss = None
    for epoch in range(1, config.bc_epochs + 1):
        if config.bc_shuffle:
            np.random.shuffle(train_samples)
        epoch_losses = []
        for _ in range(steps_per_epoch):
            if global_step >= total_steps:
                break
            batch_obs, batch_actions = _sample_batch(
                train_samples,
                config.bc_batch_size,
                config.device,
                getattr(config, "bc_crop_height", 0),
                getattr(config, "bc_crop_width", 0),
            )
            if train_encoder:
                embedding = encoder(batch_obs)
            else:
                with torch.no_grad():
                    embedding = encoder(batch_obs)
            pred_actions = action_mlp(embedding)
            loss = loss_fn(pred_actions, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.bc_grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())
            global_step += 1
        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        eval_loss = evaluate() if (epoch % config.bc_eval_every == 0) else None
        if (epoch in checkpoint_set) and eval_loss is None:
            eval_loss = evaluate()
        if eval_loss is not None:
            final_eval_loss = eval_loss
        log_payload = {"train/loss": avg_train_loss, "epoch": epoch}
        if eval_loss is not None:
            log_payload["eval/loss"] = eval_loss
        log_fn(log_payload)
        if epoch in checkpoint_set and checkpoint_cb is not None:
            checkpoint_cb(epoch, encoder, action_mlp, eval_loss)
        if global_step >= total_steps:
            print("Reached max BC steps; stopping early.")
            break

    if final_eval_loss is None:
        final_eval_loss = evaluate()
        if final_eval_loss is not None:
            log_fn({"eval/loss": final_eval_loss})

    print("BC training complete.")
    return final_eval_loss


def evaluate_in_environment(config, encoder, policy, env_cfg, run=None):
    """Roll out the trained policy in the real environment and log metrics."""
    log_fn = run.log if run else wandb.log

    size = tuple(getattr(config, "size", (84, 84)))
    device = torch.device(config.device)
    encoder.eval().to(device)
    policy.eval().to(device)

    env = _make_robomimic_env(env_cfg, size)
    record_video = not bool(getattr(env_cfg, "disable_video", False))
    video_recorder = (
        EpisodeVideoRecorder(
            directory=env_cfg.video_dir,
            fps=env_cfg.video_fps,
            camera_key=env_cfg.video_camera,
            camera_keys=env_cfg.video_camera_keys,
            flip_keys=env_cfg.flip_camera_keys,
        )
        if record_video
        else None
    )

    rewards = []
    successes = []
    successes_env = []
    best_reward = -np.inf
    best_video = None

    for episode in range(env_cfg.episodes):
        raw_obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        info = {}
        env_success = False
        if video_recorder:
            video_recorder.start_episode(episode)
        episode_actions: list = []
        while not done and steps < env_cfg.max_env_steps:
            step_obs = _prepare_obs(
                raw_obs,
                cnn_keys_order=env_cfg.bc_cnn_keys_order,
                mlp_keys_order=env_cfg.bc_mlp_keys_order,
                camera_keys=env_cfg.camera_obs_keys,
                flip_keys=env_cfg.flip_camera_keys,
                crop_height=getattr(env_cfg, "bc_crop_height", None),
                crop_width=getattr(env_cfg, "bc_crop_width", None),
            )
            tensor_obs = _obs_to_torch(step_obs, device)
            with torch.no_grad():
                embedding = encoder(tensor_obs)
                action = policy(embedding)
            action_np = action.squeeze(0).cpu().numpy()
            episode_actions.append(action_np.copy())
            if env_cfg.clip_actions:
                action_np = np.clip(action_np, -1.0, 1.0)
            raw_obs, reward, done, info = env.step(action_np)
            if video_recorder:
                video_recorder.add_frame(raw_obs)
            total_reward += float(reward)
            steps += 1
            env_success = env_success or env._check_success()
            if env_success:
                done = True
            if done:
                break

        env_success = env_success or env._check_success()
        successes_env.append(env_success)
        success = total_reward > 0.0
        rewards.append(total_reward)
        successes.append(success)
        if episode_actions:
            acts = np.stack(episode_actions)
            action_dim = acts.shape[1]
            steps_arr = np.arange(acts.shape[0])
            fig, axes = plt.subplots(action_dim, 1, figsize=(10, 2 * action_dim), sharex=True)
            if action_dim == 1:
                axes = [axes]
            for i in range(action_dim):
                axes[i].plot(steps_arr, acts[:, i], label="policy_action")
                axes[i].set_ylabel(f"Dim {i}")
                if i == 0:
                    axes[i].legend()
            axes[-1].set_xlabel("Step")
            plt.tight_layout()
            log_fn({f"env/episode_{episode}/actions_image": wandb.Image(fig)}, commit=False)
            plt.close(fig)
        saved_video = video_recorder.finish_episode() if video_recorder else None
        log_fn(
            {
                "env/episode_reward": total_reward,
                "env/episode_success": success,
                "env/episode_env_success": env_success,
                "env/episode_steps": steps,
            },
            commit=False,
        )
        print(f"[env eval] episode {episode+1}/{env_cfg.episodes} reward={total_reward:.3f} success={success} env_success={env_success} steps={steps}")
        if saved_video and total_reward >= best_reward:
            best_reward = total_reward
            best_video = saved_video

        # Debug: print env info and is_success for troubleshooting success detection.
        try:
            success_state = env.is_success() if hasattr(env, "is_success") else None
        except Exception:
            success_state = None
        print(f"[env eval] episode {episode+1} info={info} is_success={success_state}")

    mean_reward = float(np.mean(rewards)) if rewards else float("nan")
    env_successes = [s for s in successes_env] if 'successes_env' in locals() else []
    env_success_rate = float(np.mean(env_successes)) if env_successes else 0.0
    reward_success_rate = float(np.mean(successes)) if successes else 0.0
    metrics = {
        "env/mean_reward": mean_reward,
        "env/success_rate": env_success_rate,
        "env/reward_success_rate": reward_success_rate,
        "env/best_reward": best_reward,
    }
    log_fn(metrics)
    if best_video:
        log_fn({"env/best_video": wandb.Video(str(best_video), fps=env_cfg.video_fps, caption="best_reward")})
    return metrics


def _parse_args():
    sweep_parser = argparse.ArgumentParser()
    sweep_parser.add_argument("--configs", nargs="+", help="Config presets to load with _load_config")
    sweep_parser.add_argument("--lr_grid", type=float, nargs="+", default=None)
    sweep_parser.add_argument("--weight_decay_grid", type=float, nargs="+", default=None)
    sweep_parser.add_argument("--eval_epochs", type=int, nargs="+", default=None, help="Epochs to checkpoint + run env eval (defaults to final epoch)")
    sweep_parser.add_argument("--sweep_id", type=str, default="", help="Use an existing W&B sweep id (entity/project/sweep_id)")
    sweep_parser.add_argument("--sweep_name", type=str, default="", help="Optional name for a newly created sweep")
    sweep_parser.add_argument("--agent_count", type=int, default=0, help="Number of sweep runs to execute (0 = all grid combos)")
    sweep_parser.add_argument("--env_episodes", type=int, default=3)
    sweep_parser.add_argument("--env_max_steps", type=int, default=500)
    sweep_parser.add_argument("--video_dir", type=str, default="videos/bc_sweep")
    sweep_parser.add_argument("--video_fps", type=int, default=20)
    sweep_parser.add_argument("--video_camera", type=str, default=None)
    sweep_parser.add_argument("--video_camera_keys", nargs="+", default=None)
    sweep_parser.add_argument("--camera_obs_keys", nargs="+", default=list(DEFAULT_CAMERA_KEYS))
    sweep_parser.add_argument(
        "--mlp_obs_keys",
        nargs="+",
        default=[
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ],
    )
    sweep_parser.add_argument("--flip_camera_keys", nargs="+", default=list(DEFAULT_CAMERA_KEYS))
    sweep_parser.add_argument("--clip_actions", action="store_true")
    sweep_parser.add_argument("--render", action="store_true")
    sweep_parser.add_argument("--image_size", type=int, nargs=2, default=[84, 84])
    sweep_parser.add_argument("--robosuite_task", type=str, default="Lift")
    sweep_parser.add_argument("--robosuite_robots", nargs="+", default=["Panda"])
    sweep_parser.add_argument("--robosuite_controller", type=str, default="OSC_POSE")
    sweep_parser.add_argument("--robosuite_reward_shaping", action="store_true")
    sweep_parser.add_argument("--robosuite_control_freq", type=int, default=20)
    sweep_parser.add_argument("--wandb_entity", type=str, default="4yp-a2i")
    sweep_parser.add_argument("--wandb_project", type=str, default="BC_practice")
    sweep_parser.add_argument("--wandb_group", type=str, default="bc_sweep")
    sweep_parser.add_argument("--save_dir", type=str, default="")
    sweep_parser.add_argument("--scratch_encoder", action="store_true", help="Train with a freshly initialized encoder instead of loading Dreamer weights")
    sweep_parser.add_argument("--env_config", type=str, default=None, help="Config block name in configs.yaml for env settings (e.g., lift_env_eval)")
    args, remaining = sweep_parser.parse_known_args()
    config = _load_config(args.configs, remaining)
    env_block = _load_env_block(args.env_config)
    if env_block:
        config.env_config = env_block
    return args, config


def main():
    args, base_config = _parse_args()
    tools.set_seed_everywhere(base_config.seed)

    # Fill missing config fields with sensible defaults for eval.
    base_config.camera_obs_keys = tuple(args.camera_obs_keys)
    base_config.mlp_obs_keys = tuple(args.mlp_obs_keys)
    base_config.flip_camera_keys = tuple(args.flip_camera_keys)
    base_config.bc_cnn_keys_order = tuple(getattr(base_config, "bc_cnn_keys_order", None) or DEFAULT_CNN_KEYS)
    base_config.bc_mlp_keys_order = tuple(getattr(base_config, "bc_mlp_keys_order", None) or DEFAULT_MLP_KEYS)
    base_config.clip_actions = bool(getattr(base_config, "clip_actions", False) or args.clip_actions)
    base_config.render = args.render
    base_config.robosuite_task = args.robosuite_task
    base_config.robosuite_robots = tuple(args.robosuite_robots)
    base_config.robosuite_controller = args.robosuite_controller
    base_config.robosuite_reward_shaping = args.robosuite_reward_shaping
    base_config.robosuite_control_freq = args.robosuite_control_freq
    base_config.size = tuple(args.image_size)

    lr_grid = args.lr_grid or [base_config.bc_lr]
    wd_grid = args.weight_decay_grid or [base_config.bc_weight_decay]
    # Preload data once; reused across sweep runs.
    data_bundle = _prepare_datasets(base_config)

    sweep_parameters = {
        "bc_lr": {"values": lr_grid},
        "bc_weight_decay": {"values": wd_grid},
        "seed": {"values": [base_config.seed]},
    }
    sweep_config = {
        "name": args.sweep_name or "bc_grid",
        "method": "grid",
        "metric": {"name": "eval/loss", "goal": "minimize"},
        "parameters": {k: v for k, v in sweep_parameters.items() if v["values"]},
    }

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
        print(f"Created sweep: {sweep_id}")

    total_combos = len(sweep_parameters["bc_lr"]["values"]) * len(sweep_parameters["bc_weight_decay"]["values"])
    agent_count = args.agent_count if args.agent_count > 0 else total_combos

    def run_bc_job():
        with wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group) as run:
            cfg = wandb.config
            run_config = SimpleNamespace(**vars(base_config))
            for attr in ("bc_lr", "bc_weight_decay", "seed"):
                if hasattr(cfg, attr):
                    setattr(run_config, attr, cfg[attr])
            # Ensure max steps follows epochs.
            run_config.bc_max_steps = -1
            tools.set_seed_everywhere(run_config.seed)

            env_block = getattr(run_config, "env_config", None) or {}
            for key in ("robosuite_task", "robosuite_controller", "robosuite_reward_shaping", "robosuite_control_freq", "has_renderer", "has_offscreen_renderer", "ignore_done", "camera_depths", "use_camera_obs", "bc_crop_height", "bc_crop_width"):
                if key in env_block:
                    setattr(run_config, key, env_block[key])
            if "robosuite_robots" in env_block:
                robots = env_block["robosuite_robots"]
                run_config.robosuite_robots = tuple(robots) if isinstance(robots, (list, tuple)) else (robots,)
            if "controller_configs" in env_block:
                run_config.controller_configs = env_block["controller_configs"]
            if "camera_obs_keys" in env_block:
                run_config.camera_obs_keys = tuple(env_block["camera_obs_keys"])
            if "flip_camera_keys" in env_block:
                run_config.flip_camera_keys = tuple(env_block["flip_camera_keys"])
            if "bc_cnn_keys_order" in env_block:
                run_config.bc_cnn_keys_order = tuple(env_block["bc_cnn_keys_order"])
            if "bc_mlp_keys_order" in env_block:
                run_config.bc_mlp_keys_order = tuple(env_block["bc_mlp_keys_order"])
            if "camera_heights" in env_block and "camera_widths" in env_block:
                run_config.size = (int(env_block["camera_heights"]), int(env_block["camera_widths"]))

            base_name = f"lr{run_config.bc_lr:g}_wd{run_config.bc_weight_decay:g}"
            encoder_init = "scratch" if args.scratch_encoder else "dreamer"
            if encoder_init == "scratch":
                encoder, action_mlp = _build_scratch_policy(run_config)
            else:
                encoder, action_mlp = BC_MLP(run_config)
            run.config.update({"encoder_init": encoder_init}, allow_val_change=True)
            run_name = f"{'scratch_' if encoder_init == 'scratch' else ''}{base_name}"
            run.name = run_name
            wandb.watch([encoder, action_mlp], log="all", log_freq=100)

            train_samples, eval_samples, eval_episodes_list = data_bundle
            checkpoint_epochs = sorted(set(args.eval_epochs or [run_config.bc_epochs]))

            def on_checkpoint(epoch, enc, mlp, eval_loss):
                # Save checkpoint if requested
                if args.save_dir:
                    save_root = pathlib.Path(args.save_dir).expanduser()
                    save_root.mkdir(parents=True, exist_ok=True)
                    save_path = save_root / f"{run_name}_epoch{epoch}.pt"
                    torch.save(
                        {
                            "action_mlp_state_dict": mlp.state_dict(),
                            "encoder_state_dict": enc.state_dict(),
                            "config": vars(run_config),
                            "epoch": epoch,
                            "eval_loss": eval_loss,
                        },
                        save_path,
                    )
                    print(f"[{run_name}] Saved checkpoint for epoch {epoch} to {save_path}")

                # Evaluate in environment
                env_cfg = SimpleNamespace(
                    robosuite_task=run_config.robosuite_task,
                    robosuite_robots=run_config.robosuite_robots,
                    robosuite_controller=run_config.robosuite_controller,
                    render=run_config.render,
                    robosuite_reward_shaping=run_config.robosuite_reward_shaping,
                    robosuite_control_freq=run_config.robosuite_control_freq,
                    max_env_steps=args.env_max_steps,
                    camera_obs_keys=tuple(getattr(run_config, "camera_obs_keys", tuple(args.camera_obs_keys))),
                    flip_camera_keys=tuple(getattr(run_config, "flip_camera_keys", tuple(args.flip_camera_keys))),
                    bc_cnn_keys_order=run_config.bc_cnn_keys_order,
                    bc_mlp_keys_order=run_config.bc_mlp_keys_order,
                    clip_actions=run_config.clip_actions,
                    episodes=args.env_episodes,
                    seed=run_config.seed,
                    video_dir=pathlib.Path(args.video_dir) / f"{run_name}_epoch{epoch}",
                    video_fps=args.video_fps,
                    video_camera=args.video_camera or (args.video_camera_keys[0] if args.video_camera_keys else args.camera_obs_keys[0]),
                    video_camera_keys=tuple(args.video_camera_keys) if args.video_camera_keys else tuple(args.camera_obs_keys),
                    controller_configs=getattr(run_config, "controller_configs", None),
                    has_offscreen_renderer=getattr(run_config, "has_offscreen_renderer", True),
                    has_renderer=getattr(run_config, "has_renderer", run_config.render),
                    ignore_done=getattr(run_config, "ignore_done", False),
                    camera_depths=getattr(run_config, "camera_depths", False),
                    use_camera_obs=getattr(run_config, "use_camera_obs", True),
                    bc_crop_height=getattr(run_config, "bc_crop_height", 0),
                    bc_crop_width=getattr(run_config, "bc_crop_width", 0),
                )
                env_metrics = evaluate_in_environment(run_config, enc, mlp, env_cfg, run=run)
                run.summary.update({f"epoch_{epoch}/eval_loss": eval_loss, **{f"epoch_{epoch}/{k}": v for k, v in env_metrics.items()}})
                enc.train(mode=bool(args.scratch_encoder))
                mlp.train()

            final_eval_loss = BC_MLP_train(
                run_config,
                encoder,
                action_mlp,
                train_samples,
                eval_samples,
                eval_episodes_list,
                train_encoder=bool(args.scratch_encoder),
                run=run,
                checkpoint_epochs=checkpoint_epochs,
                checkpoint_cb=on_checkpoint,
            )

            run.summary.update({"eval/loss": final_eval_loss})

            if args.save_dir:
                save_root = pathlib.Path(args.save_dir).expanduser()
                save_root.mkdir(parents=True, exist_ok=True)
                save_path = save_root / f"{run_name}_final.pt"
                torch.save(
                    {
                        "action_mlp_state_dict": action_mlp.state_dict(),
                        "encoder_state_dict": encoder.state_dict(),
                        "config": vars(run_config),
                        "epoch": run_config.bc_epochs,
                        "eval_loss": final_eval_loss,
                    },
                    save_path,
                )
                print(f"[{run.name}] Saved final checkpoint to {save_path}")

    wandb.agent(
        sweep_id,
        function=run_bc_job,
        project=args.wandb_project,
        entity=args.wandb_entity,
        count=agent_count,
    )


if __name__ == "__main__":
    main()
