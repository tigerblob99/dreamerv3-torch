"""On-policy online RL on Robosuite with fresh rollouts only.

Design goals:
- Use a pre-trained world model as a frozen latent encoder/dynamics module.
- Collect trajectories with the current policy directly in the real environment.
- Compute lambda-return targets from dense environment rewards.
- Update actor first, then critic, and discard collected data after each update.
"""
# Run:
# python AC_RL/RL_train.py --configs rl_train --env_config can_env_eval --policy_init checkpoint --train_batches_per_collect 8 --mini_batch_size 64

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections import OrderedDict, defaultdict
from types import SimpleNamespace

import numpy as np
import ruamel.yaml as yaml
from ruamel.yaml import YAML
import torch

# Ensure repo root is importable
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools
from models import WorldModel
from AC_RL.actor import Actor
from AC_RL.critic import Critic
from RL_finetune import _load_pretrained as _load_pretrained_finetune
from joint_train import EnvWorker, evaluate_online
from parallel import Parallel
from bc_mlp.BC_MLP_eval import _prepare_obs

try:
    import wandb
except ImportError:
    wandb = None


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu().item())
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _load_world_model_only(world_model: WorldModel, checkpoint_path: str, device):
    """Load only world model weights from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "wm" in checkpoint:
        world_model.load_state_dict(checkpoint["wm"])
        return
    if "world_model" in checkpoint:
        world_model.load_state_dict(checkpoint["world_model"])
        return
    if "agent_state_dict" in checkpoint:
        wm_state = {
            k[len("_wm.") :]: v
            for k, v in checkpoint["agent_state_dict"].items()
            if k.startswith("_wm.")
        }
        if not wm_state:
            wm_state = {
                k[len("_wm._orig_mod.") :]: v
                for k, v in checkpoint["agent_state_dict"].items()
                if k.startswith("_wm._orig_mod.")
            }
        if wm_state:
            world_model.load_state_dict(wm_state, strict=False)
            return
    raise KeyError("Checkpoint missing world model weights.")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _make_env_workers(config, num_envs, purpose="collect"):
    crop_h = int(getattr(config, "image_crop_height", 0))
    crop_w = int(getattr(config, "image_crop_width", 0))
    image_hw = (
        (crop_h, crop_w)
        if (crop_h > 0 and crop_w > 0)
        else tuple(getattr(config, "size", (84, 84)))
    )

    env_config = SimpleNamespace(
        robosuite_task=getattr(config, "robosuite_task", "PickPlaceCan"),
        robosuite_robots=getattr(config, "robosuite_robots", ["Panda"]),
        robosuite_controller=getattr(config, "robosuite_controller", "OSC_POSE"),
        robosuite_reward_shaping=getattr(config, "robosuite_reward_shaping", True),
        robosuite_control_freq=getattr(config, "robosuite_control_freq", 20),
        max_env_steps=getattr(config, "max_env_steps", 500),
        ignore_done=getattr(config, "ignore_done", False),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_depths=False,
        camera_obs_keys=tuple(config.camera_obs_keys),
        seed=config.seed,
        render=False,
    )
    if hasattr(config, "controller_configs"):
        env_config.controller_configs = config.controller_configs

    print(f"Creating {num_envs} {purpose} envs (task={env_config.robosuite_task})...")
    return [
        Parallel(lambda cfg=env_config, hw=image_hw: EnvWorker(cfg, hw), "process")
        for _ in range(num_envs)
    ]


def _obs_to_encoder_input(raw_obs, config, wm):
    return _prepare_obs(
        raw_obs,
        cnn_keys_order=config.bc_cnn_keys_order,
        mlp_keys_order=config.bc_mlp_keys_order,
        camera_keys=config.camera_obs_keys,
        flip_keys=config.flip_camera_keys,
        crop_height=config.image_crop_height if config.image_crop_height > 0 else None,
        crop_width=config.image_crop_width if config.image_crop_width > 0 else None,
        config=config,
    )


def _stack_obs_batch(obs_list, config, wm, device):
    batch = defaultdict(list)
    for obs in obs_list:
        for k, v in obs.items():
            batch[k].append(v)

    data = {}
    for k, v_list in batch.items():
        arr = np.stack(v_list)
        tensor = torch.as_tensor(arr, device=device).float()
        if k == "image":
            tensor = tensor / 255.0
            if (
                getattr(config, "image_standardize", False)
                and wm._dataset_image_mean is not None
            ):
                tensor = (tensor - wm._dataset_image_mean) / wm._dataset_image_std
        data[k] = tensor
    return data


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_episodes(
    wm,
    actor,
    envs,
    config,
    num_steps: int,
    *,
    sample: bool = True,
    random_actions: bool = False,
):
    wm.eval()
    actor.eval()

    device = config.device
    num_envs = len(envs)
    max_env_steps = int(getattr(config, "max_env_steps", 500))

    rssm_state = None
    prev_action = torch.zeros((num_envs, config.num_actions), device=device)
    is_first = torch.ones(num_envs, device=device)

    ep_buffers = [defaultdict(list) for _ in range(num_envs)]
    ep_steps = [0] * num_envs
    ep_success = [False] * num_envs
    completed = []
    completed_success = []
    total_steps_collected = 0

    obs_list = [None] * num_envs
    for i in range(num_envs):
        raw_obs = envs[i].reset()()
        obs_list[i] = _obs_to_encoder_input(raw_obs, config, wm)
        _append_transition(
            ep_buffers[i],
            obs_list[i],
            action=None,
            reward=0.0,
            is_first_flag=True,
            is_terminal_flag=False,
            discount=1.0,
            action_dim=config.num_actions,
        )

    while total_steps_collected < num_steps:
        data = _stack_obs_batch(obs_list, config, wm, device)

        with torch.no_grad():
            embed = wm.encoder(data)
            post, _ = wm.dynamics.obs_step(
                rssm_state, prev_action, embed, is_first, sample=False
            )
            rssm_state = post
            feat = wm.dynamics.get_feat(post)

            if random_actions:
                action_tensor = (
                    2.0 * torch.rand((num_envs, config.num_actions), device=device) - 1.0
                )
            else:
                action_tensor = actor.generate_actions(feat, sample=sample)
            if getattr(config, "clip_actions", True):
                action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
            prev_action = action_tensor

        action_np = action_tensor.cpu().numpy()
        promises = [envs[i].step(action_np[i]) for i in range(num_envs)]

        for i, promise in enumerate(promises):
            obs, reward, done, info, success = promise()
            ep_steps[i] += 1
            total_steps_collected += 1
            ep_success[i] = ep_success[i] or success

            processed_obs = _obs_to_encoder_input(obs, config, wm)
            env_done = done or success or (ep_steps[i] >= max_env_steps)

            _append_transition(
                ep_buffers[i],
                processed_obs,
                action=action_np[i],
                reward=float(reward),
                is_first_flag=False,
                is_terminal_flag=env_done,
                discount=0.0 if env_done else 1.0,
            )

            if env_done:
                episode = _finalise_episode(ep_buffers[i])
                completed.append(episode)
                completed_success.append(bool(ep_success[i]))

                raw_obs = envs[i].reset()()
                obs_list[i] = _obs_to_encoder_input(raw_obs, config, wm)
                ep_buffers[i] = defaultdict(list)
                _append_transition(
                    ep_buffers[i],
                    obs_list[i],
                    action=None,
                    reward=0.0,
                    is_first_flag=True,
                    is_terminal_flag=False,
                    discount=1.0,
                    action_dim=config.num_actions,
                )
                ep_steps[i] = 0
                ep_success[i] = False
                is_first[i] = 1.0
                prev_action[i] = 0.0
            else:
                obs_list[i] = processed_obs
                is_first[i] = 0.0

    # Include in-progress episodes so each update always has fresh data.
    for i in range(num_envs):
        if len(ep_buffers[i].get("reward", ())) > 1:
            completed.append(_finalise_episode(ep_buffers[i]))
            completed_success.append(bool(ep_success[i]))

    actor.train()
    return completed, total_steps_collected, completed_success


def _append_transition(
    buf,
    obs,
    *,
    action,
    reward,
    is_first_flag,
    is_terminal_flag,
    discount,
    action_dim: int | None = None,
):
    for k, v in obs.items():
        if k in ("is_first", "is_terminal"):
            continue
        buf[k].append(np.asarray(v))

    buf["is_first"].append(np.array(is_first_flag, dtype=np.float32))
    buf["is_terminal"].append(np.array(is_terminal_flag, dtype=np.float32))
    buf["reward"].append(np.array(reward, dtype=np.float32))
    buf["discount"].append(np.array(discount, dtype=np.float32))

    if action is None:
        if action_dim is None:
            raise ValueError(
                "action_dim is required for reset transition placeholder action"
            )
        buf["action"].append(np.zeros(int(action_dim), dtype=np.float32))
    else:
        buf["action"].append(np.asarray(action, dtype=np.float32))


def _finalise_episode(buf):
    return {k: np.stack(v, axis=0) for k, v in buf.items()}


def _sample_fresh_batch(episodes, config, device, *, batch_size: int | None = None):
    episodes_dict = OrderedDict()
    for idx, episode in enumerate(episodes):
        if len(episode.get("action", ())) > 1:
            episodes_dict[f"ep_{idx:06d}"] = episode
    if not episodes_dict:
        raise RuntimeError("No valid collected episodes available for fresh batch sampling.")

    sample_seed = np.random.randint(0, 2**31 - 1)
    episode_gen = tools.sample_episodes(
        episodes_dict, length=int(config.batch_length), seed=sample_seed
    )
    effective_batch_size = int(batch_size or config.batch_size)
    batch_np = next(tools.from_generator(episode_gen, effective_batch_size))
    return {k: torch.as_tensor(v, device=device) for k, v in batch_np.items()}


def _train_step(
    wm,
    actor,
    critic,
    batch,
    config,
):
    """One on-policy train step from fresh real environment trajectories."""
    with torch.no_grad():
        data = wm.preprocess(dict(batch))
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat_real = wm.dynamics.get_feat(post)  # (B, T, D)

    rewards = data["reward"][:, 1:]
    discounts = data.get("discount")
    if discounts is None:
        discounts = config.discount * (1.0 - data["is_terminal"]).unsqueeze(-1)
    else:
        if discounts.ndim == 2:
            discounts = discounts.unsqueeze(-1)
    discounts = discounts[:, 1:]

    # Required update order.
    critic.update_slow_target()
    target, weights, value_seq = critic.compute_targets(
        feat_real, rewards, discounts=discounts
    )

    actions = data["action"][:, 1:]
    actor_metrics = actor.update(feat_real, actions, target, weights, value_seq)
    critic_metrics = critic.update_from_targets(feat_real, target, weights)

    metrics = {}
    metrics.update({f"actor/{k}": v for k, v in actor_metrics.items()})
    metrics.update({f"critic/{k}": v for k, v in critic_metrics.items()})
    metrics.update(tools.tensorstats(rewards, "env_reward"))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def rl_train(config):
    tools.set_seed_everywhere(int(config.seed))

    if "cuda" not in str(config.device) or not torch.cuda.is_available():
        config.precision = 32

    logdir = pathlib.Path(config.logdir or "logdir/rl_train")
    logdir.mkdir(parents=True, exist_ok=True)
    config.logdir = str(logdir)
    logger = tools.Logger(logdir, 0)

    run = None
    if wandb is not None:
        try:
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "RL_online"),
                entity=os.getenv("WANDB_ENTITY"),
                name=os.getenv("WANDB_NAME"),
                config=vars(config),
                dir=str(logdir),
                mode=os.getenv("WANDB_MODE", "online"),
            )
            logger.attach_wandb(wandb, run)
        except Exception:
            run = None

    ref_dir = getattr(config, "expert_dir", "") or getattr(config, "offline_traindir", "")
    if not ref_dir:
        raise ValueError(
            "expert_dir or offline_traindir must be provided to infer obs/act shapes"
        )

    ref_eps = tools.load_episodes(ref_dir, limit=1)
    if not ref_eps:
        raise RuntimeError(f"No episodes found in {ref_dir} to infer spaces")

    sample_ep = next(iter(ref_eps.values()))
    obs_space, act_space = tools._define_spaces(sample_ep, config)
    config.num_actions = act_space.shape[0]

    wm = WorldModel(obs_space, act_space, 0, config).to(config.device)
    feat_size = (
        config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        if config.dyn_discrete
        else config.dyn_stoch + config.dyn_deter
    )
    actor = Actor(config, feat_size, config.num_actions).to(config.device)
    critic = Critic(config, feat_size).to(config.device)

    ckpt_path = getattr(config, "checkpoint", "")
    if not ckpt_path:
        candidate = logdir / "latest.pt"
        if candidate.exists():
            ckpt_path = str(candidate)

    if ckpt_path:
        print(f"Loading checkpoint from {ckpt_path}")
        policy_init = str(getattr(config, "policy_init", "")).strip().lower()
        if not policy_init:
            policy_init = (
                "checkpoint"
                if bool(getattr(config, "load_actor_from_checkpoint", False))
                else "random"
            )

        if policy_init == "checkpoint":
            _load_pretrained_finetune(wm, actor, ckpt_path, config.device)
        elif policy_init == "random":
            checkpoint = torch.load(ckpt_path, map_location=config.device)
            if "wm" in checkpoint:
                wm.load_state_dict(checkpoint["wm"])
            elif "world_model" in checkpoint:
                wm.load_state_dict(checkpoint["world_model"])
            elif "agent_state_dict" in checkpoint:
                wm_state = {
                    k[len("_wm.") :]: v
                    for k, v in checkpoint["agent_state_dict"].items()
                    if k.startswith("_wm.")
                }
                if not wm_state:
                    wm_state = {
                        k[len("_wm._orig_mod.") :]: v
                        for k, v in checkpoint["agent_state_dict"].items()
                        if k.startswith("_wm._orig_mod.")
                    }
                if not wm_state:
                    raise KeyError("Checkpoint missing world model weights.")
                wm.load_state_dict(wm_state, strict=False)
            else:
                raise KeyError("Checkpoint missing world model weights.")
        else:
            raise ValueError(
                f"Unknown policy_init='{policy_init}'. Use 'random' or 'checkpoint'."
            )
    else:
        print("[warn] No checkpoint provided; world model will be randomly initialized.")

    # Keep world model frozen for RL.
    wm.eval()
    wm.requires_grad_(False)

    num_collect_envs = int(getattr(config, "num_collect_envs", getattr(config, "envs", 4)))
    num_eval_envs = int(getattr(config, "num_eval_envs", getattr(config, "num_envs", 4)))
    collect_envs = _make_env_workers(config, num_collect_envs, purpose="collect")
    eval_envs = _make_env_workers(config, num_eval_envs, purpose="eval")

    total_updates = int(getattr(config, "rl_updates", getattr(config, "rl_epochs", 200)))
    collect_steps = int(getattr(config, "collect_steps", 2000))
    save_every = int(getattr(config, "save_every", 10))
    eval_every = int(getattr(config, "eval_every", 5))
    train_batches_per_collect = int(getattr(config, "train_batches_per_collect", 1))
    mini_batch_size = int(getattr(config, "mini_batch_size", 0))
    if mini_batch_size <= 0:
        mini_batch_size = int(config.batch_size)

    global_env_steps = 0

    try:
        for update in range(1, total_updates + 1):
            print(
                f"\n=== Update {update}/{total_updates}: collect {collect_steps} steps ==="
            )

            episodes, steps_collected, success_flags = collect_episodes(
                wm,
                actor,
                collect_envs,
                config,
                num_steps=collect_steps,
                sample=True,
                random_actions=False,
            )
            global_env_steps += int(steps_collected)

            if not episodes:
                print("No episodes collected; skipping this update.")
                continue

            actor.train()
            batch = None
            metrics_buffer = defaultdict(list)
            for _ in range(train_batches_per_collect):
                batch = _sample_fresh_batch(
                    episodes, config, config.device, batch_size=mini_batch_size
                )
                step_metrics = _train_step(wm, actor, critic, batch, config)
                for k, v in step_metrics.items():
                    metrics_buffer[k].append(_to_float(v))

            step_metrics = {
                k: float(np.mean(v)) if len(v) > 0 else 0.0
                for k, v in metrics_buffer.items()
            }
            step_metrics["train_batches_per_collect"] = float(train_batches_per_collect)
            step_metrics["mini_batch_size"] = float(mini_batch_size)

            ep_rewards = [float(np.asarray(ep["reward"][1:]).sum()) for ep in episodes]
            ep_lengths = [max(0, int(len(ep["reward"]) - 1)) for ep in episodes]
            collect_metrics = {
                "collect/episodes": len(episodes),
                "collect/steps": int(steps_collected),
                "collect/mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
                "collect/mean_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
                "collect/success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
            }

            logger.step = global_env_steps
            logger.scalar("update", update)
            for k, v in collect_metrics.items():
                logger.scalar(k, v)
            for k, v in step_metrics.items():
                logger.scalar(f"train/{k}", _to_float(v))

            if (
                getattr(config, "video_pred_log", False)
                and eval_every > 0
                and update % eval_every == 0
            ):
                try:
                    with torch.no_grad():
                        video_pred = wm.video_pred(batch)
                    video_np = video_pred.detach().cpu().numpy()
                    if video_np.ndim == 4:
                        video_np = video_np[None]
                    logger.video("rl_openl", video_np)
                except Exception as exc:
                    print(f"[video_pred] failed: {exc}")

            logger.write(fps=False)

            if eval_every > 0 and update % eval_every == 0:
                print(f"Evaluating online at update {update}...")
                online_metrics = evaluate_online(
                    wm,
                    actor.actionMLP,
                    config,
                    global_env_steps,
                    run,
                    eval_envs,
                )
                logger.step = global_env_steps
                for name, value in online_metrics.items():
                    logger.scalar(name, value)
                logger.write(fps=False)
                actor.train()

            if save_every > 0 and update % save_every == 0:
                torch.save(
                    {
                        "wm": wm.state_dict(),
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "update": update,
                        "global_env_steps": global_env_steps,
                        "config": vars(config),
                    },
                    logdir / "rl_latest.pt",
                )
                print(f"Saved checkpoint at update {update}")

    finally:
        print("Closing environments...")
        for env in collect_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass
        if run is not None and wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass

    print("RL online training finished.")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _load_env_block(name: str | None, configs):
    if not name:
        return {}
    if name not in configs:
        raise KeyError(f"Config block '{name}' not found in configs.yaml")
    block = configs[name] or {}
    if not hasattr(block, "items"):
        raise TypeError(f"Config block '{name}' must be a mapping.")
    return block


def _parse_config(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--configs", nargs="+")
    pre_parser.add_argument("--env_config", type=str, default=None)
    args, remaining = pre_parser.parse_known_args(argv)

    cfg_path = pathlib.Path(__file__).resolve().parent.parent / "configs.yaml"
    if hasattr(yaml, "safe_load"):
        try:
            configs = yaml.safe_load(cfg_path.read_text())
        except AttributeError:
            parser = YAML(typ="safe", pure=True)
            configs = parser.load(cfg_path.read_text())
    else:
        parser = YAML(typ="safe", pure=True)
        configs = parser.load(cfg_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *(args.configs or [])]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    complex_defaults = {}
    env_defaults = {}
    if args.env_config:
        print(f"Loading env config block: {args.env_config}")
        env_defaults = _load_env_block(args.env_config, configs)
        if env_defaults:
            for k, v in env_defaults.items():
                if hasattr(v, "items") or isinstance(v, (list, tuple)):
                    complex_defaults[k] = v
                else:
                    defaults[k] = v

    defaults.setdefault("rl_updates", defaults.get("rl_epochs", 200))
    defaults.setdefault("collect_steps", 2000)
    defaults.setdefault("num_collect_envs", defaults.get("envs", 4))
    defaults.setdefault("num_eval_envs", defaults.get("num_envs", 4))
    defaults.setdefault("eval_episodes", defaults.get("eval_episode_num", 10))
    defaults.setdefault("max_env_steps", defaults.get("time_limit", 500))
    defaults.setdefault("clip_actions", True)
    defaults.setdefault("checkpoint", "")
    defaults.setdefault("expert_dir", defaults.get("offline_traindir", ""))
    defaults.setdefault("num_workers", 0)
    defaults.setdefault("eval_only", False)

    defaults.setdefault("load_actor_from_checkpoint", False)
    defaults.setdefault("policy_init", "random")
    defaults.setdefault("train_batches_per_collect", 1)
    defaults.setdefault("mini_batch_size", 0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--env_config", type=str, default=None)

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        if key in complex_defaults:
            continue
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    config = parser.parse_args(remaining)
    for k, v in complex_defaults.items():
        setattr(config, k, v)
    if args.env_config:
        setattr(config, "env_config", args.env_config)
        if "controller_configs" in env_defaults:
            setattr(config, "controller_configs", env_defaults["controller_configs"])
    return config


if __name__ == "__main__":
    cfg = _parse_config()
    rl_train(cfg)
