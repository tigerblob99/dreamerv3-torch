"""Compact on-policy RL training on Robosuite using a frozen world model."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from ruamel.yaml import YAML

# Ensure repo root is importable when running this file directly.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools
from AC_RL.actor import Actor
from AC_RL.critic import Critic
from RL_finetune import _load_pretrained as _load_pretrained_finetune
from envs.robosuite_env import make_Robosuite_env
from models import WorldModel
from parallel import Parallel

try:
    import wandb
except ImportError:
    wandb = None


def _mean_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    arr = np.asarray(value)
    return float(arr.mean()) if arr.size else 0.0


def _make_envs(config, count: int, purpose: str):
    """Create Robosuite envs with Dreamer-style parallel wrappers."""
    base_seed = int(getattr(config, "seed", 0)) + (10_000 if purpose == "eval" else 0)
    task = getattr(config, "robosuite_task", "Lift")
    print(f"Creating {count} {purpose} envs (task={task})...")
    return [
        Parallel(
            lambda cfg=config, seed=base_seed + i: make_Robosuite_env(cfg, seed=seed),
            "process",
        )
        for i in range(count)
    ]


def _close_envs(envs):
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


def _to_model_obs(raw_obs):
    """Convert env obs into encoder input tensors."""
    out = {}
    for key, value in raw_obs.items():
        if key in ("is_first", "is_terminal"):
            continue
        arr = np.asarray(value)
        if key == "image":
            out[key] = arr.astype(np.uint8, copy=False)
        else:
            out[key] = arr.astype(np.float32, copy=False)
    return out


def _stack_obs(obs_list, device):
    data = {}
    for key in obs_list[0]:
        arr = np.stack([obs[key] for obs in obs_list])
        tensor = torch.as_tensor(arr, device=device).float()
        if key == "image":
            tensor = tensor / 255.0
        data[key] = tensor
    return data

def _append_transition(
    buf,
    obs,
    action,
    reward: float,
    is_first: bool,
    is_terminal: bool,
    discount: float,
    action_dim: int,
):
    for key, value in obs.items():
        buf[key].append(np.asarray(value))
    buf["action"].append(
        np.zeros(action_dim, dtype=np.float32)
        if action is None
        else np.asarray(action, dtype=np.float32)
    )
    buf["reward"].append(np.float32(reward))
    buf["discount"].append(np.float32(discount))
    buf["is_first"].append(np.float32(is_first))
    buf["is_terminal"].append(np.float32(is_terminal))


def _collect_episodes(wm, actor, envs, config, num_steps: int):
    """Collect fresh on-policy episodes and keep a t=0 placeholder transition."""
    wm.eval()
    actor.eval()

    device = config.device
    num_envs = len(envs)
    max_env_steps = int(getattr(config, "max_env_steps", 500))

    rssm_state = None
    prev_action = torch.zeros((num_envs, config.num_actions), device=device)
    is_first = torch.ones(num_envs, device=device)

    reset_promises = [env.reset() for env in envs]
    obs_list = [_to_model_obs(promise()) for promise in reset_promises]
    ep_buffers = [defaultdict(list) for _ in range(num_envs)]
    ep_steps = [0] * num_envs
    ep_success = [False] * num_envs

    for i in range(num_envs):
        _append_transition(
            ep_buffers[i],
            obs_list[i],
            action=None,
            reward=0.0,
            is_first=True,
            is_terminal=False,
            discount=1.0,
            action_dim=config.num_actions,
        )

    completed = []
    completed_success = []
    total_steps = 0

    while total_steps < num_steps:
        data = _stack_obs(obs_list, device)
        with torch.no_grad():
            embed = wm.encoder(data)
            # is_first resets only the env slots that just started a new episode.
            post, _ = wm.dynamics.obs_step(rssm_state, prev_action, embed, is_first, sample=False)
            rssm_state = post
            feat = wm.dynamics.get_feat(post)
            action_tensor = actor.generate_actions(feat, sample=True)
            if getattr(config, "clip_actions", True):
                action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
            prev_action = action_tensor

        action_np = action_tensor.detach().cpu().numpy()
        step_promises = [env.step(action_np[i]) for i, env in enumerate(envs)]
        for i, promise in enumerate(step_promises):
            raw_obs, reward, done, info = promise()
            ep_steps[i] += 1
            total_steps += 1
            ep_success[i] = ep_success[i] or bool(info.get("success", False))

            next_obs = _to_model_obs(raw_obs)
            _append_transition(
                ep_buffers[i],
                next_obs,
                action=action_np[i],
                reward=float(reward),
                is_first=False,
                is_terminal=done,
                discount=0.0 if done else 1.0,
                action_dim=config.num_actions,
            )

            if done:
                completed.append({k: np.stack(v, axis=0) for k, v in ep_buffers[i].items()})
                completed_success.append(bool(ep_success[i]))
                ep_buffers[i] = defaultdict(list)
                ep_steps[i] = 0
                ep_success[i] = False
                obs_list[i] = _to_model_obs(envs[i].reset()())
                _append_transition(
                    ep_buffers[i],
                    obs_list[i],
                    action=None,
                    reward=0.0,
                    is_first=True,
                    is_terminal=False,
                    discount=1.0,
                    action_dim=config.num_actions,
                )
                is_first[i] = 1.0
                prev_action[i] = 0.0
            else:
                obs_list[i] = next_obs
                is_first[i] = 0.0

            if total_steps >= num_steps:
                break

    # Keep partial episodes so each update always has fresh transitions.
    for i in range(num_envs):
        if len(ep_buffers[i].get("reward", ())) > 1:
            completed.append({k: np.stack(v, axis=0) for k, v in ep_buffers[i].items()})
            completed_success.append(bool(ep_success[i]))

    actor.train()
    return completed, total_steps, completed_success


def _sample_batch(episodes, config, device, batch_size: int):
    ep_dict = OrderedDict(
        (f"ep_{i:06d}", ep) for i, ep in enumerate(episodes) if len(ep.get("action", ())) > 1
    )
    episode_gen = tools.sample_episodes(
        ep_dict,
        length=int(config.batch_length),
        seed=np.random.randint(0, 2**31 - 1),
        sampling_mode=getattr(config, "dataset_episode_sampling", "shorter"),
    )
    batch = next(tools.from_generator(episode_gen, int(batch_size)))
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def _train_step(wm, actor, critic, batch, config):
    with torch.no_grad():
        data = wm.preprocess(dict(batch))
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)

    # Data format has a placeholder at t=0; learning targets start from t=1.
    rewards = data["reward"][:, 1:]
    discounts = data.get("discount")
    if discounts is None:
        discounts = config.discount * (1.0 - data["is_terminal"]).unsqueeze(-1)
    elif discounts.ndim == 2:
        discounts = discounts.unsqueeze(-1)
    discounts = discounts[:, 1:]
    actions = data["action"][:, 1:]

    # Update order is intentional: slow-target sync, then actor and critic updates.
    critic.update_slow_target()
    target, weights, values = critic.compute_targets(feat, rewards, discounts=discounts)
    actor_metrics = actor.update(feat, actions, target, weights, values)
    critic_metrics = critic.update_from_targets(feat, target, weights)

    out = {f"actor/{k}": v for k, v in actor_metrics.items()}
    out.update({f"critic/{k}": v for k, v in critic_metrics.items()})
    out.update(tools.tensorstats(rewards, "env_reward"))
    return out


def _critic_only_step(wm, critic, batch, config):
    with torch.no_grad():
        data = wm.preprocess(dict(batch))
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)

    rewards = data["reward"][:, 1:]
    discounts = data.get("discount")
    if discounts is None:
        discounts = config.discount * (1.0 - data["is_terminal"]).unsqueeze(-1)
    elif discounts.ndim == 2:
        discounts = discounts.unsqueeze(-1)
    discounts = discounts[:, 1:]

    critic.update_slow_target()
    target, weights, _ = critic.compute_targets(feat, rewards, discounts=discounts)
    critic_metrics = critic.update_from_targets(feat, target, weights)

    out = {f"critic/{k}": v for k, v in critic_metrics.items()}
    out.update(tools.tensorstats(rewards, "env_reward"))
    return out


def _critic_pretrain(wm, critic, episodes, config, logger, batch_size: int):
    pretrain_steps = int(getattr(config, "rl_critic_pretrain_steps", 0))
    if pretrain_steps <= 0:
        return
    valid_episodes = [ep for ep in episodes if len(ep.get("action", ())) > 1]
    if not valid_episodes:
        raise RuntimeError("Critic pretraining requested but no episodes were loaded.")

    pretrain_log_every = int(
        getattr(config, "rl_critic_pretrain_log_every", getattr(config, "log_every", 100))
    )
    print(
        "Starting critic pretraining: "
        f"steps={pretrain_steps} episodes={len(valid_episodes)} batch={int(batch_size)}"
    )
    metrics_acc = defaultdict(list)
    for pre_step in range(1, pretrain_steps + 1):
        batch = _sample_batch(valid_episodes, config, config.device, batch_size)
        step_metrics = _critic_only_step(wm, critic, batch, config)
        for name, value in step_metrics.items():
            metrics_acc[name].append(_mean_scalar(value))

        should_log = pretrain_log_every > 0 and (
            pre_step % pretrain_log_every == 0 or pre_step == pretrain_steps
        )
        if should_log:
            logger.step = pre_step
            logger.scalar("pretrain/step", pre_step)
            for name, values in metrics_acc.items():
                logger.scalar(f"pretrain/{name}", float(np.mean(values)))
            logger.write(fps=False)
            metrics_acc.clear()
    print("Critic pretraining complete.")


def evaluate_online(wm, policy, config, envs):
    wm.eval()
    policy.eval()

    device = config.device
    num_envs = len(envs)
    total_episodes = int(getattr(config, "eval_episodes", 0))
    max_env_steps = int(getattr(config, "max_env_steps", 500))
    if total_episodes <= 0 or num_envs <= 0:
        return {"eval_online/success_rate": 0.0, "eval_online/mean_return": 0.0}

    obs_batch = [None] * num_envs
    rssm_state = None
    prev_action = torch.zeros((num_envs, config.num_actions), device=device)
    is_first = torch.ones(num_envs, device=device)

    ep_rewards = [0.0] * num_envs
    ep_steps = [0] * num_envs
    ep_success = [False] * num_envs

    completed = 0
    assigned = 0
    returns = []
    successes = []

    for i in range(num_envs):
        if assigned >= total_episodes:
            break
        obs_batch[i] = _to_model_obs(envs[i].reset()())
        assigned += 1

    while completed < total_episodes:
        active = [i for i, obs in enumerate(obs_batch) if obs is not None]
        if not active:
            break

        template = obs_batch[active[0]]
        model_obs = [
            obs_batch[i] if obs_batch[i] is not None else {k: np.zeros_like(v) for k, v in template.items()}
            for i in range(num_envs)
        ]
        data = _stack_obs(model_obs, device)

        with torch.no_grad():
            embed = wm.encoder(data)
            post, _ = wm.dynamics.obs_step(rssm_state, prev_action, embed, is_first, sample=False)
            rssm_state = post
            feat = wm.dynamics.get_feat(post)
            action_tensor = policy(feat)
            if getattr(config, "clip_actions", True):
                action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
            prev_action = action_tensor

        action_np = action_tensor.detach().cpu().numpy()
        step_promises = {i: envs[i].step(action_np[i]) for i in active}
        for i in active:
            raw_obs, reward, done, info = step_promises[i]()
            success = bool(info.get("success", False))
            ep_rewards[i] += float(reward)
            ep_steps[i] += 1
            ep_success[i] = ep_success[i] or success

            env_done = bool(done) or success or (ep_steps[i] >= max_env_steps)
            if env_done:
                completed += 1
                returns.append(ep_rewards[i])
                successes.append(ep_success[i])
                if assigned < total_episodes:
                    obs_batch[i] = _to_model_obs(envs[i].reset()())
                    ep_rewards[i] = 0.0
                    ep_steps[i] = 0
                    ep_success[i] = False
                    assigned += 1
                    is_first[i] = 1.0
                    prev_action[i] = 0.0
                else:
                    obs_batch[i] = None
                    is_first[i] = 0.0
                    prev_action[i] = 0.0
            else:
                obs_batch[i] = _to_model_obs(raw_obs)
                is_first[i] = 0.0

    return {
        "eval_online/success_rate": float(np.mean(successes)) if successes else 0.0,
        "eval_online/mean_return": float(np.mean(returns)) if returns else 0.0,
    }


def _load_wm_from_checkpoint(wm, checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "wm" in checkpoint:
        wm.load_state_dict(checkpoint["wm"])
        return
    if "world_model" in checkpoint:
        wm.load_state_dict(checkpoint["world_model"])
        return
    state = checkpoint.get("agent_state_dict", {})
    wm_state = {k[len("_wm.") :]: v for k, v in state.items() if k.startswith("_wm.")}
    if not wm_state:
        wm_state = {
            k[len("_wm._orig_mod.") :]: v for k, v in state.items() if k.startswith("_wm._orig_mod.")
        }
    if not wm_state:
        raise KeyError("Checkpoint missing world model weights.")
    wm.load_state_dict(wm_state, strict=False)


def rl_train(config):
    tools.set_seed_everywhere(int(config.seed))
    if "cuda" not in str(config.device) or not torch.cuda.is_available():
        config.precision = 32

    logdir = pathlib.Path(getattr(config, "logdir", "logdir/rl_train"))
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
        raise ValueError("expert_dir or offline_traindir must be provided.")
    ref_eps = tools.load_episodes(ref_dir, limit=1)
    if not ref_eps:
        raise RuntimeError(f"No episodes found in {ref_dir}")
    sample_ep = next(iter(ref_eps.values()))
    obs_space, act_space = tools._define_spaces(sample_ep, config)
    config.num_actions = int(act_space.shape[0])

    wm = WorldModel(obs_space, act_space, 0, config).to(config.device)
    feat_size = (
        config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        if config.dyn_discrete
        else config.dyn_stoch + config.dyn_deter
    )
    actor = Actor(config, feat_size, config.num_actions).to(config.device)
    critic = Critic(config, feat_size).to(config.device)

    ckpt_path = getattr(config, "checkpoint", "") or str(logdir / "latest.pt")
    if ckpt_path and pathlib.Path(ckpt_path).exists():
        policy_init = str(getattr(config, "policy_init", "")).strip().lower() or (
            "checkpoint" if bool(getattr(config, "load_actor_from_checkpoint", False)) else "random"
        )
        print(f"Loading checkpoint ({policy_init}) from {ckpt_path}")
        if policy_init == "checkpoint":
            _load_pretrained_finetune(wm, actor, ckpt_path, config.device)
        else:
            _load_wm_from_checkpoint(wm, ckpt_path, config.device)
    else:
        print("[warn] No checkpoint provided; world model starts random.")

    wm.eval()
    wm.requires_grad_(False)

    total_updates = int(getattr(config, "rl_updates", getattr(config, "rl_epochs", 200)))
    collect_steps = int(getattr(config, "collect_steps", 2000))
    save_every = int(getattr(config, "save_every", 10))
    eval_every = int(getattr(config, "eval_every", 5))
    batches_per_collect = int(getattr(config, "train_batches_per_collect", 1))
    mini_batch_size = int(getattr(config, "mini_batch_size", 0)) or int(config.batch_size)
    pretrain_steps = int(getattr(config, "rl_critic_pretrain_steps", 0))

    if pretrain_steps > 0:
        pretrain_dir = getattr(config, "rl_critic_pretrain_dir", "") or ref_dir
        pretrain_limit = int(
            getattr(config, "rl_critic_pretrain_dataset_size", getattr(config, "dataset_size", 0))
            or 0
        )
        pretrain_eps = tools.load_episodes(
            pretrain_dir, limit=(pretrain_limit if pretrain_limit > 0 else None)
        )
        if not pretrain_eps:
            raise RuntimeError(
                f"Critic pretraining requested but no episodes found in {pretrain_dir}"
            )
        _critic_pretrain(
            wm,
            critic,
            list(pretrain_eps.values()),
            config,
            logger,
            batch_size=mini_batch_size,
        )

    num_collect_envs = int(getattr(config, "num_collect_envs", getattr(config, "envs", 4)))
    num_eval_envs = int(getattr(config, "num_eval_envs", getattr(config, "num_envs", 4)))
    collect_envs = _make_envs(config, num_collect_envs, "collect")
    eval_envs = _make_envs(config, num_eval_envs, "eval")

    global_env_steps = 0
    restart_every = int(getattr(config, "env_restart_every", 0) or 0)
    next_env_restart = None
    if restart_every > 0:
        next_env_restart = int(global_env_steps + restart_every)
        print(
            "[startup] env restarts enabled: "
            f"every={restart_every} next_at={int(next_env_restart)}"
        )

    try:
        for update in range(1, total_updates + 1):
            print(f"\n=== Update {update}/{total_updates}: collect {collect_steps} steps ===")
            episodes, steps_collected, success_flags = _collect_episodes(
                wm, actor, collect_envs, config, num_steps=collect_steps
            )
            if not episodes:
                continue
            global_env_steps += int(steps_collected)

            # Recreate worker processes periodically to avoid long-run simulator instability.
            if next_env_restart is not None and global_env_steps >= next_env_restart:
                print(f"[runtime] restarting envs at step={global_env_steps}.")
                _close_envs(collect_envs + eval_envs)
                collect_envs = _make_envs(config, num_collect_envs, "collect")
                eval_envs = _make_envs(config, num_eval_envs, "eval")
                next_env_restart = int(global_env_steps + restart_every)

            metrics_acc = defaultdict(list)
            batch = None
            for _ in range(batches_per_collect):
                batch = _sample_batch(episodes, config, config.device, mini_batch_size)
                step_metrics = _train_step(wm, actor, critic, batch, config)
                for name, value in step_metrics.items():
                    metrics_acc[name].append(_mean_scalar(value))

            train_metrics = {k: float(np.mean(v)) for k, v in metrics_acc.items()}
            ep_rewards = [float(np.asarray(ep["reward"][1:]).sum()) for ep in episodes]
            ep_lengths = [max(0, int(len(ep["reward"]) - 1)) for ep in episodes]

            logger.step = global_env_steps
            logger.scalar("update", update)
            logger.scalar("collect/episodes", len(episodes))
            logger.scalar("collect/steps", int(steps_collected))
            logger.scalar("collect/mean_reward", float(np.mean(ep_rewards)) if ep_rewards else 0.0)
            logger.scalar("collect/mean_length", float(np.mean(ep_lengths)) if ep_lengths else 0.0)
            logger.scalar(
                "collect/success_rate", float(np.mean(success_flags)) if success_flags else 0.0
            )
            logger.scalar("train/train_batches_per_collect", float(batches_per_collect))
            logger.scalar("train/mini_batch_size", float(mini_batch_size))
            for name, value in train_metrics.items():
                logger.scalar(f"train/{name}", value)
            logger.write(fps=False)

            if eval_every > 0 and update % eval_every == 0:
                print(f"Evaluating online at update {update}...")
                for name, value in evaluate_online(wm, actor.actionMLP, config, eval_envs).items():
                    logger.scalar(name, value)
                logger.step = global_env_steps
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
        _close_envs(collect_envs + eval_envs)
        if run is not None and wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass

    print("RL online training finished.")


def _parse_config(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--configs", nargs="+", default=[])
    pre.add_argument("--env_config", type=str, default=None)
    known, remaining = pre.parse_known_args(argv)

    cfg_path = pathlib.Path(__file__).resolve().parent.parent / "configs.yaml"
    configs = YAML(typ="safe", pure=True).load(cfg_path.read_text())

    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    defaults = {}
    for name in ["defaults", *known.configs]:
        merge(defaults, configs[name])

    env_defaults = configs.get(known.env_config, {}) if known.env_config else {}
    complex_defaults = {}
    for k, v in env_defaults.items():
        if isinstance(v, (dict, list, tuple)):
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
    defaults.setdefault("policy_init", "random")
    defaults.setdefault("train_batches_per_collect", 1)
    defaults.setdefault("mini_batch_size", 0)
    defaults.setdefault("rl_critic_pretrain_steps", 0)
    defaults.setdefault("rl_critic_pretrain_log_every", defaults.get("log_every", 100))
    defaults.setdefault("rl_critic_pretrain_dataset_size", defaults.get("dataset_size", 0))
    defaults.setdefault("rl_critic_pretrain_dir", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--env_config", type=str, default=known.env_config)
    for key, value in sorted(defaults.items()):
        if key in complex_defaults:
            continue
        typ = tools.args_type(value)
        parser.add_argument(f"--{key}", type=typ, default=typ(value))

    config = parser.parse_args(remaining)
    for key, value in complex_defaults.items():
        setattr(config, key, value)
    if known.env_config:
        setattr(config, "env_config", known.env_config)
        if "controller_configs" in env_defaults:
            setattr(config, "controller_configs", env_defaults["controller_configs"])
    return config


if __name__ == "__main__":
    rl_train(_parse_config())
