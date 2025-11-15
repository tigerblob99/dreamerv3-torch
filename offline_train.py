import argparse
import os
import pathlib
import sys
from collections import defaultdict
from typing import Dict, Tuple

os.environ.setdefault("MUJOCO_GL", "osmesa")

import gym
import numpy as np
import ruamel.yaml as yaml
import torch

import tools
from dreamer import Dreamer, make_dataset

try:
    import wandb  
except ImportError:  
    wandb = None


_EXCLUDE_KEYS = {"action", "reward", "discount", "is_first", "is_terminal"}


class _EvalDataset:
    def __init__(self, batches):
        if not batches:
            raise ValueError("No evaluation episodes prepared for dataset.")
        self._batches = batches
        self._idx = 0

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return self

    def __next__(self):
        batch = self._batches[self._idx]
        self._idx = (self._idx + 1) % len(self._batches)
        return batch


def _prepare_padded_episode(episode: Dict[str, np.ndarray], batch_length: int) -> Dict[str, np.ndarray]:
    if batch_length <= 0:
        raise ValueError("batch_length must be positive for evaluation batches.")
    clean = {}
    length = None
    for key, value in episode.items():
        if key.startswith("log_"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr[None]
        if length is None:
            length = arr.shape[0]
        seq = arr[:batch_length].copy()
        if seq.shape[0] == 0:
            raise ValueError("Encountered empty episode when preparing evaluation batch.")
        clean[key] = seq
    if not clean:
        raise ValueError("Episode does not contain any usable keys for evaluation.")
    steps = next(iter(clean.values())).shape[0]
    if steps < batch_length:
        pad_len = batch_length - steps
        for key, seq in clean.items():
            pad_value = seq[-1:]
            pad = np.repeat(pad_value, pad_len, axis=0)
            clean[key] = np.concatenate([seq, pad], axis=0)
        steps = batch_length
    if "is_first" not in clean:
        flags = np.zeros(steps, dtype=bool)
        flags[0] = True
        clean["is_first"] = flags
    else:
        flags = clean["is_first"]
        flags = flags[:steps].copy()
        flags[0] = True
        if flags.shape[0] < steps:
            pad = np.zeros(steps - flags.shape[0], dtype=flags.dtype)
            flags = np.concatenate([flags, pad], axis=0)
        flags[1:] = False
        clean["is_first"] = flags
    return {key: value[None, ...] for key, value in clean.items()}


def _make_eval_dataset(eval_eps, config):
    episodes = [ep for ep in eval_eps.values() if isinstance(ep, dict)]
    if not episodes:
        raise ValueError("No evaluation episodes available for dataset creation.")
    batch_length = int(getattr(config, "batch_length", 1))
    prepared = [_prepare_padded_episode(ep, batch_length) for ep in episodes]
    return _EvalDataset(prepared)


def _infer_spaces(episode: Dict[str, np.ndarray]) -> Tuple[gym.spaces.Dict, gym.spaces.Box]:
    obs_spaces = {}
    for key, value in episode.items():
        if key in _EXCLUDE_KEYS:
            continue
        shape = value.shape[1:]
        if value.dtype == np.uint8:
            low = np.zeros(shape, dtype=np.uint8)
            high = np.full(shape, 255, dtype=np.uint8)
            obs_spaces[key] = gym.spaces.Box(low=low, high=high, dtype=np.uint8, shape=shape)
        else:
            low = np.full(shape, -np.inf, dtype=np.float32)
            high = np.full(shape, np.inf, dtype=np.float32)
            obs_spaces[key] = gym.spaces.Box(low=low, high=high, dtype=np.float32, shape=shape)
    if not obs_spaces:
        raise ValueError("Episode does not contain observable keys besides action/reward scalars.")
    obs_space = gym.spaces.Dict(obs_spaces)
    action = episode.get("action")
    if action is None:
        raise ValueError("Episode is missing the 'action' key required for offline training.")
    act_shape = action.shape[1:]
    low = np.full(act_shape, -np.inf, dtype=np.float32)
    high = np.full(act_shape, np.inf, dtype=np.float32)
    act_space = gym.spaces.Box(low=low, high=high, dtype=np.float32, shape=act_shape)
    return obs_space, act_space


def _setup_config(config):
    config.offline_updates = int(getattr(config, "offline_updates", 100000))
    base_log_every = int(getattr(config, "log_every", 1000) or 1)
    config.offline_log_every = int(getattr(config, "offline_log_every", base_log_every))
    config.offline_checkpoint_every = int(
        getattr(config, "offline_checkpoint_every", max(1, config.offline_updates // 10))
    )
    config.offline_eval_batches = int(getattr(config, "offline_eval_batches", 0))
    config.offline_eval_video_batches = int(
        getattr(config, "offline_eval_video_batches", max(1, min(2, config.offline_eval_batches or 1)))
    )
    config.offline_eval_every = int(getattr(config, "offline_eval_every", config.offline_log_every))
    if not getattr(config, "offline_traindir", None):
        raise ValueError("--offline_traindir must be provided for offline training.")
    dataset_dir = pathlib.Path(config.offline_traindir).expanduser()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Offline dataset directory not found: {dataset_dir}")
    if not getattr(config, "logdir", None):
        config.logdir = dataset_dir.parent / f"{dataset_dir.name}_offline_log"
    config.logdir = str(pathlib.Path(config.logdir).expanduser())
    config.traindir = config.traindir or dataset_dir
    config.evaldir = config.evaldir or dataset_dir
    config.log_every = config.offline_log_every
    config.eval_every = config.offline_eval_every
    if getattr(config, "eval_only", False) and config.offline_eval_batches <= 0:
        config.offline_eval_batches = 1
    return config


def _offline_eval(agent, eval_dataset, config, logger):
    if eval_dataset is None or len(eval_dataset) == 0:
        print("No evaluation episodes available; skipping eval.")
        return
    if config.offline_eval_batches <= 0:
        print("offline_eval_batches <= 0; skipping eval.")
        return

    dataset = eval_dataset
    metrics = defaultdict(list)
    video_budget = min(config.offline_eval_video_batches, config.offline_eval_batches)
    agent.eval()

    processed_batches = 0
    for batch_idx in range(config.offline_eval_batches):
        batch = next(dataset)
        with torch.no_grad():
            data = agent._wm.preprocess(batch)
            embed = agent._wm.encoder(data)
            post, _ = agent._wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )
            feat = agent._wm.dynamics.get_feat(post)

            decoded = agent._wm.heads["decoder"](feat)
            for key, dist in decoded.items():
                if key not in data:
                    continue
                recon = dist.mode()
                truth = data[key]
                mse = torch.mean((recon - truth) ** 2).item()
                metrics[f"eval/{key}_mse"].append(mse)

            reward_pred = agent._wm.heads["reward"](feat).mode()
            reward_truth = data["reward"].unsqueeze(-1)
            metrics["eval/reward_mse"].append(
                torch.mean((reward_pred - reward_truth) ** 2).item()
            )

            if "cont" in agent._wm.heads and "cont" in data:
                cont_dist = agent._wm.heads["cont"](feat)
                cont_pred = cont_dist.mean if hasattr(cont_dist, "mean") else cont_dist.mode()
                cont_truth = data["cont"].unsqueeze(-1) if data["cont"].ndim == 2 else data["cont"]
                cont_l1 = torch.mean(torch.abs(cont_pred - cont_truth)).item()
                metrics["eval/cont_l1"].append(cont_l1)

            if config.video_pred_log and batch_idx < video_budget:
                video_pred = agent._wm.video_pred(batch)
                video_pred = to_np(video_pred)
                if video_pred.ndim == 4:
                    video_pred = video_pred[None]
                logger.video(f"eval_openl/batch_{batch_idx}", video_pred)
        processed_batches += 1

    for name, values in metrics.items():
        logger.scalar(name, float(np.mean(values)))
    logger.scalar("eval/batches", float(processed_batches))
    logger.write(fps=False)


def offline_train(config):
    tools.set_seed_everywhere(config.seed)
    if getattr(config, "deterministic_run", False):
        tools.enable_deterministic_run()
    config = _setup_config(config)
    logdir = pathlib.Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir, 0)

    print(f"Loading offline episodes from {config.offline_traindir}.")
    train_eps = tools.load_episodes(config.offline_traindir, limit=config.dataset_size)
    if not train_eps:
        raise RuntimeError(f"No episodes found in {config.offline_traindir}.")
    if config.offline_evaldir:
        eval_dir = pathlib.Path(config.offline_evaldir)
        npz_count = len(list(eval_dir.glob('*.npz')))
        eval_limit = npz_count or None
        eval_eps = tools.load_episodes(config.offline_evaldir, limit=eval_limit)
    else:
        eval_eps = train_eps
    first_episode = next(iter(train_eps.values()))
    obs_space, act_space = _infer_spaces(first_episode)
    config.num_actions = getattr(act_space, "n", act_space.shape[0])

    train_dataset = None if config.eval_only else make_dataset(train_eps, config)
    eval_dataset = _make_eval_dataset(eval_eps, config)

    agent = Dreamer(obs_space, act_space, config, logger, train_dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)

    wandb_run = None
    if wandb is not None:
        try:
            wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "dreamerv3"),
                entity=os.getenv("WANDB_ENTITY"),
                name=os.getenv("WANDB_NAME"),
                config=vars(config),
                dir=str(logdir),
                mode=os.getenv("WANDB_MODE", "online"),
            )
            try:
                wandb.watch(agent, log="all", log_freq=1000)
            except Exception:
                print("Failed to watch model parameters with wandb.", flush=True)
        except Exception:
            wandb_run = None
    if wandb_run is not None:
        logger.attach_wandb(wandb, wandb_run)

    checkpoint_path = logdir / "latest.pt"
    if checkpoint_path.exists():
        print(f"Resuming from {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    if config.eval_only:
        print("Running offline evaluation only.")
        logger.step = agent._update_count
        _offline_eval(agent, eval_dataset, config, logger)
        if wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
        return

    print("Starting offline training.")
    for update in range(config.offline_updates):
        agent._train(next(train_dataset))
        agent._update_count += 1
        agent._metrics.setdefault("update_count", []).append(agent._update_count)
        logger.step = agent._update_count
        if agent._should_log(agent._update_count):
            for name, values in agent._metrics.items():
                if not values:
                    continue
                logger.scalar(name, float(np.mean(values)))
                agent._metrics[name] = []
            if config.video_pred_log:
                try:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    video_pred = to_np(video_pred)
                    if video_pred.ndim == 4:
                        video_pred = video_pred[None]
                    logger.video("eval_openl", video_pred)
                except StopIteration:
                    pass
            logger.scalar("offline_update", agent._update_count)
            logger.write(fps=False)
        if (
            config.offline_eval_every > 0
            and agent._update_count % config.offline_eval_every == 0
            and config.offline_eval_batches > 0
        ):
            _offline_eval(agent, eval_dataset, config, logger)
        if (agent._update_count % config.offline_checkpoint_every) == 0:
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                },
                checkpoint_path,
            )

    print("Offline training finished.")
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        },
        checkpoint_path,
    )
    if config.offline_eval_batches > 0:
        print("Running offline evaluation after training.")
    logger.step = agent._update_count
    _offline_eval(agent, eval_dataset, config, logger)
    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def to_np(tensor):
    return tensor.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    defaults.setdefault("eval_only", False)
    defaults.setdefault("offline_updates", 100000)
    defaults.setdefault("offline_log_every", 1000)
    defaults.setdefault("offline_checkpoint_every", 10000)
    defaults.setdefault("offline_eval_every", 0)
    defaults.setdefault("offline_eval_batches", 0)
    defaults.setdefault("offline_eval_video_batches", 1)
    defaults.setdefault("offline_traindir", "")
    defaults.setdefault("offline_evaldir", "")

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining)
    offline_train(config)
