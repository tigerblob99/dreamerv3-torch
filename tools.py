import datetime
import collections
import io
import os
import json
import pathlib
import re
import time
import random

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter
import gym


to_np = lambda x: x.detach().cpu().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step
        self._wandb_module = None
        self._wandb_run = None

    def attach_wandb(self, wandb_module, wandb_run):
        if wandb_module is None or wandb_run is None:
            return
        self._wandb_module = wandb_module
        self._wandb_run = wandb_run
        try:
            # Use environment steps as the canonical x-axis for all metrics.
            self._wandb_run.define_metric("env_step")
            self._wandb_run.define_metric("*", step_metric="env_step", step_sync=True)
        except Exception:
            pass

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = self._prepare_video_array(value)

    def write(self, fps=False, step=None):
        if step is None:
            step = self.step
        step = int(step)
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        wandb_payload = {} if self._wandb_ready() else None
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
            if wandb_payload is not None:
                wandb_payload[name] = value
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
            if wandb_payload is not None:
                wandb_payload[name] = self._wandb_module.Image(value)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            writer_video, wandb_video = self._format_video_for_logging(value)
            self._writer.add_video(name, writer_video, step, 16)
            if wandb_payload is not None:
                wandb_payload[name] = self._wandb_module.Video(
                    wandb_video, fps=16, format="mp4"
                )

        self._writer.flush()
        if wandb_payload:
            wandb_payload.setdefault("env_step", step)
            self._wandb_run.log(wandb_payload, step=step)
            # Keep wandb's internal global step aligned to env steps for hooks like wandb.watch.
            try:
                self._wandb_run.step = step
            except Exception:
                pass
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)
        if self._wandb_ready():
            step = int(step)
            self._wandb_run.log({name: float(value), "env_step": step}, step=step)
            try:
                self._wandb_run.step = step
            except Exception:
                pass

    def offline_video(self, name, value, step):
        value = self._prepare_video_array(value)
        writer_video, wandb_video = self._format_video_for_logging(value)
        self._writer.add_video(name, writer_video, step, 16)
        if self._wandb_ready():
            step = int(step)
            self._wandb_run.log(
                {
                    name: self._wandb_module.Video(wandb_video, fps=16, format="mp4"),
                    "env_step": step,
                },
                step=step,
            )
            try:
                self._wandb_run.step = step
            except Exception:
                pass

    def _prepare_video_array(self, value):
        value = np.array(value)
        if value.ndim == 4:
            value = value[None]
        if value.ndim != 5:
            raise ValueError(
                f"Video tensors must have shape [B, T, H, W, C]; got {value.shape}"
            )
        *_, C = value.shape
        if C > 3 and C % 3 == 0:
            cams = C // 3
            parts = np.split(value, cams, axis=-1)
            value = np.concatenate(parts, axis=3)
        return value

    def _format_video_for_logging(self, value):
        if np.issubdtype(value.dtype, np.floating):
            video_uint8 = np.clip(255 * value, 0, 255).astype(np.uint8)
        else:
            video_uint8 = value.astype(np.uint8)
        B, T, H, W, C = video_uint8.shape
        writer_video = (
            video_uint8.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        )
        wandb_video = writer_video[0]
        return writer_video, wandb_video

    def _wandb_ready(self):
        return self._wandb_module is not None and self._wandb_run is not None


def _as_clean_episode_arrays(episode):
    clean = {}
    for key, value in episode.items():
        if key.startswith("log_"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr[None]
        clean[key] = arr
    if not clean:
        return None
    steps = next(iter(clean.values())).shape[0]
    if steps < 2:
        return None
    for key, value in clean.items():
        if value.shape[0] != steps:
            raise ValueError(f"Inconsistent episode length for key '{key}'.")
    return clean


def _episode_signature(episode):
    keys = []
    for key, value in episode.items():
        arr = np.asarray(value)
        keys.append((key, arr.shape[1:], arr.dtype.str))
    return tuple(sorted(keys))


def _parse_store_float_dtype(name):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dataset_store_float_dtype: {name}")
    return mapping[key]


def _parse_episode_sampling_mode(name):
    key = str(name).strip().lower()
    aliases = {
        "shorter": "shorter",
        "short": "shorter",
        "inverse": "shorter",
        "inverse_transition": "shorter",
        "inverse_transitions": "shorter",
        "uniform": "uniform",
        "equal": "uniform",
    }
    if key not in aliases:
        raise ValueError(
            "Unsupported dataset_episode_sampling mode: "
            f"{name}. Expected one of: 'shorter', 'uniform'."
        )
    return aliases[key]


class ReplayStoreDataset:
    def __init__(self, replay_store, batch_size, batch_length, device=None):
        self._store = replay_store
        self._batch_size = int(batch_size)
        self._batch_length = int(batch_length)
        self._device = device
        if self._batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")
        if self._batch_length <= 0:
            raise ValueError(f"batch_length must be positive; got {batch_length}.")

    def __iter__(self):
        return self

    def __next__(self):
        return self._store.sample_batch(
            self._batch_size, self._batch_length, device=self._device
        )


class _BaseReplayStore:
    def __init__(self, config, name, allow_cpu_fallback):
        self._name = str(name)
        self._allow_cpu_fallback = bool(allow_cpu_fallback)
        self._episodes = collections.OrderedDict()
        self._ref_signature = None
        self._warned_incompatible = False
        self._rng = np.random.RandomState(int(getattr(config, "seed", 0)))
        self._dataset_limit = int(getattr(config, "dataset_size", 0) or 0)
        self._oom_fallbacks = 0
        self._last_sample_ms = 0.0
        self._last_spill_copy_ms = 0.0

        self._store_float_dtype = _parse_store_float_dtype(
            getattr(config, "dataset_store_float_dtype", "float32")
        )
        self._episode_sampling_mode = _parse_episode_sampling_mode(
            getattr(config, "dataset_episode_sampling", "shorter")
        )
        store_image_dtype = str(
            getattr(config, "dataset_store_image_dtype", "uint8")
        ).lower()
        if store_image_dtype != "uint8":
            raise ValueError(
                "Only dataset_store_image_dtype='uint8' is supported for replay store."
            )
        self._store_image_dtype = torch.uint8

        self._gpu_device = torch.device(getattr(config, "device", "cuda:0"))
        self._gpu_enabled = (
            self._gpu_device.type == "cuda" and torch.cuda.is_available()
        )
        if not self._gpu_enabled and not self._allow_cpu_fallback:
            raise ValueError(
                f"{self.__class__.__name__} requires CUDA device in config.device."
            )
        if not self._gpu_enabled and self._allow_cpu_fallback:
            print(
                f"[{self.__class__.__name__}] CUDA unavailable; storing all data in CPU spill."
            )

        self._gpu_transitions = 0
        self._spill_transitions = 0
        self._gpu_bytes = 0
        self._spill_bytes = 0

    @staticmethod
    def _is_oom_error(err):
        return "out of memory" in str(err).lower()

    @staticmethod
    def _tensor_nbytes(tensor):
        return int(tensor.numel() * tensor.element_size())

    def _storage_dtype_for(self, key, arr):
        if key == "image":
            return self._store_image_dtype
        if key in ("is_first", "is_terminal"):
            return torch.bool
        if np.issubdtype(arr.dtype, np.bool_):
            return torch.bool
        if np.issubdtype(arr.dtype, np.floating):
            return self._store_float_dtype
        if np.issubdtype(arr.dtype, np.integer):
            return self._store_float_dtype
        raise NotImplementedError(f"Unsupported dtype for key '{key}': {arr.dtype}")

    def _tensorize_episode(self, episode, device):
        tensors = {}
        nbytes = 0
        for key, arr in episode.items():
            dtype = self._storage_dtype_for(key, arr)
            tensor = torch.as_tensor(arr, dtype=dtype)
            if device is not None:
                tensor = tensor.to(device=device, non_blocking=False)
            tensors[key] = tensor
            nbytes += self._tensor_nbytes(tensor)
        return tensors, nbytes

    def _prepare_episode(self, episode):
        clean = _as_clean_episode_arrays(episode)
        if clean is None:
            return None
        sig = _episode_signature(clean)
        if self._ref_signature is None:
            self._ref_signature = sig
        elif sig != self._ref_signature:
            if not self._warned_incompatible:
                print(
                    f"[{self.__class__.__name__}] Dropping incompatible episode in {self._name}."
                )
                self._warned_incompatible = True
            return None
        return clean

    def _remove_episode(self, episode_id):
        meta = self._episodes.pop(episode_id, None)
        if meta is None:
            return
        if meta["location"] == "gpu":
            self._gpu_transitions -= meta["transitions"]
            self._gpu_bytes -= meta["bytes"]
        else:
            self._spill_transitions -= meta["transitions"]
            self._spill_bytes -= meta["bytes"]

    def _store_episode_meta(self, episode_id, data, location, steps, nbytes):
        self._remove_episode(episode_id)
        transitions = int(max(steps - 1, 0))
        meta = {
            "id": episode_id,
            "data": data,
            "location": location,
            "steps": int(steps),
            "transitions": transitions,
            "bytes": int(nbytes),
        }
        self._episodes[episode_id] = meta
        if location == "gpu":
            self._gpu_transitions += transitions
            self._gpu_bytes += int(nbytes)
        else:
            self._spill_transitions += transitions
            self._spill_bytes += int(nbytes)

    def add_episode(self, episode_id, episode):
        clean = self._prepare_episode(episode)
        if clean is None:
            return False
        steps = next(iter(clean.values())).shape[0]

        if self._gpu_enabled:
            try:
                data, nbytes = self._tensorize_episode(clean, self._gpu_device)
                self._store_episode_meta(episode_id, data, "gpu", steps, nbytes)
            except RuntimeError as err:
                if not self._allow_cpu_fallback or not self._is_oom_error(err):
                    raise
                self._oom_fallbacks += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                data, nbytes = self._tensorize_episode(clean, torch.device("cpu"))
                self._store_episode_meta(episode_id, data, "spill", steps, nbytes)
        else:
            data, nbytes = self._tensorize_episode(clean, torch.device("cpu"))
            self._store_episode_meta(episode_id, data, "spill", steps, nbytes)

        self.evict_to_limit(self._dataset_limit)
        return True

    def evict_to_limit(self, limit=None):
        if limit is None:
            limit = self._dataset_limit
        limit = int(limit or 0)
        if limit <= 0:
            return
        while len(self) > limit and self._episodes:
            oldest_id = next(iter(self._episodes.keys()))
            self._remove_episode(oldest_id)

    def __len__(self):
        return int(self._gpu_transitions + self._spill_transitions)

    def _sample_sequence(self, length, target_device):
        if target_device is None:
            target_device = self._gpu_device if self._gpu_enabled else torch.device("cpu")
        target_device = torch.device(target_device)
        entries = [meta for meta in self._episodes.values() if meta["transitions"] >= 1]
        if not entries:
            raise ValueError("No replay episodes with at least 1 transition are available.")

        transitions = np.array(
            [max(meta["transitions"], 1) for meta in entries], dtype=np.float64
        )
        if self._episode_sampling_mode == "uniform":
            weights = np.full_like(
                transitions, 1.0 / float(len(transitions)), dtype=np.float64
            )
        else:
            weights = 1.0 / transitions
            weights /= np.sum(weights)

        spill_copy_ms = 0.0
        size = 0
        ret = None
        while size < length:
            meta = entries[int(self._rng.choice(len(entries), p=weights))]
            episode = meta["data"]
            total = meta["steps"]
            if total < 2:
                continue

            if ret is None:
                start = int(self._rng.randint(0, total - 1))
                end = min(start + length, total)
                segment = {}
                for key, value in episode.items():
                    seq = value[start:end].clone()
                    if seq.device != target_device:
                        t0 = time.perf_counter()
                        seq = seq.to(target_device, non_blocking=True)
                        spill_copy_ms += (time.perf_counter() - t0) * 1000.0
                    segment[key] = seq
                ret = segment
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                take = min(length - size, total)
                appended = {}
                for key, value in episode.items():
                    seq = value[:take].clone()
                    if seq.device != target_device:
                        t0 = time.perf_counter()
                        seq = seq.to(target_device, non_blocking=True)
                        spill_copy_ms += (time.perf_counter() - t0) * 1000.0
                    appended[key] = seq
                ret = {key: torch.cat([ret[key], appended[key]], dim=0) for key in ret}
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = int(next(iter(ret.values())).shape[0])
        return ret, spill_copy_ms

    def sample_batch(self, batch_size, length, device=None):
        batch_size = int(batch_size)
        length = int(length)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")
        if length <= 0:
            raise ValueError(f"length must be positive; got {length}.")
        start_time = time.perf_counter()
        batch = []
        total_copy_ms = 0.0
        target_device = device if device is not None else (
            self._gpu_device if self._gpu_enabled else torch.device("cpu")
        )
        for _ in range(batch_size):
            sequence, copy_ms = self._sample_sequence(length, target_device)
            batch.append(sequence)
            total_copy_ms += copy_ms
        data = {}
        for key in batch[0]:
            data[key] = torch.stack([sample[key] for sample in batch], dim=0)
        self._last_sample_ms = (time.perf_counter() - start_time) * 1000.0
        self._last_spill_copy_ms = float(total_copy_ms)
        return data

    def stats(self):
        return {
            "replay_gpu_episodes": int(
                sum(1 for meta in self._episodes.values() if meta["location"] == "gpu")
            ),
            "replay_gpu_transitions": int(self._gpu_transitions),
            "replay_gpu_bytes": int(self._gpu_bytes),
            "replay_spill_episodes": int(
                sum(1 for meta in self._episodes.values() if meta["location"] != "gpu")
            ),
            "replay_spill_transitions": int(self._spill_transitions),
            "replay_oom_fallbacks": int(self._oom_fallbacks),
            "replay_sample_ms": float(self._last_sample_ms),
            "replay_spill_copy_ms": float(self._last_spill_copy_ms),
        }


class GpuReplayStore(_BaseReplayStore):
    def __init__(self, config, name="replay"):
        super().__init__(config=config, name=name, allow_cpu_fallback=False)


class HybridReplayStore(_BaseReplayStore):
    def __init__(self, config, name="replay"):
        super().__init__(config=config, name=name, allow_cpu_fallback=True)


def _episode_end_reason(info):
    if not isinstance(info, dict):
        return False, 0.0, 0.0, 0.0

    env_info = info.get("env_info")
    if not isinstance(env_info, dict):
        env_info = {}

    tracked = (
        ("success" in info)
        or ("task_success" in info)
        or ("success" in env_info)
        or ("task_success" in env_info)
    )
    if not tracked:
        return False, 0.0, 0.0, 0.0

    success = bool(
        info.get(
            "success",
            info.get(
                "task_success",
                env_info.get("success", env_info.get("task_success", False)),
            ),
        )
    )
    timeout = bool(info.get("episode_timeout", env_info.get("episode_timeout", False)))

    if not success and not timeout:
        discount = info.get("discount", env_info.get("discount", None))
        if discount is not None and np.isclose(
            float(np.asarray(discount)), 1.0, atol=1e-6
        ):
            timeout = True

    if success:
        timeout = False
    other = (not success) and (not timeout)
    return True, float(success), float(timeout), float(other)


def simulate(
    agent,
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
    action_repeat=1,
    on_episode_done=None,
):
    # initialize or unpack simulation state
    start_logger_step = int(logger.step)
    eval_end_success = []
    eval_end_timeout = []
    eval_end_other = []
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        env_lengths = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)
    else:
        step, episode, done, env_lengths, obs, agent_state, reward = state
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset envs if necessary
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results]
            for index, result in zip(indices, results):
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        action, agent_state = agent(obs, done, agent_state)
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        env_lengths += 1
        step += len(envs)
        env_lengths *= 1 - done
        if not is_eval:
            logger.step = start_logger_step + int(step * action_repeat)
        # add to cache
        terminal_infos = {}
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                # Intentionally disable storing policy log-probabilities in replay.
                # This avoids schema mismatches with offline demos that do not have "logprob".
                transition.update({k: v for k, v in a.items() if k != "logprob"})
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, env.id, transition)
            if d:
                terminal_infos[env.id] = info

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                terminal_info = terminal_infos.get(envs[i].id, {})
                tracked, end_success, end_timeout, end_other = _episode_end_reason(
                    terminal_info
                )
                if on_episode_done is None:
                    # Preserve original CPU replay behavior path exactly.
                    save_episodes(directory, {envs[i].id: cache[envs[i].id]})
                    episode_length = len(cache[envs[i].id]["reward"]) - 1
                    score = float(np.array(cache[envs[i].id]["reward"]).sum())
                    video = cache[envs[i].id]["image"]
                    # record logs given from environments
                    for key in list(cache[envs[i].id].keys()):
                        if "log_" in key:
                            logger.scalar(
                                key, float(np.array(cache[envs[i].id][key]).sum())
                            )
                            # log items won't be used later
                            cache[envs[i].id].pop(key)

                    if not is_eval:
                        step_in_dataset = erase_over_episodes(cache, limit)
                        logger.scalar(f"dataset_size", step_in_dataset)
                        logger.scalar(f"train_return", score)
                        logger.scalar(f"train_length", episode_length)
                        logger.scalar(f"train_episodes", len(cache))
                        if tracked:
                            logger.scalar("train_end_success", end_success)
                            logger.scalar("train_end_timeout", end_timeout)
                            logger.scalar("train_end_other", end_other)
                        logger.write(step=logger.step)
                    else:
                        if not "eval_lengths" in locals():
                            eval_lengths = []
                            eval_scores = []
                            eval_done = False
                        # start counting scores for evaluation
                        eval_scores.append(score)
                        eval_lengths.append(episode_length)
                        if tracked:
                            eval_end_success.append(end_success)
                            eval_end_timeout.append(end_timeout)
                            eval_end_other.append(end_other)

                        score = sum(eval_scores) / len(eval_scores)
                        mean_eval_length = sum(eval_lengths) / len(eval_lengths)
                        logger.video(f"eval_policy", np.array(video)[None])

                        if len(eval_scores) >= episodes and not eval_done:
                            logger.scalar(f"eval_return", score)
                            logger.scalar(f"eval_length", mean_eval_length)
                            logger.scalar(f"eval_episodes", len(eval_scores))
                            if eval_end_success:
                                logger.scalar(
                                    "eval_end_success_rate",
                                    float(np.mean(eval_end_success)),
                                )
                                logger.scalar(
                                    "eval_end_timeout_rate",
                                    float(np.mean(eval_end_timeout)),
                                )
                                logger.scalar(
                                    "eval_end_other_rate",
                                    float(np.mean(eval_end_other)),
                                )
                            logger.write(step=logger.step)
                            eval_done = True
                else:
                    episode_data = {
                        key: np.asarray(value) for key, value in cache[envs[i].id].items()
                    }
                    save_episodes(directory, {envs[i].id: episode_data})
                    episode_length = len(episode_data["reward"]) - 1
                    score = float(np.array(episode_data["reward"]).sum())
                    video = episode_data["image"]
                    # record logs given from environments
                    for key in list(episode_data.keys()):
                        if "log_" in key:
                            logger.scalar(
                                key, float(np.array(episode_data[key]).sum())
                            )
                            episode_data.pop(key)

                    callback_stats = None
                    if not is_eval:
                        callback_stats = on_episode_done(envs[i].id, episode_data)
                        cache.pop(envs[i].id, None)

                    if not is_eval:
                        step_in_dataset = int(callback_stats.get("dataset_size", 0))
                        train_episodes = int(
                            callback_stats.get("train_episodes", len(cache))
                        )
                        for name, value in callback_stats.items():
                            if name in ("dataset_size", "train_episodes"):
                                continue
                            logger.scalar(name, float(value))
                        logger.scalar(f"dataset_size", step_in_dataset)
                        logger.scalar(f"train_return", score)
                        logger.scalar(f"train_length", episode_length)
                        logger.scalar(f"train_episodes", train_episodes)
                        if tracked:
                            logger.scalar("train_end_success", end_success)
                            logger.scalar("train_end_timeout", end_timeout)
                            logger.scalar("train_end_other", end_other)
                        logger.write(step=logger.step)
                    else:
                        if not "eval_lengths" in locals():
                            eval_lengths = []
                            eval_scores = []
                            eval_done = False
                        # start counting scores for evaluation
                        eval_scores.append(score)
                        eval_lengths.append(episode_length)
                        if tracked:
                            eval_end_success.append(end_success)
                            eval_end_timeout.append(end_timeout)
                            eval_end_other.append(end_other)

                        score = sum(eval_scores) / len(eval_scores)
                        mean_eval_length = sum(eval_lengths) / len(eval_lengths)
                        logger.video(f"eval_policy", np.array(video)[None])

                        if len(eval_scores) >= episodes and not eval_done:
                            logger.scalar(f"eval_return", score)
                            logger.scalar(f"eval_length", mean_eval_length)
                            logger.scalar(f"eval_episodes", len(eval_scores))
                            if eval_end_success:
                                logger.scalar(
                                    "eval_end_success_rate",
                                    float(np.mean(eval_end_success)),
                                )
                                logger.scalar(
                                    "eval_end_timeout_rate",
                                    float(np.mean(eval_end_timeout)),
                                )
                                logger.scalar(
                                    "eval_end_other_rate",
                                    float(np.mean(eval_end_other)),
                                )
                            logger.write(step=logger.step)
                            eval_done = True
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (
        step - steps,
        episode - episodes,
        done,
        env_lengths,
        obs,
        agent_state,
        reward,
    )


def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


def _define_spaces(episode, config):
    """Constructs observation space by filtering episode keys against config regex."""
    exclude_keys = {
        "action",
        "reward",
        "discount",
        "is_first",
        "is_terminal",
        "policy_target",
    }
    obs_spaces = {}
    mlp_pat = config.encoder.get("mlp_keys", "$^")
    cnn_pat = config.encoder.get("cnn_keys", "$^")

    for key, value in episode.items():
        if key in exclude_keys:
            continue

        is_mlp = re.match(mlp_pat, key)
        is_cnn = re.match(cnn_pat, key)
        if not (is_mlp or is_cnn):
            continue

        shape = value.shape[1:]
        if is_cnn:
            h = config.image_crop_height
            w = config.image_crop_width
            if h > 0 and w > 0:
                shape = (h, w, shape[-1])

        if value.dtype == np.uint8:
            low, high = 0, 255
            dtype = np.uint8
        else:
            low, high = -np.inf, np.inf
            dtype = np.float32

        obs_spaces[key] = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

    if not obs_spaces:
        raise ValueError("No keys matched config.encoder regex!")

    obs_space = gym.spaces.Dict(obs_spaces)
    action = episode.get("action")
    if action is None:
        raise ValueError("Episode missing 'action'.")
    # TODO: This is specific to robosuite environments.
    # For more general environments, this should be configurable.
    act_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=action.shape[1:], dtype=np.float32
    )

    return obs_space, act_space


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0, sampling_mode="shorter"):
    np_random = np.random.RandomState(seed)
    sampling_mode = _parse_episode_sampling_mode(sampling_mode)

    if not episodes:
        raise ValueError("No episodes available for sampling.")

    def _signature_map(episode):
        fields = {}
        for key, value in episode.items():
            if key.startswith("log_"):
                continue
            arr = np.asarray(value)
            fields[key] = (arr.shape[1:], str(arr.dtype))
        return fields

    def _signature(episode):
        sig_map = _signature_map(episode)
        return tuple(sorted((key, *sig_map[key]) for key in sig_map))

    ref_episode = next(iter(episodes.values()))
    ref_sig_map = _signature_map(ref_episode)
    ref_sig = _signature(ref_episode)
    warned_incompatible = False

    while True:
        valid_episodes = []
        dropped_incompatible = 0
        dropped_too_short = 0
        first_incompatible_name = None
        first_incompatible_sig_map = None
        for episode_name, episode in episodes.items():
            if _signature(episode) != ref_sig:
                dropped_incompatible += 1
                if first_incompatible_sig_map is None:
                    first_incompatible_name = str(episode_name)
                    first_incompatible_sig_map = _signature_map(episode)
                continue
            total = len(next(iter(episode.values())))
            if total < 2:
                dropped_too_short += 1
                continue
            valid_episodes.append(episode)

        if not valid_episodes:
            raise ValueError(
                "No compatible episodes with at least 2 frames are available for sampling."
            )

        if not warned_incompatible and (dropped_incompatible or dropped_too_short):
            print(
                "[tools.sample_episodes] Dropped "
                f"{dropped_incompatible} incompatible and {dropped_too_short} short episode(s)."
            )
            if dropped_incompatible and first_incompatible_sig_map is not None:
                missing = sorted(set(ref_sig_map) - set(first_incompatible_sig_map))
                extra = sorted(set(first_incompatible_sig_map) - set(ref_sig_map))
                changed = sorted(
                    key
                    for key in (set(ref_sig_map) & set(first_incompatible_sig_map))
                    if ref_sig_map[key] != first_incompatible_sig_map[key]
                )
                print(
                    "[tools.sample_episodes] Example incompatible episode: "
                    f"{first_incompatible_name}"
                )
                if missing:
                    print(f"[tools.sample_episodes] Missing keys: {missing}")
                if extra:
                    print(f"[tools.sample_episodes] Extra keys: {extra}")
                if changed:
                    preview = changed[:8]
                    diff = {
                        key: {
                            "ref": ref_sig_map[key],
                            "episode": first_incompatible_sig_map[key],
                        }
                        for key in preview
                    }
                    print(
                        "[tools.sample_episodes] Changed key shape/dtype "
                        f"(showing {len(preview)}/{len(changed)}): {diff}"
                    )
            warned_incompatible = True

        size = 0
        ret = None
        lengths = np.array(
            [len(next(iter(episode.values()))) for episode in valid_episodes],
            dtype=np.float64,
        )
        transition_counts = np.maximum(lengths - 1.0, 1.0)
        if sampling_mode == "uniform":
            p = np.full_like(
                transition_counts,
                1.0 / float(len(transition_counts)),
                dtype=np.float64,
            )
        else:
            # Prefer shorter episodes by sampling with inverse transition count.
            p = 1.0 / transition_counts
            p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(valid_episodes, p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


def compute_image_dataset_stats(episodes):
    """Compute per-pixel mean and std over all frames in all episodes.

    Returns (mean, std) with shape (1, H, W, C) in float64 numpy arrays. Raises
    if no image key is present.
    """
    first = None
    for ep in episodes.values():
        if "image" in ep:
            first = np.asarray(ep["image"])
            break
    if first is None:
        raise ValueError("No 'image' key found in provided episodes for statistics computation.")
    if first.ndim != 4:
        raise ValueError(f"Expected image shape [T, H, W, C], got {first.shape}.")
    pixel_shape = first.shape[1:]
    total_frames = 0
    sum_img = np.zeros(pixel_shape, dtype=np.float64)
    sum_sq = np.zeros(pixel_shape, dtype=np.float64)
    for ep in episodes.values():
        if "image" not in ep:
            continue
        imgs = np.asarray(ep["image"], dtype=np.float64)
        if imgs.ndim != 4:
            raise ValueError(f"Expected image shape [T, H, W, C], got {imgs.shape}.")
        imgs = imgs / 255.0
        sum_img += imgs.sum(axis=0)
        sum_sq += np.square(imgs).sum(axis=0)
        total_frames += imgs.shape[0]
    if total_frames == 0:
        raise ValueError("No frames available to compute image statistics.")
    mean = sum_img / total_frames
    var = np.maximum(sum_sq / total_frames - np.square(mean), 0.0)
    std = np.sqrt(var)
    return mean[None, ...], std[None, ...]


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            scale = (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
            out = out * scale
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            scale = (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
            out = out * scale
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class FaithfulContDist:
    """Faithful heteroscedastic normal (Stirn et al.).

    Splits the NLL into two terms so that:
      - The mean is trained with unit-variance log_prob (equivalent to MSE).
      - The std is trained with its own log_prob using a detached mean,
        so std gradients only update the std head.
    Requires that the trunk features are already detached before the std head
    (handled in MLP.forward).
    """

    def __init__(self, mean, std, absmax=None):
        super().__init__()
        self._mean = mean
        self._std = std
        self.absmax = absmax

    @property
    def base_dist(self):
        return torchd.independent.Independent(
            torchd.normal.Normal(self._mean, self._std), 1
        )

    @property
    def mean(self):
        return self._mean

    def mode(self):
        out = self._mean
        if self.absmax is not None:
            scale = (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
            out = out * scale
        return out

    def sample(self, sample_shape=()):
        out = self.base_dist.rsample(sample_shape)
        if self.absmax is not None:
            scale = (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
            out = out * scale
        return out

    def entropy(self):
        return self.base_dist.entropy()

    def log_prob(self, x):
        # Mean term: unit variance → gradients are pure MSE for the mean/trunk
        mean_dist = torchd.independent.Independent(
            torchd.normal.Normal(self._mean, torch.ones_like(self._std)), 1
        )
        # Std term: detached mean → gradients only update the std head
        std_dist = torchd.independent.Independent(
            torchd.normal.Normal(self._mean.detach(), self._std), 1
        )
        return mean_dist.log_prob(x) + std_dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = to_np(norm)
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif _supports_object_dict(attr):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)


def _supports_object_dict(value):
    try:
        getattr(value, "__dict__")
    except AttributeError:
        return False
    except Exception:
        return False
    return True
