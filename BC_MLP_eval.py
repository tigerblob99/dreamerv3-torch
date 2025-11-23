"""Roll out a behavior-cloned Dreamer encoder + MLP policy inside the robomimic Can task."""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import importlib

import gym  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

try:
	import cv2  # type: ignore
except ImportError:
	cv2 = None

import networks
import tools
from BC_MLP_train import _build_encoder, _extract_encoder_state, _load_config
from envs.robosuite_env import (
	_ensure_composite_controller_config,
	_resolve_controller_loader,
)


@dataclass
class EvalConfig:
	policy_path: pathlib.Path
	checkpoint: pathlib.Path
	episodes: int
	max_env_steps: int
	camera_obs_keys: Tuple[str, ...]
	mlp_obs_keys: Tuple[str, ...]
	camera_image_key: str
	clip_actions: bool
	render: bool
	device: str
	seed: int
	robosuite_task: str
	robosuite_robots: Tuple[str, ...]
	robosuite_controller: str
	robosuite_reward_shaping: bool
	robosuite_control_freq: int
	flip_camera_keys: Tuple[str, ...]
	bc_hidden_units: int
	bc_hidden_layers: int
	bc_activation: str
	bc_use_layernorm: bool
	video_dir: pathlib.Path | None
	video_fps: int
	video_camera: str | None
	video_camera_keys: Tuple[str, ...] | None
	dataset_path: pathlib.Path | None
	dataset_obs_path: pathlib.Path | None
	encoder_snapshot_path: pathlib.Path | None
	replay_dataset_demo: bool
	replay_demo_key: str | None
	replay_max_steps: int | None
	action_dim: int | None = None
def _make_robomimic_can_env(cfg: EvalConfig, image_hw: Tuple[int, int]):
	suite = importlib.import_module("robosuite")

	load_controller_config = _resolve_controller_loader(suite)
	controller_cfg = load_controller_config(default_controller=cfg.robosuite_controller)
	controller_cfg = _ensure_composite_controller_config(controller_cfg, cfg.robosuite_robots)

	env = suite.make(
		env_name=cfg.robosuite_task,
		robots=list(cfg.robosuite_robots),
		controller_configs=controller_cfg,
		use_object_obs=True,
		use_camera_obs=True,
		camera_names=[name.replace("_image", "") for name in cfg.camera_obs_keys],
		camera_heights=image_hw[0],
		camera_widths=image_hw[1],
		has_renderer=cfg.render,
		has_offscreen_renderer=True,
		reward_shaping=cfg.robosuite_reward_shaping,
		control_freq=cfg.robosuite_control_freq,
		horizon=cfg.max_env_steps,
		ignore_done=False,
	)

	try:
		env.reset(seed=cfg.seed)
	except TypeError:
		seed_fn = getattr(env, "seed", None)
		if callable(seed_fn):
			seed_fn(cfg.seed)
		env.reset()
	return env




def _stack_cameras(obs: Dict[str, Any], camera_keys: Iterable[str], flip_keys: Iterable[str] = ()): 
	frames: List[Any] = []
	flip_set = set(flip_keys)
	for key in camera_keys:
		if key not in obs:
			raise KeyError(f"Camera observation '{key}' missing from environment output")
		frame = np.asarray(obs[key])
		if frame.dtype != np.float32:
			frame = frame.astype(np.float32)
		if frame.max() > 1.0:
			frame /= 255.0
		if key in flip_set:
			frame = np.flip(frame, axis=(0, 1))
		frames.append(frame)
	return np.concatenate(frames, axis=-1)


def _log_obs_snapshot(
	label: str,
	obs: Dict[str, Any],
	*,
	preview: int = 8,
	full_keys: Iterable[str] | None = None,
	output_path: pathlib.Path | None = None,
) -> None:
	full_set = set(full_keys or [])
	lines = [f"[obs] {label} -> {len(obs)} keys"]
	for key in sorted(obs.keys()):
		arr = np.asarray(obs[key])
		if key in full_set:
			preview_str = np.array2string(
				arr,
				precision=4,
				suppress_small=True,
				separator=", ",
				max_line_width=1_000_000,
				threshold=arr.size,
			)
		else:
			flat = arr.reshape(-1)
			if preview <= 0 or flat.size <= preview:
				preview_str = np.array2string(flat, precision=4, suppress_small=True)
			else:
				preview_str = np.array2string(flat[:preview], precision=4, suppress_small=True)
		lines.append(
			f"  - {key}: shape={arr.shape}, dtype={arr.dtype}, sample={preview_str}"
		)
	text = "\n".join(lines)
	print(text)
	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(text + "\n")


def _load_dataset_obs_frame(
	dataset_file: h5py.File,
	demo_key: str,
	frame_idx: int,
) -> Dict[str, Any] | None:
	data_group = dataset_file.get("data")
	if data_group is None or demo_key not in data_group:
		return None
	demo = data_group[demo_key]
	obs_group = demo.get("obs")
	if obs_group is None:
		return None
	result: Dict[str, Any] = {}
	for key in obs_group.keys():
		dset = obs_group[key]
		if dset.shape[0] == 0:
			continue
		idx = max(0, min(frame_idx, dset.shape[0] - 1))
		result[key] = np.array(dset[idx])
	return result


def _compare_processed_obs(
	label: str,
	ref_obs: Dict[str, Any],
	candidate_obs: Dict[str, Any],
) -> None:
	shared = sorted(set(ref_obs.keys()) & set(candidate_obs.keys()))
	if not shared:
		print(f"[compare] {label}: no overlapping keys")
		return
	print(f"[compare] {label}: {len(shared)} shared keys")
	for key in shared:
		ref = np.asarray(ref_obs[key], dtype=np.float32)
		cand = np.asarray(candidate_obs[key], dtype=np.float32)
		if ref.shape != cand.shape:
			print(f"  - {key}: SHAPE mismatch ref={ref.shape} cand={cand.shape}")
			continue
		diff = np.abs(ref - cand)
		print(
			f"  - {key}: max={diff.max():.6f}, mean={diff.mean():.6f}, rmse={np.sqrt((diff ** 2).mean()):.6f}"
		)


def _prepare_obs(
	raw_obs: Dict[str, Any],
	*,
	camera_keys: Tuple[str, ...],
	flip_keys: Tuple[str, ...],
	mlp_keys: Tuple[str, ...],
	is_first: bool,
	is_terminal: bool,
) -> Dict[str, Any]:
	processed: Dict[str, Any] = {}
	processed["image"] = _stack_cameras(raw_obs, camera_keys, flip_keys)
	trig_key_map = {
		"robot0_joint_pos_sin": "aux_robot0_joint_pos_sin",
		"robot0_joint_pos_cos": "aux_robot0_joint_pos_cos",
	}
	seen: set[str] = set()
	for key in mlp_keys:
		if key in trig_key_map:
			# These are renamed below to keep aux_ prefix consistent with training data.
			continue
		if key in seen:
			continue
		seen.add(key)
		if key not in raw_obs:
			continue
		processed[key] = np.asarray(raw_obs[key], dtype=np.float32)

	for raw_key, aux_key in trig_key_map.items():
		if raw_key not in raw_obs:
			continue
		processed[aux_key] = np.asarray(raw_obs[raw_key], dtype=np.float32)

	processed["is_first"] = np.array([1.0 if is_first else 0.0], dtype=np.float32)
	processed["is_terminal"] = np.array([1.0 if is_terminal else 0.0], dtype=np.float32)
	return processed


def _dict_to_space(obs: Dict[str, Any]):
	spaces: Dict[str, Any] = {}
	for key, value in obs.items():
		arr = np.asarray(value)
		dtype = np.uint8 if arr.dtype == np.uint8 else np.float32
		low = 0 if dtype == np.uint8 else -np.inf
		high = 255 if dtype == np.uint8 else np.inf
		spaces[key] = gym.spaces.Box(low=low, high=high, shape=arr.shape, dtype=dtype)
	return gym.spaces.Dict(spaces)


def _obs_to_torch(obs: Dict[str, Any], device: Any) -> Dict[str, Any]:
	tensors: Dict[str, Any] = {}
	for key, value in obs.items():
		arr = np.asarray(value)
		if arr.ndim == 0:
			arr = arr[None]
		batch = np.expand_dims(arr, axis=0)
		tensors[key] = torch.as_tensor(batch, device=device).float()
	return tensors


def _replay_dataset_demo(
	env,
	config: EvalConfig,
	dataset_file: h5py.File,
	demo_key: str,
	video_recorder: EpisodeVideoRecorder,
	imgs_dir: pathlib.Path,
	encoder: torch.nn.Module,
	policy: torch.nn.Module,
	device: torch.device,
	*,
	max_steps: int | None = None,
):
	data_group = dataset_file["data"].get(demo_key)
	if data_group is None:
		print(f"[replay] demo '{demo_key}' missing from dataset; skipping open-loop replay")
		return
	actions_ds = data_group.get("actions")
	states_ds = data_group.get("states")
	if actions_ds is None or states_ds is None:
		print(f"[replay] demo '{demo_key}' missing actions/states; skipping open-loop replay")
		return
	actions = np.asarray(actions_ds)
	if actions.size == 0:
		print(f"[replay] demo '{demo_key}' has no actions; skipping open-loop replay")
		return
	available_steps = actions.shape[0]
	limit = max_steps if max_steps and max_steps > 0 else available_steps
	limit = min(limit, available_steps)
	print(f"[replay] Starting dataset action replay for '{demo_key}' ({limit} steps)")
	env.reset()
	env.sim.set_state_from_flattened(states_ds[0])
	env.sim.forward()
	raw_obs = env._get_observations()
	video_recorder.start_episode(-1)
	video_recorder.add_frame(raw_obs)
	total_reward = 0.0
	steps = 0
	episode_actions: List[np.ndarray] = []
	policy_actions: List[np.ndarray] = []
	action_diffs: List[np.ndarray] = []
	for idx in range(limit):
		action = np.asarray(actions[idx], dtype=np.float32)
		episode_actions.append(action.copy())
		step_obs = _prepare_obs(
			raw_obs,
			camera_keys=config.camera_obs_keys,
			flip_keys=config.flip_camera_keys,
			mlp_keys=config.mlp_obs_keys,
			is_first=(idx == 0),
			is_terminal=False,
		)
		tensor_obs = _obs_to_torch(step_obs, device)
		with torch.no_grad():
			embedding = encoder(tensor_obs)
			policy_act = policy(embedding).squeeze(0).cpu().numpy()
		if config.clip_actions:
			policy_act = np.clip(policy_act, -1.0, 1.0)
		policy_actions.append(policy_act.copy())
		action_diffs.append(policy_act - action)
		raw_obs, reward, done, _ = env.step(action)
		video_recorder.add_frame(raw_obs)
		total_reward += float(reward)
		steps += 1
		if done:
			print(f"[replay] Environment terminated at step {idx + 1}")
			break
	saved_video = video_recorder.finish_episode()
	if saved_video:
		replay_name = saved_video.with_name(f"{demo_key}_dataset_replay.mp4")
		try:
			saved_video.rename(replay_name)
		except OSError:
			replay_name = saved_video
		print(f"[replay] Saved dataset replay video to {replay_name}")
	if episode_actions:
		acts = np.stack(episode_actions)
		preds = np.stack(policy_actions) if policy_actions else None
		dim = acts.shape[1]
		fig, axes = plt.subplots(dim, 1, figsize=(12, 2.4 * dim), sharex=True)
		if dim == 1:
			axes = [axes]
		for i in range(dim):
			axes[i].plot(acts[:, i], label="dataset", linewidth=2)
			if preds is not None:
				axes[i].plot(preds[:, i], label="policy", linewidth=1, linestyle="--")
			axes[i].set_ylabel(f"Dim {i}")
		axes[-1].set_xlabel("Replay Step")
		if preds is not None:
			axes[0].legend(loc="upper right")
		plt.tight_layout()
		plot_path = imgs_dir / f"{demo_key}_dataset_replay_actions.png"
		plt.savefig(plot_path)
		plt.close(fig)
		print(f"[replay] Saved dataset vs policy action plot to {plot_path}")
	if action_diffs:
		diffs = np.stack(action_diffs)
		dim = diffs.shape[1]
		fig, axes = plt.subplots(dim, 1, figsize=(12, 2.0 * dim), sharex=True)
		if dim == 1:
			axes = [axes]
		for i in range(dim):
			axes[i].plot(diffs[:, i])
			axes[i].set_ylabel(f"Δ Dim {i}")
		axes[-1].set_xlabel("Replay Step")
		plt.suptitle("Policy − Dataset action difference")
		plt.tight_layout()
		diff_path = imgs_dir / f"{demo_key}_dataset_replay_action_diff.png"
		plt.savefig(diff_path)
		plt.close(fig)
		print(f"[replay] Saved action difference plot to {diff_path}")
	print(f"[replay] Finished '{demo_key}' replay: reward={total_reward:.2f}, steps={steps}")



class EpisodeVideoRecorder:
	def __init__(
		self,
		directory: pathlib.Path | None,
		fps: int,
		camera_key: str | None,
		camera_keys: Tuple[str, ...] | None = None,
		flip_keys: Iterable[str] = (),
	):
		self._dir = directory
		self._fps = fps
		self._camera_key = camera_key
		self._camera_keys = tuple(camera_keys or ()) if camera_keys else None
		self._flip_keys = set(flip_keys)
		self._writer = None
		self._path: pathlib.Path | None = None
		self._episode_idx: int | None = None

	def enabled(self) -> bool:
		if self._dir is None:
			return False
		return bool(self._camera_key or (self._camera_keys and len(self._camera_keys) > 0))

	def start_episode(self, episode_idx: int):
		self._episode_idx = episode_idx
		self._path = None
		self._release()

	def add_frame(self, obs: Dict[str, Any]):
		if not self.enabled():
			return
		frames: List[np.ndarray] = []
		if self._camera_key is not None:
			if self._camera_key not in obs:
				return
			frame = np.asarray(obs[self._camera_key])
			if self._camera_key in self._flip_keys:
				frame = np.flip(frame, axis=(0, 1))
			frames.append(frame)
		elif self._camera_keys:
			for key in self._camera_keys:
				if key not in obs:
					continue
				frame = np.asarray(obs[key])
				if key in self._flip_keys:
					frame = np.flip(frame, axis=(0, 1))
				frames.append(frame)
		if not frames:
			return
		try:
			frame = self._combine_frames(frames)
		except ValueError:
			return
		if frame.ndim != 3 or frame.shape[-1] not in (1, 3):
			return
		frame_uint8 = self._to_uint8(frame)
		if self._writer is None:
			if cv2 is None:
				raise ImportError("opencv-python is required for video recording; install it to enable --video_dir")
			self._init_writer()
			self._path = self._dir / f"episode_{self._episode_idx:03d}.mp4"
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			h, w = frame_uint8.shape[0], frame_uint8.shape[1]
			self._writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (w, h))
		self._writer.write(cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))

	def finish_episode(self) -> pathlib.Path | None:
		self._release()
		return self._path

	def _init_writer(self):
		if self._dir is None:
			return
		self._dir.mkdir(parents=True, exist_ok=True)

	def _release(self):
		if self._writer is not None:
			self._writer.release()
		self._writer = None

	@staticmethod
	def _combine_frames(frames: List[np.ndarray]) -> np.ndarray:
		if len(frames) == 1:
			return frames[0]
		shapes = [frame.shape for frame in frames]
		first_shape = shapes[0]
		for shape in shapes[1:]:
			if shape[:2] != first_shape[:2]:
				raise ValueError("All video camera frames must share height and width")
		return np.concatenate(frames, axis=1)

	@staticmethod
	def _to_uint8(frame: Any) -> Any:
		if frame.dtype == np.uint8:
			return frame
		if np.issubdtype(frame.dtype, np.floating):
			return np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
		return frame.astype(np.uint8)



def _build_action_mlp(input_dim: int, action_dim: int, cfg: EvalConfig, *, use_layernorm: bool, use_bias: bool):
	act_cls = getattr(torch.nn, cfg.bc_activation)
	layers: List[Any] = []
	last_dim = input_dim
	for _ in range(cfg.bc_hidden_layers):
		layers.append(torch.nn.Linear(last_dim, cfg.bc_hidden_units, bias=use_bias))
		if use_layernorm:
			layers.append(torch.nn.LayerNorm(cfg.bc_hidden_units, eps=1e-3))
		layers.append(act_cls())
		last_dim = cfg.bc_hidden_units
	layers.append(torch.nn.Linear(last_dim, action_dim, bias=True))
	mlp = torch.nn.Sequential(*layers)
	mlp.apply(lambda m: tools.weight_init(m) if isinstance(m, torch.nn.Linear) else None)
	return mlp


def _extract_success(info, env) -> bool:
	success = False
	if isinstance(info, dict):
		if "success" in info:
			success = bool(info["success"])
		env_info = info.get("env_info")
		if isinstance(env_info, dict):
			success = bool(env_info.get("task_success", env_info.get("success", success)))
		success = bool(info.get("task_success", success))
	checker = getattr(env, "is_success", None)
	if callable(checker):
		try:
			result = checker()
			if isinstance(result, dict):
				success = bool(result.get("task", success))
			else:
				success = bool(result)
		except Exception:
			pass
	return success



def evaluate_policy(config: EvalConfig, dreamer_cfg):
	tools.set_seed_everywhere(config.seed)
	size = tuple(getattr(dreamer_cfg, "size", (84, 84)))
	device = torch.device(config.device)
	policy_ckpt = torch.load(config.policy_path, map_location=torch.device("cpu"))
	policy_cfg = policy_ckpt.get("config", {})
	for attr in (
		"bc_hidden_units",
		"bc_hidden_layers",
		"bc_activation",
		"bc_use_layernorm",
	):
		if attr in policy_cfg:
			setattr(config, attr, policy_cfg[attr])

	default_video_key = None
	if config.video_camera is None and (not config.video_camera_keys or len(config.video_camera_keys) == 0):
		default_video_key = config.camera_obs_keys[0] if config.camera_obs_keys else None
	video_recorder = EpisodeVideoRecorder(
		directory=config.video_dir,
		fps=config.video_fps,
		camera_key=config.video_camera or default_video_key,
		camera_keys=config.video_camera_keys,
		flip_keys=config.flip_camera_keys,
	)
	env = _make_robomimic_can_env(config, size)
	imgs_dir = pathlib.Path("imgs")
	imgs_dir.mkdir(exist_ok=True)

	dataset_file = None
	dataset_obs_file = None
	demo_keys: List[str] = []
	obs_demo_keys: List[str] = []
	if config.dataset_path:
		if not config.dataset_path.exists():
			raise FileNotFoundError(f"Dataset not found at {config.dataset_path}")
		dataset_file = h5py.File(config.dataset_path, "r")
		if "data" in dataset_file:
			demo_keys = sorted(dataset_file["data"].keys())
	if config.dataset_obs_path:
		obs_path = config.dataset_obs_path
		if not obs_path.exists():
			raise FileNotFoundError(f"Observation dataset not found at {obs_path}")
		if dataset_file is not None and obs_path.resolve() == config.dataset_path.resolve():
			dataset_obs_file = dataset_file
		else:
			dataset_obs_file = h5py.File(obs_path, "r")
		if "data" in dataset_obs_file:
			obs_demo_keys = sorted(dataset_obs_file["data"].keys())
	elif dataset_file is not None:
		dataset_obs_file = dataset_file
		obs_demo_keys = list(demo_keys)

	if dataset_file and demo_keys:
		initial_state = dataset_file["data"][demo_keys[0]]["states"][0]
		env.reset()
		env.sim.set_state_from_flattened(initial_state)
		env.sim.forward()
		obs = env._get_observations()
		_log_obs_snapshot("initial dataset state (warmup)", obs)
	else:
		obs = env.reset()
		_log_obs_snapshot("initial random reset (warmup)", obs)

	processed = _prepare_obs(
		obs,
		camera_keys=config.camera_obs_keys,
		flip_keys=config.flip_camera_keys,
		mlp_keys=config.mlp_obs_keys,
		is_first=True,
		is_terminal=False,
	)
	if dataset_obs_file and obs_demo_keys:
		dataset_obs = _load_dataset_obs_frame(dataset_obs_file, obs_demo_keys[0], 0)
		if dataset_obs is None:
			print(
				f"[compare] dataset obs missing in {config.dataset_obs_path or config.dataset_path}; skip encoder input comparison"
			)
		else:
			dataset_processed = _prepare_obs(
				dataset_obs,
				camera_keys=config.camera_obs_keys,
				flip_keys=config.flip_camera_keys,
				mlp_keys=config.mlp_obs_keys,
				is_first=True,
				is_terminal=False,
			)
			_compare_processed_obs(
				"demo0 processed obs vs env warmup",
				dataset_processed,
				processed,
			)
			if config.encoder_snapshot_path:
				_log_obs_snapshot(
					"encoder inputs demo0 warmup",
					processed,
					preview=-1,
					full_keys=("image",),
					output_path=config.encoder_snapshot_path,
				)
	obs_space = _dict_to_space(processed)
	encoder = _build_encoder(dreamer_cfg, obs_space)
	encoder.eval().to(config.device)

	checkpoint = config.checkpoint.expanduser()
	if not checkpoint.exists():
		raise FileNotFoundError(f"Dreamer checkpoint not found at {checkpoint}")
	ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
	agent_state = ckpt.get("agent_state_dict")
	if agent_state is None:
		raise KeyError("agent_state_dict missing in checkpoint")
	encoder_state = _extract_encoder_state(agent_state)
	encoder.load_state_dict(encoder_state, strict=False)

	action_dim = getattr(env, "action_dim", None)
	if action_dim is None:
		raise AttributeError("Environment is missing 'action_dim' attribute required for policy output size")
	saved_in_dim = policy_ckpt.get("encoder_outdim")
	if saved_in_dim and saved_in_dim != encoder.outdim:
		print(
			f"[warn] Encoder outdim mismatch (runtime={encoder.outdim}, checkpoint={saved_in_dim}); "
			"ensure camera resolution matches training (try --image_size 84 84)."
		)
		# prefer checkpoint dimension to preserve layer shapes
		input_dim = int(saved_in_dim)
	else:
		input_dim = encoder.outdim
	policy = _build_action_mlp(
		input_dim,
		action_dim,
		config,
		use_layernorm=getattr(config, "bc_use_layernorm", False),
		use_bias=False,
	).to(config.device)
	policy.eval()
	policy.load_state_dict(policy_ckpt["action_mlp_state_dict"])

	if config.replay_dataset_demo:
		if not dataset_file or not demo_keys:
			print("[replay] --replay_dataset_demo requested but --dataset_path is missing demos; skipping replay")
		else:
			replay_key = config.replay_demo_key or demo_keys[0]
			if replay_key not in dataset_file["data"]:
				print(f"[replay] demo '{replay_key}' not found in dataset; available keys start with {demo_keys[:3]}")
			else:
				_replay_dataset_demo(
					env,
					config,
					dataset_file,
					replay_key,
					video_recorder,
					imgs_dir,
					encoder,
					policy,
					device,
					max_steps=config.replay_max_steps,
				)

	episode_rewards: List[float] = []
	success_flags: List[bool] = []

	for episode in range(config.episodes):
		if dataset_file and demo_keys:
			demo_key = demo_keys[episode % len(demo_keys)]
			initial_state = dataset_file["data"][demo_key]["states"][0]
			env.reset()
			env.sim.set_state_from_flattened(initial_state)
			env.sim.forward()
			raw_obs = env._get_observations()
			print(f"Initialized episode {episode + 1} from dataset demo '{demo_key}'")
			_log_obs_snapshot(
				f"episode {episode + 1} dataset state ({demo_key})",
				raw_obs,
			)
		else:
			raw_obs = env.reset()
			print(f"Initialized episode {episode + 1} with random reset")
			_log_obs_snapshot(
				f"episode {episode + 1} random reset",
				raw_obs,
			)
		video_recorder.start_episode(episode)
		video_recorder.add_frame(raw_obs)
		done = False
		total_reward = 0.0
		steps = 0
		episode_actions = []
		while not done and steps < config.max_env_steps:
			step_obs = _prepare_obs(
				raw_obs,
				camera_keys=config.camera_obs_keys,
				flip_keys=config.flip_camera_keys,
				mlp_keys=config.mlp_obs_keys,
				is_first=(steps == 0),
				is_terminal=done,
			)
			tensor_obs = _obs_to_torch(step_obs, device)
			with torch.no_grad():
				embedding = encoder(tensor_obs)
				action = policy(embedding)
			action_np = action.squeeze(0).cpu().numpy()
			episode_actions.append(action_np.copy())
			if config.clip_actions:
				action_np = np.clip(action_np, -1.0, 1.0)
			raw_obs, reward, done, info = env.step(action_np)
			video_recorder.add_frame(raw_obs)
			total_reward += float(reward)
			steps += 1
			if done:
				success_flags.append(_extract_success(info, env))

		if episode_actions:
			acts = np.array(episode_actions)
			dim = acts.shape[1]
			fig, axes = plt.subplots(dim, 1, figsize=(10, 2 * dim), sharex=True)
			if dim == 1:
				axes = [axes]
			for i in range(dim):
				axes[i].plot(acts[:, i])
				axes[i].set_ylabel(f"Dim {i}")
			plt.xlabel("Step")
			plt.tight_layout()
			plt.savefig(imgs_dir / f"episode_{episode}_actions.png")
			plt.close(fig)

		episode_rewards.append(total_reward)
		saved_video = video_recorder.finish_episode()
		if saved_video:
			print(f"Saved episode {episode + 1} video to {saved_video}")
		print(
			f"Episode {episode + 1}/{config.episodes}: reward={total_reward:.2f}, steps={steps}, "
			f"success={success_flags[-1] if success_flags else False}"
		)

	mean_reward = float(np.mean(episode_rewards)) if episode_rewards else float("nan")
	std_reward = float(np.std(episode_rewards)) if episode_rewards else float("nan")
	success_rate = float(np.mean(success_flags)) if success_flags else 0.0
	print(
		f"Finished {len(episode_rewards)} episode(s). "
		f"Reward mean={mean_reward:.2f} ± {std_reward:.2f}, success_rate={success_rate * 100:.1f}%"
	)


def _parse_args():
	parser = argparse.ArgumentParser(description="Evaluate BC policy in robomimic Can environment")
	parser.add_argument("--configs", nargs="+", default=["bc_defaults"], help="Config presets to load")
	parser.add_argument("--policy_path", type=pathlib.Path, required=True, help="Path to saved bc_action_mlp.pt")
	parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
	parser.add_argument("--max_env_steps", type=int, default=500, help="Max steps per episode before truncation")
	parser.add_argument("--video_dir", type=str, default="", help="Directory to store MP4 rollouts (disabled if empty)")
	parser.add_argument("--video_fps", type=int, default=20, help="FPS for saved videos")
	parser.add_argument("--video_camera", type=str, default=None, help="Observation key used for video frames")
	parser.add_argument(
		"--video_camera_keys",
		nargs="+",
		default=None,
		help="Observation keys to tile together in saved videos (defaults to --camera_obs_keys when --video_camera is unset)",
	)
	parser.add_argument("--image_size", type=int, nargs=2, default=[84, 84], help="Camera height width")
	parser.add_argument(
		"--flip_camera_keys",
		nargs="+",
		default=[],
		help="Observation keys whose frames should be flipped vertically and horizontally before stacking",
	)
	parser.add_argument(
		"--camera_obs_keys",
		nargs="+",
		default=["agentview_image", "robot0_eye_in_hand_image"],
		help="Observation keys to stack into the Dreamer image encoder",
	)
	parser.add_argument(
		"--mlp_obs_keys",
		nargs="+",
		default=[
			"robot0_joint_pos",
			"robot0_joint_vel",
			"robot0_gripper_qpos",
			"robot0_gripper_qvel",
		],
		help="Observation keys fed through the encoder MLP",
	)
	parser.add_argument("--clip_actions", action="store_true", help="Clip policy actions to [-1, 1]")
	parser.add_argument("--render", action="store_true", help="Enable robosuite on-screen rendering")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--robosuite_task", type=str, default="PickPlaceCan", help="Robosuite task name")
	parser.add_argument("--robosuite_robots", nargs="+", default=["Panda"], help="Robots to instantiate")
	parser.add_argument("--robosuite_controller", type=str, default="OSC_POSE")
	parser.add_argument("--robosuite_reward_shaping", action="store_true")
	parser.add_argument("--robosuite_control_freq", type=int, default=20)
	parser.add_argument("--bc_hidden_units", type=int, default=1024)
	parser.add_argument("--bc_hidden_layers", type=int, default=4)
	parser.add_argument("--bc_activation", type=str, default="SiLU")
	parser.add_argument("--bc_use_layernorm", action="store_true")
	parser.add_argument("--dataset_path", type=str, default="datasets/canPH_raw.hdf5", help="Path to HDF5 dataset for initial states")
	parser.add_argument(
		"--dataset_obs_path",
		type=str,
		default="datasets/imagecanPH_v15.hdf5",
		help="HDF5 dataset containing observation tensors for warmup comparison (defaults to image dataset)",
	)
	parser.add_argument(
		"--replay_dataset_demo",
		action="store_true",
		help="Before policy rollouts, replay the specified dataset demo open-loop and save video/action plots",
	)
	parser.add_argument(
		"--replay_demo_key",
		type=str,
		default="",
		help="Dataset demo key to replay (defaults to the first demo in --dataset_path)",
	)
	parser.add_argument(
		"--replay_max_steps",
		type=int,
		default=0,
		help="Optional maximum number of steps to replay (0 uses the full demo length)",
	)
	parser.add_argument(
		"--encoder_snapshot_file",
		type=str,
		default="",
		help="File to store encoder input snapshot for the first dataset demo (empty to disable)",
	)
	args, remaining = parser.parse_known_args()
	config = _load_config(args.configs, remaining)

	policy_path = args.policy_path.expanduser()
	if not policy_path.exists():
		raise FileNotFoundError(f"Policy checkpoint not found at {policy_path}")
	config.size = tuple(args.image_size)
	checkpoint_value = getattr(config, "checkpoint", "")
	if not checkpoint_value:
		raise ValueError("--checkpoint must be provided to load the Dreamer encoder weights")
	checkpoint_path = pathlib.Path(checkpoint_value).expanduser()
	video_dir = pathlib.Path(args.video_dir).expanduser() if args.video_dir else None
	video_camera_keys = tuple(args.video_camera_keys) if args.video_camera_keys else None
	if args.video_camera is None and video_camera_keys is None:
		video_camera_keys = tuple(args.camera_obs_keys)
	video_camera = args.video_camera
	dataset_path = pathlib.Path(args.dataset_path).expanduser() if args.dataset_path else None
	dataset_obs_path = pathlib.Path(args.dataset_obs_path).expanduser() if args.dataset_obs_path else None
	encoder_snapshot_path = pathlib.Path(args.encoder_snapshot_file).expanduser() if args.encoder_snapshot_file else None
	replay_demo_key = args.replay_demo_key.strip() or None
	replay_max_steps = args.replay_max_steps if args.replay_max_steps > 0 else None

	eval_cfg = EvalConfig(
		policy_path=policy_path,
		checkpoint=checkpoint_path,
		episodes=args.episodes,
		max_env_steps=args.max_env_steps,
		camera_obs_keys=tuple(args.camera_obs_keys),
		flip_camera_keys=tuple(args.flip_camera_keys),
		mlp_obs_keys=tuple(args.mlp_obs_keys),
		camera_image_key="image",
		clip_actions=args.clip_actions,
		render=args.render,
		device=getattr(config, "device", "cuda:0"),
		seed=args.seed,
		robosuite_task=args.robosuite_task,
		robosuite_robots=tuple(args.robosuite_robots),
		robosuite_controller=args.robosuite_controller,
		robosuite_reward_shaping=args.robosuite_reward_shaping,
		robosuite_control_freq=args.robosuite_control_freq,
		bc_hidden_units=args.bc_hidden_units,
		bc_hidden_layers=args.bc_hidden_layers,
		bc_activation=args.bc_activation,
		bc_use_layernorm=args.bc_use_layernorm,
		video_dir=video_dir,
		video_fps=args.video_fps,
		video_camera=video_camera,
		video_camera_keys=video_camera_keys,
		dataset_path=dataset_path,
		dataset_obs_path=dataset_obs_path,
		encoder_snapshot_path=encoder_snapshot_path,
		replay_dataset_demo=args.replay_dataset_demo,
		replay_demo_key=replay_demo_key,
		replay_max_steps=replay_max_steps,
	)

	config.camera_obs_keys = eval_cfg.camera_obs_keys
	config.mlp_obs_keys = eval_cfg.mlp_obs_keys
	config.robosuite_task = eval_cfg.robosuite_task
	config.robosuite_robots = eval_cfg.robosuite_robots
	config.robosuite_controller = eval_cfg.robosuite_controller
	config.robosuite_reward_shaping = eval_cfg.robosuite_reward_shaping
	config.robosuite_control_freq = eval_cfg.robosuite_control_freq
	config.flip_camera_keys = eval_cfg.flip_camera_keys
	config.bc_hidden_units = eval_cfg.bc_hidden_units
	config.bc_hidden_layers = eval_cfg.bc_hidden_layers
	config.bc_activation = eval_cfg.bc_activation
	config.bc_use_layernorm = eval_cfg.bc_use_layernorm
	config.video_dir = eval_cfg.video_dir
	config.video_fps = eval_cfg.video_fps
	config.video_camera = eval_cfg.video_camera
	config.video_camera_keys = eval_cfg.video_camera_keys

	eval_cfg.checkpoint = checkpoint_path
	return eval_cfg, config


def main():
	eval_cfg, dreamer_cfg = _parse_args()
	evaluate_policy(eval_cfg, dreamer_cfg)


if __name__ == "__main__":
	main()

