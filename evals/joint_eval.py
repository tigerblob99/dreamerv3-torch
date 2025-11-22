#!/usr/bin/env python3
"""Joint-angle open-loop evaluation utility for DreamerV3."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import ruamel.yaml as yaml
import torch

import tools
from dreamer import Dreamer


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs.yaml"
FAN_WARMUP_STEPS = 1


def _recursive_update(base: MutableMapping, update: Mapping) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and key in base and isinstance(base[key], Mapping):
            _recursive_update(base[key], value)
        else:
            base[key] = value


def _load_config_defaults(config_names: Sequence[str] | None) -> Dict[str, Any]:
    config_data = yaml.safe_load(CONFIG_PATH.read_text())
    names = ["defaults", *(config_names or [])]
    defaults: Dict[str, Any] = {}
    for name in names:
        if name not in config_data:
            raise KeyError(f"Config '{name}' not found in {CONFIG_PATH}")
        _recursive_update(defaults, config_data[name])
    defaults.setdefault("eval_only", True)
    defaults.setdefault("parallel", False)
    defaults.setdefault("compile", False)
    return defaults


def _infer_spaces(episode: Mapping[str, Any]) -> Tuple[Any, Any]:
    observation_spaces = {}
    ignored = {"action", "reward", "discount"}
    for key, value in episode.items():
        if key.startswith("log_") or key in ignored:
            continue
        arr = np.asarray(value)
        if arr.ndim <= 1:
            shape = (1,)
        else:
            shape = arr.shape[1:]
        dtype = arr.dtype
        if np.issubdtype(dtype, np.floating):
            low, high = -np.inf, np.inf
        elif np.issubdtype(dtype, np.bool_):
            low, high = 0.0, 1.0
            dtype = np.float32
        elif np.issubdtype(dtype, np.unsignedinteger):
            low, high = 0, np.iinfo(dtype).max
        else:
            low, high = -np.inf, np.inf
        observation_spaces[key] = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    if "action" not in episode:
        raise KeyError("Episode file does not contain an 'action' key")
    act = np.asarray(episode["action"])
    if np.issubdtype(act.dtype, np.integer) and act.ndim == 2 and act.shape[-1] == 1:
        action_space = gym.spaces.Discrete(int(np.max(act) + 1))
    else:
        act_shape = act.shape[1:] if act.ndim > 1 else (1,)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=act_shape, dtype=np.float32)
    return gym.spaces.Dict(observation_spaces), action_space


def _load_episode(path: pathlib.Path) -> Dict[str, Any]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _resolve_episode_file(episode_path: str) -> Tuple[pathlib.Path, Dict[str, Any]]:
    path = pathlib.Path(episode_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix != ".npz":
        raise ValueError(f"Episode path must point to a .npz file: {path}")
    return path, _load_episode(path)


def _prepare_sequence(episode: Mapping[str, Any], device: torch.device, max_steps: int | None) -> Dict[str, Any]:
    tensors: Dict[str, Any] = {}
    for key, value in episode.items():
        if key.startswith("log_"):
            continue
        arr = np.asarray(value)
        if max_steps is not None:
            arr = arr[:max_steps]
        tensor = torch.tensor(arr, device=device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        tensors[key] = tensor.unsqueeze(0)
    return tensors


def _build_agent(obs_space: Any, act_space: Any, config, logdir: pathlib.Path) -> Dreamer:
    logger = tools.Logger(logdir, step=0)
    config.num_actions = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    agent = Dreamer(obs_space, act_space, config, logger, dataset=None).to(config.device)
    agent.eval()
    agent.requires_grad_(False)
    return agent


def _load_checkpoint(agent: Dreamer, checkpoint_path: pathlib.Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=agent._config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])


def _decode_tensor(dist) -> torch.Tensor:
    if hasattr(dist, "mode"):
        return dist.mode()
    if hasattr(dist, "mean"):
        return dist.mean()
    return dist


def _compute_fan_rollouts(
    world_model,
    start_steps: Sequence[int],
    data: Mapping[str, torch.Tensor],
    embed: torch.Tensor,
    joint_keys: Sequence[str],
    horizon: int,
    warmup_steps: int = FAN_WARMUP_STEPS,
) -> Dict[str, np.ndarray]:
    actions = data["action"]
    total_steps = actions.shape[1]
    arrays: Dict[str, np.ndarray] = {}
    key_shapes: Dict[str, Tuple[int, ...]] = {}
    for key in joint_keys:
        if key not in data:
            raise KeyError(f"Episode is missing observation key '{key}' required for fan rollout")
        trailing = tuple(int(dim) for dim in data[key].shape[2:])
        if not trailing:
            trailing = (1,)
        key_shapes[key] = trailing
        arrays[key] = np.full((total_steps, horizon, *trailing), np.nan, dtype=np.float32)

    is_first = data.get("is_first")
    if is_first is None:
        raise KeyError("Dataset tensor 'is_first' is required for fan rollouts")

    def _extract_is_first(step: int) -> torch.Tensor:
        value = is_first[:, step]
        if value.ndim > 2:
            value = value.squeeze(-1)
        if value.dtype != torch.bool:
            value = value > 0.5
        return value

    for start in start_steps:
        if start >= total_steps - 1:
            continue
        valid_h = min(horizon, total_steps - start - 1)
        if valid_h <= 0:
            continue
        warmup_start = max(0, start - warmup_steps + 1)
        state = None
        for idx in range(warmup_start, start + 1):
            action_obs = actions[:, idx]
            embed_obs = embed[:, idx]
            first_flag = _extract_is_first(idx)
            if idx == warmup_start:
                first_flag = torch.ones_like(first_flag, dtype=torch.bool)
            state, _ = world_model.dynamics.obs_step(state, action_obs, embed_obs, first_flag, sample=False)
        for j in range(valid_h):
            action_idx = start + j
            action = actions[:, action_idx]
            state = world_model.dynamics.img_step(state, action, sample=False)
            feat = world_model.dynamics.get_feat(state)
            added_time_dim = False
            if feat.ndim == 2:
                feat = feat.unsqueeze(1)
                added_time_dim = True
            decoded = world_model.heads["decoder"](feat)
            for key in joint_keys:
                if key not in decoded:
                    raise KeyError(f"Decoder does not predict key '{key}'")
                pred = _decode_tensor(decoded[key])
                if added_time_dim and pred.ndim >= 3:
                    pred = pred[:, 0]
                arrays[key][start, j] = pred.squeeze(0).detach().cpu().numpy().reshape(key_shapes[key])

    return {f"fan_pred_{key}": value for key, value in arrays.items()}


@dataclass
class JointEvalSummary:
    episode_name: str
    num_steps: int
    correction_interval: int
    joint_keys: Sequence[str]
    per_key_rmse: Dict[str, float]
    per_key_mae: Dict[str, float]
    per_horizon_mse: Dict[int, float]
    last_step_errors: Dict[str, float]


def run_joint_rollout(
    world_model,
    sequence: Mapping[str, Any],
    joint_keys: Sequence[str],
    correction_interval: int,
    fan_horizon: int = 0,
) -> Tuple[JointEvalSummary, Dict[str, Any]]:
    if correction_interval <= 0:
        raise ValueError("correction_interval must be positive")
    missing = [key for key in joint_keys if key not in sequence]
    if missing:
        raise KeyError(f"Episode is missing observation keys: {missing}")

    with torch.no_grad():
        data = world_model.preprocess(dict(sequence))
        embed = world_model.encoder(data)
        actions = data["action"]
        total_steps = actions.shape[1]
        joint_predictions: Dict[str, List[np.ndarray]] = {k: [] for k in joint_keys}
        joint_truth: Dict[str, List[np.ndarray]] = {k: [] for k in joint_keys}
        per_key_sqerr: Dict[str, List[float]] = {k: [] for k in joint_keys}
        per_key_abserr: Dict[str, List[float]] = {k: [] for k in joint_keys}
        horizon_bins: Dict[int, List[float]] = collections.defaultdict(list)
        correction_mask: List[bool] = []
        horizons: List[int] = []
        fan_start_steps: List[int] = []

        state = None

        for t in range(total_steps):
            action_t = actions[:, t]
            is_first = data["is_first"][:, t]
            is_first = is_first.squeeze(-1) > 0.5 if is_first.ndim > 2 else is_first.bool()
            state, _ = world_model.dynamics.obs_step(state, action_t, embed[:, t], is_first, sample=False)

            feat = world_model.dynamics.get_feat(state)
            added_time_dim = False
            if feat.ndim == 2:
                feat = feat.unsqueeze(1)
                added_time_dim = True
            decoded = world_model.heads["decoder"](feat)
            step_mse_values = []

            for key in joint_keys:
                if key not in decoded:
                    raise KeyError(f"Decoder does not predict key '{key}'")
                pred = _decode_tensor(decoded[key])
                if added_time_dim and pred.ndim >= 3:
                    pred = pred[:, 0]
                pred = pred.squeeze(0)
                truth = data[key][:, t].squeeze(0)
                diff = pred - truth
                mse = float(torch.mean(diff**2).cpu().item())
                mae = float(torch.mean(torch.abs(diff)).cpu().item())
                joint_predictions[key].append(pred.cpu().numpy())
                joint_truth[key].append(truth.cpu().numpy())
                per_key_sqerr[key].append(mse)
                per_key_abserr[key].append(mae)
                step_mse_values.append(mse)

            steps_since = t % correction_interval
            needs_fan = steps_since == 0
            horizon_bins[steps_since].append(float(np.mean(step_mse_values)))
            correction_mask.append(needs_fan)
            horizons.append(steps_since)

            if fan_horizon > 0 and needs_fan and t < total_steps - 1:
                fan_start_steps.append(t)

    per_key_rmse = {k: float(np.sqrt(np.mean(per_key_sqerr[k]))) for k in joint_keys}
    per_key_mae = {k: float(np.mean(per_key_abserr[k])) for k in joint_keys}
    last_step_errors = {k: float(per_key_sqerr[k][-1]) for k in joint_keys}
    per_horizon_mse = {h: float(np.mean(vals)) for h, vals in sorted(horizon_bins.items())}

    arrays: Dict[str, Any] = {
        **{f"pred_{k}": np.stack(v, axis=0) for k, v in joint_predictions.items()},
        **{f"truth_{k}": np.stack(v, axis=0) for k, v in joint_truth.items()},
        "correction_mask": np.array(correction_mask, dtype=bool),
        "steps_since_correction": np.array(horizons, dtype=np.int32),
    }

    if fan_horizon > 0:
        fan_arrays = _compute_fan_rollouts(
            world_model,
            fan_start_steps,
            data,
            embed,
            joint_keys,
            fan_horizon,
            warmup_steps=FAN_WARMUP_STEPS,
        )
        fan_arrays["fan_horizon"] = np.array(fan_horizon, dtype=np.int32)
        arrays.update(fan_arrays)

    summary = JointEvalSummary(
        episode_name="",
        num_steps=int(actions.shape[1]),
        correction_interval=correction_interval,
        joint_keys=list(joint_keys),
        per_key_rmse=per_key_rmse,
        per_key_mae=per_key_mae,
        per_horizon_mse=per_horizon_mse,
        last_step_errors=last_step_errors,
    )
    return summary, arrays


def _print_summary(summary: JointEvalSummary) -> None:
    print(f"Episode: {summary.episode_name}")
    print(f"Steps evaluated: {summary.num_steps}")
    print(f"Correction interval: {summary.correction_interval}")
    print("Per-key errors (RMSE / MAE):")
    for key in summary.joint_keys:
        print(f"  {key:<24} rmse={summary.per_key_rmse[key]:.6f}  mae={summary.per_key_mae[key]:.6f}")
    print("Average MSE grouped by open-loop horizon:")
    for horizon, mse in summary.per_horizon_mse.items():
        print(f"  +{horizon:02d} steps -> mse={mse:.6f}")


def _save_outputs(args: argparse.Namespace, summary: JointEvalSummary, arrays: Mapping[str, Any]) -> None:
    if args.output_json:
        out = pathlib.Path(args.output_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(summary), indent=2))
    if args.save_npz:
        npz = pathlib.Path(args.save_npz).expanduser()
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **arrays)


def _plot_joint_trajectories(
    episode_name: str,
    joint_keys: Sequence[str],
    arrays: Mapping[str, Any],
    output_dir: pathlib.Path,
) -> List[pathlib.Path]:
    output_dir = pathlib.Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[pathlib.Path] = []
    fan_horizon_value = None
    if "fan_horizon" in arrays:
        fan_horizon_value = int(np.asarray(arrays["fan_horizon"]).item())
    for key in joint_keys:
        truth_key = f"truth_{key}"
        if truth_key not in arrays:
            continue
        truth = np.asarray(arrays[truth_key])
        steps = truth.shape[0]
        truth_flat = truth.reshape(steps, -1)
        pred_key = f"pred_{key}"
        pred_flat = None
        if pred_key in arrays:
            pred = np.asarray(arrays[pred_key])
            if pred.shape != truth.shape:
                raise ValueError(f"Prediction and truth shapes differ for {key}: {pred.shape} vs {truth.shape}")
            pred_flat = pred.reshape(steps, -1)
        fan_key = f"fan_pred_{key}"
        fan_flat = None
        if fan_key in arrays:
            fan = np.asarray(arrays[fan_key])
            fan_flat = fan.reshape(fan.shape[0], fan.shape[1], -1)
        dims = truth_flat.shape[1]
        cols = 1
        rows = dims
        fig, axes = plt.subplots(rows, cols, figsize=(6, rows * 3.2), sharex=True)
        axes = np.atleast_1d(axes).flatten()
        time = np.arange(steps)
        fan_label_added = False
        pred_label_added = False
        for dim in range(dims):
            ax = axes[dim]
            ax.plot(time, truth_flat[:, dim], label="ground truth", linewidth=1.2, color="#1f77b4")
            if fan_flat is not None:
                horizon = fan_flat.shape[1]
                for start in range(steps):
                    seq = fan_flat[start, :, dim]
                    mask = np.isfinite(seq)
                    if not mask.any():
                        continue
                    times = start + np.arange(horizon)
                    ax.plot(
                        times[mask],
                        seq[mask],
                        color="#ff0e0e",
                        linewidth=2.0,
                        alpha=0.3,
                        label="_nolegend_",
                    )
                if not fan_label_added:
                    label = (
                        f"fan rollout (h={fan_horizon_value})"
                        if fan_horizon_value
                        else "fan rollout"
                    )
                    ax.plot([], [], color="#ff7f0e", linewidth=0.8, alpha=0.3, label=label)
                    fan_label_added = True
            if pred_flat is not None:
                label = "recreated from posterior" if not pred_label_added else "_nolegend_"
                ax.plot(
                    time,
                    pred_flat[:, dim],
                    label=label,
                    linewidth=1.0,
                    color="#ff7f0e",
                    linestyle="--",
                )
                if not pred_label_added:
                    pred_label_added = True
            ax.set_title(f"{key} [dim {dim}]")
            ax.set_xlabel("step")
            ax.set_ylabel("angle")
        for ax in axes[dims:]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(1, 0.97),
                borderaxespad=0.5,
            )
        fig.suptitle(f"{episode_name} – {key} joint trajectories")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        safe_key = key.replace("/", "_")
        filename = f"{episode_name}_{safe_key}.png" if episode_name else f"{safe_key}.png"
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(output_path)
    return saved_paths


def parse_args() -> Tuple[argparse.Namespace, argparse.Namespace]:
    stage_one = argparse.ArgumentParser(add_help=False)
    stage_one.add_argument("--configs", nargs="+")
    parsed_stage_one, remaining = stage_one.parse_known_args()
    defaults = _load_config_defaults(parsed_stage_one.configs)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=parsed_stage_one.configs)
    parser.add_argument("--logdir", required=True, help="Path to the Dreamer run directory.")
    parser.add_argument("--checkpoint", type=str, help="Optional override for checkpoint path.")
    parser.add_argument(
        "--episode-path",
        required=True,
        help="Path to the saved episode (.npz) to evaluate.",
    )
    parser.add_argument("--correction-interval", type=int, default=10, help="Steps between observation corrections.")
    parser.add_argument("--joint-keys", nargs="+", default=["robot0_joint_pos"], help="Observation keys to track.")
    parser.add_argument("--max-steps", type=int, help="Optional limit on the number of evaluated steps.")
    parser.add_argument("--output-json", type=str, help="Optional path to dump the summary as JSON.")
    parser.add_argument("--save-npz", type=str, help="Optional path to persist per-step arrays.")
    parser.add_argument("--plot-dir", type=str, help="Directory to store joint trajectory plots (defaults inside logdir).")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating trajectory plots.")
    parser.add_argument(
        "--fan-horizon",
        type=int,
        default=0,
        help="If >0, generate an [n,h,d] open-loop prediction fan of length h for each start step.",
    )
    reserved = {action.dest for action in parser._actions}
    for key, value in sorted(defaults.items(), key=lambda item: item[0]):
        if key in reserved:
            continue
        parser.add_argument(f"--{key}", type=tools.args_type(value), default=value)
    args = parser.parse_args(remaining)
    config_values = {k: getattr(args, k) for k in defaults.keys()}
    config = argparse.Namespace(**config_values)
    return args, config


def main() -> None:
    args, config = parse_args()
    logdir = pathlib.Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.logdir = str(logdir)
    config.traindir = config.traindir or (logdir / "train_eps")
    config.evaldir = config.evaldir or (logdir / "eval_eps")
    config.offline_traindir = config.offline_traindir or ""
    config.offline_evaldir = config.offline_evaldir or ""

    checkpoint_path = pathlib.Path(args.checkpoint or (logdir / "latest.pt")).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    episode_path, episode = _resolve_episode_file(args.episode_path)
    obs_space, act_space = _infer_spaces(episode)
    agent = _build_agent(obs_space, act_space, config, logdir / "joint_eval_logs")
    _load_checkpoint(agent, checkpoint_path)

    device = torch.device(config.device)
    sequence = _prepare_sequence(episode, device=device, max_steps=args.max_steps)
    summary, arrays = run_joint_rollout(
        agent._wm,
        sequence,
        joint_keys=args.joint_keys,
        correction_interval=args.correction_interval,
        fan_horizon=max(0, args.fan_horizon),
    )
    summary.episode_name = episode_path.stem
    _print_summary(summary)
    _save_outputs(args, summary, arrays)
    if not args.no_plot:
        plot_dir = pathlib.Path(args.plot_dir or (logdir / "joint_eval_plots"))
        saved = _plot_joint_trajectories(summary.episode_name, args.joint_keys, arrays, plot_dir)
        if saved:
            print(f"Saved {len(saved)} joint trajectory plot(s) to {plot_dir}")
        else:
            print("No joint trajectory plots were generated (missing prediction keys).")


if __name__ == "__main__":
    main()

