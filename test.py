#!/usr/bin/env python3

"""Utility script to inspect dataset observations at a single timestep."""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Iterable, Tuple

import h5py  # type: ignore
import numpy as np  # type: ignore

DEFAULT_OUTPUT = pathlib.Path("notes/first_obs.txt")
NON_OBS_KEYS = {"action", "reward", "discount"}


def _array_to_string(
	arr: np.ndarray,
	*,
	preview: int,
	force_full: bool = False,
) -> str:
	if force_full or preview < 0:
		return np.array2string(
			arr,
			precision=4,
			suppress_small=True,
			separator=", ",
			max_line_width=1_000_000,
			threshold=arr.size,
		)
	flat = arr.reshape(-1)
	if flat.size == 0:
		return "[]"
	trim = flat[: min(preview, flat.size)]
	return np.array2string(trim, precision=4, suppress_small=True)



def _format_snapshot(
	label: str,
	values: Dict[str, np.ndarray],
	*,
	preview: int,
	full_keys: Iterable[str] | None = None,
) -> str:
	full_set = set(full_keys or [])
	lines = [f"[snapshot] {label}: {len(values)} keys"]
	for key in sorted(values.keys()):
		arr = np.asarray(values[key])
		sample = _array_to_string(arr, preview=preview, force_full=(key in full_set))
		lines.append(
			f"  - {key}: shape={arr.shape}, dtype={arr.dtype}, sample={sample}"
		)
	return "\n".join(lines)


def _select_frame(arr: np.ndarray, idx: int) -> np.ndarray:
	if arr.ndim == 0:
		return arr
	if arr.shape[0] == 0:
		return arr
	clamped = max(0, min(idx, arr.shape[0] - 1))
	return arr[clamped]


def _resolve_keys(all_keys: Iterable[str], requested: Iterable[str] | None) -> Tuple[str, ...]:
	all_keys = tuple(all_keys)
	if requested:
		missing = sorted(set(requested) - set(all_keys))
		if missing:
			raise KeyError(f"Requested keys missing from dataset: {missing}")
		return tuple(requested)
	filtered = [key for key in sorted(all_keys) if key not in NON_OBS_KEYS]
	return tuple(filtered)


def _load_hdf5_snapshot(
	path: pathlib.Path,
	demo: str,
	frame: int,
	group: str,
	keys: Iterable[str] | None,
) -> Dict[str, np.ndarray]:
	with h5py.File(path, "r") as f:
		if "data" not in f:
			raise KeyError("HDF5 file is missing 'data' group")
		data_group = f["data"]
		if demo not in data_group:
			raise KeyError(f"Trajectory '{demo}' not found in {path}")
		demo_group = data_group[demo]
		if group not in demo_group:
			raise KeyError(f"Group '{group}' missing under trajectory '{demo}'")
		obs_group = demo_group[group]
		keys_to_dump = _resolve_keys(obs_group.keys(), keys)
		snapshot: Dict[str, np.ndarray] = {}
		for key in keys_to_dump:
			data = obs_group[key][()]
			data = np.asarray(data)
			clip = _select_frame(data, frame)
			snapshot[key] = clip
		return snapshot


def _resolve_npz_file(dataset_path: pathlib.Path, demo: str) -> pathlib.Path:
	if dataset_path.is_file():
		return dataset_path
	matches = sorted(dataset_path.glob(f"{demo}-*.npz"))
	if not matches:
		raise FileNotFoundError(
			f"Could not find file for demo '{demo}' inside {dataset_path} (expected pattern {demo}-*.npz)"
		)
	if len(matches) > 1:
		print(
			f"[warn] Multiple files match demo '{demo}' in {dataset_path}; using {matches[0].name}"
		)
	return matches[0]


def _load_npz_snapshot(path: pathlib.Path, frame: int, keys: Iterable[str] | None) -> Dict[str, np.ndarray]:
	with np.load(path, allow_pickle=False) as data:
		keys_to_dump = _resolve_keys(data.files, keys)
		snapshot: Dict[str, np.ndarray] = {}
		for key in keys_to_dump:
			arr = np.asarray(data[key])
			clip = _select_frame(arr, frame) if arr.ndim >= 1 else arr
			snapshot[key] = clip
		return snapshot


def main() -> None:
	parser = argparse.ArgumentParser(description="Print observation snapshot from dataset")
	parser.add_argument(
		"--dataset",
		type=pathlib.Path,
		default=pathlib.Path("datasets/robomimic_data_MV/can_PH_train"),
		help="Path to a dataset directory (.npz episodes) or HDF5 file",
	)
	parser.add_argument("--demo", type=str, default="demo_0", help="Trajectory/demo identifier")
	parser.add_argument("--frame", type=int, default=0, help="Timestep index to inspect (0-based)")
	parser.add_argument(
		"--group",
		type=str,
		default="obs",
		help="For HDF5 datasets, which group under the trajectory to read (e.g., obs or next_obs)",
	)
	parser.add_argument("--keys", nargs="+", default=None, help="Optional list of keys to print")
	parser.add_argument(
		"--preview",
		type=int,
		default=-1,
		help="Number of flattened values to show per key",
	)
	parser.add_argument(
		"--full-keys",
		nargs="+",
		default=["image"],
		help="Keys that should be dumped in full (no preview). Use empty string to disable",
	)
	parser.add_argument(
		"--output",
		type=pathlib.Path,
		default=DEFAULT_OUTPUT,
		help="Where to save the formatted snapshot (use '-' for stdout)",
	)
	args = parser.parse_args()

	dataset_path = args.dataset.expanduser()
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

	frame = max(0, args.frame)
	keys = args.keys

	if dataset_path.is_dir():
		npz_candidates = sorted(dataset_path.glob("*.npz"))
		if not npz_candidates:
			raise FileNotFoundError(
				f"Directory {dataset_path} does not contain any .npz files; provide an HDF5 file instead"
			)
		resolved = _resolve_npz_file(dataset_path, args.demo)
		snapshot = _load_npz_snapshot(resolved, frame, keys)
		label = f"NPZ {resolved.name} frame={frame}"
	else:
		suffix = dataset_path.suffix.lower()
		if suffix == ".npz":
			resolved = dataset_path
			snapshot = _load_npz_snapshot(resolved, frame, keys)
			label = f"NPZ {resolved.name} frame={frame}"
		elif suffix in {".h5", ".hdf5"}:
			snapshot = _load_hdf5_snapshot(dataset_path, args.demo, frame, args.group, keys)
			label = f"HDF5 {dataset_path.name}::{args.demo}/{args.group} frame={frame}"
		else:
			raise ValueError(
				f"Unsupported dataset type '{dataset_path.suffix}'. Use .npz/.hdf5 or a directory of .npz files"
			)

	full_keys = [] if args.full_keys == [""] else args.full_keys
	formatted = _format_snapshot(label, snapshot, preview=args.preview, full_keys=full_keys)
	if str(args.output) == "-":
		print(formatted)
	else:
		output_path = args.output.expanduser()
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(formatted + "\n")
		print(f"[saved] snapshot written to {output_path}")


if __name__ == "__main__":
	main()
