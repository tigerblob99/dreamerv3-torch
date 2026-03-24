#!/usr/bin/env python3
"""Quick reward plots for Dreamer-style `.npz` episode directories.

This script only reads the `reward` array from each `.npz` file, so it stays
fast even when episodes contain large image observations.

Example:
  python plot_npz_rewards.py \
    --dir datasets/robomimic_data_MV/can_PH_Shaped_shifted_0-1 \
    --out rewards_summary.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: numpy. Install it in your env and rerun."
        ) from exc
    return np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: matplotlib. Install it in your env and rerun."
        ) from exc
    return plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="datasets/robomimic_data_MV/can_PH_Shaped_shifted_0-1",
        help="Directory containing episode .npz files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="rewards_summary.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--plot_episodes",
        type=int,
        default=12,
        help="Number of episode reward curves to overlay in the time-series plot.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Limit number of .npz files to scan (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to sample episodes for the overlay plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window (requires a display).",
    )
    args = parser.parse_args()

    np = _require_numpy()
    plt = _require_matplotlib()

    directory = Path(args.dir).expanduser()
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found in {directory}")
    if args.max_episodes and args.max_episodes > 0:
        files = files[: int(args.max_episodes)]

    rewards = []
    returns = []
    lengths = []
    skipped = 0

    for filename in files:
        try:
            with np.load(filename) as data:
                if "reward" not in data:
                    skipped += 1
                    continue
                r = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
        except Exception:
            skipped += 1
            continue
        if r.size == 0:
            skipped += 1
            continue
        rewards.append(r)
        returns.append(float(r.sum()))
        lengths.append(int(r.shape[0]))

    if not rewards:
        raise SystemExit("No valid episodes with `reward` found.")

    all_rewards = np.concatenate(rewards, axis=0)
    returns_arr = np.asarray(returns, dtype=np.float32)
    lengths_arr = np.asarray(lengths, dtype=np.int32)

    print(f"Episodes scanned: {len(files)}")
    print(f"Episodes used:   {len(rewards)} (skipped={skipped})")
    print(
        "Reward stats:    "
        f"min={all_rewards.min():.4g} mean={all_rewards.mean():.4g} "
        f"std={all_rewards.std():.4g} max={all_rewards.max():.4g}"
    )
    print(
        "Return stats:    "
        f"min={returns_arr.min():.4g} mean={returns_arr.mean():.4g} "
        f"std={returns_arr.std():.4g} max={returns_arr.max():.4g}"
    )
    print(
        "Length stats:    "
        f"min={lengths_arr.min()} mean={lengths_arr.mean():.3g} "
        f"max={lengths_arr.max()}"
    )

    rng = random.Random(int(args.seed))
    indices = list(range(len(rewards)))
    rng.shuffle(indices)
    plot_n = max(1, min(int(args.plot_episodes), len(indices)))
    plot_indices = indices[:plot_n]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_ts, ax_hist, ax_ret, ax_len = axes.flat

    for idx in plot_indices:
        ax_ts.plot(rewards[idx], alpha=0.75, linewidth=1.2)
    ax_ts.set_title(f"Reward vs time (sampled {plot_n} episodes)")
    ax_ts.set_xlabel("t")
    ax_ts.set_ylabel("reward")
    ax_ts.grid(alpha=0.2)

    ax_hist.hist(all_rewards, bins=100)
    ax_hist.set_title("Per-step reward distribution")
    ax_hist.set_xlabel("reward")
    ax_hist.set_ylabel("count")

    ax_ret.hist(returns_arr, bins=50)
    ax_ret.set_title("Episode return distribution (sum of rewards)")
    ax_ret.set_xlabel("return")
    ax_ret.set_ylabel("episodes")

    ax_len.hist(lengths_arr, bins=50)
    ax_len.set_title("Episode length distribution")
    ax_len.set_xlabel("T")
    ax_len.set_ylabel("episodes")

    fig.tight_layout()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=150)
    print(f"Wrote plot: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

