import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=pathlib.Path, required=True)
    parser.add_argument("--dataset_limit", type=int, default=0, help="Optional frame limit; 0 means all.")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("dataset_mean.png"))
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    limit = args.dataset_limit if args.dataset_limit > 0 else None
    print(f"Loading episodes from {dataset_dir} for stats (limit={limit}).")
    train_eps = tools.load_episodes(dataset_dir, limit=limit)
    ds_mean, ds_std = tools.compute_image_dataset_stats(train_eps)
    print(
        f"Mean image: shape {ds_mean.shape}, min {ds_mean.min():.4f}, max {ds_mean.max():.4f}, avg {ds_mean.mean():.4f}"
    )
    print(
        f"Std image: shape {ds_std.shape}, min {ds_std.min():.4f}, max {ds_std.max():.4f}, avg {ds_std.mean():.4f}"
    )
    mean_img = np.clip(ds_mean[0], 0.0, 1.0)
    c = mean_img.shape[-1]
    # Show first three channels and, if available, the next three side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    vis1 = mean_img[..., :3] if c >= 3 else mean_img
    axes[0].imshow(vis1)
    axes[0].set_title("Channels 0-2")
    axes[0].axis("off")
    if c >= 6:
        vis2 = mean_img[..., 3:6]
        axes[1].imshow(vis2)
        axes[1].set_title("Channels 3-5")
        axes[1].axis("off")
    else:
        axes[1].axis("off")
        axes[1].set_title("Channels 3-5 (missing)")
    out_path = args.out.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved dataset mean visualization to {out_path}")


if __name__ == "__main__":
    main()
