"""RL dataset that yields complete, single-episode sequences of fixed length.

Unlike JointDataset (which stitches fragments from multiple episodes),
every sample here is a contiguous sub-sequence of exactly ``batch_length``
steps drawn from one episode.  The reward signal from the expert
demonstrations is preserved and returned alongside observations, actions,
and flags.

Cropping behaviour:
  * **train** mode  → random crop (per-sample, consistent across timesteps)
  * **eval**  mode  → deterministic center crop
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

# Ensure the repo root is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import tools

_EXCLUDE_KEYS = {"action", "reward", "discount", "is_first", "is_terminal"}


def _episode_start_weights(ep_len: int, batch_length: int) -> np.ndarray:
    """Return uniform weights for all start indices.

    With wrap-around windows every start ``0..ep_len-1`` is valid and equally
    representative, so uniform weights suffice.
    """
    ep_len = int(ep_len)
    if ep_len == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.ones((ep_len,), dtype=np.float64)


class RlDataset(Dataset):
    """PyTorch Dataset of fixed-length, single-episode sequences.

    Parameters
    ----------
    directory : str | pathlib.Path
        Path to the directory containing ``.npz`` episode files.
    config : object
        Configuration namespace.  Expected attributes:
        ``batch_length``, ``dataset_size``, ``image_crop_height``,
        ``image_crop_width``.
    mode : str, optional
        ``'train'`` for random cropping, ``'eval'`` for center cropping.
    """

    def __init__(self, directory: str | pathlib.Path, config, mode: str = "train"):
        super().__init__()
        self.config = config
        self.mode = mode
        self.directory = pathlib.Path(directory).expanduser()
        self.batch_length = int(config.batch_length)

        # Cropping parameters
        self.crop_h = int(getattr(config, "image_crop_height", 0))
        self.crop_w = int(getattr(config, "image_crop_width", 0))
        self.do_crop = self.crop_h > 0 and self.crop_w > 0
        # Assume 84×84 source images (consistent with the rest of the repo).
        self.orig_h = 84
        self.orig_w = 84

        # Load episodes
        if not self.directory.exists():
            print(f"Warning: Dataset directory {self.directory} does not exist.")
            self.episodes: dict = {}
            self.episode_list: list = []
        else:
            self.episodes = tools.load_episodes(
                self.directory, limit=getattr(config, "dataset_size", None)
            )
            self.episode_list = list(self.episodes.values())

        self.num_episodes = len(self.episode_list)

        # Build a flat index of valid (episode_idx, start_step) pairs.
        # With wrap-around windows every start 0..ep_len-1 is valid: when
        # start + batch_length exceeds ep_len the window wraps to the
        # beginning of the same episode.
        self.indices: list[tuple[int, int]] = []
        self.sample_weights: list[float] = []
        total_steps = 0
        for ep_idx, ep in enumerate(self.episode_list):
            ep_len = len(ep["action"])
            total_steps += ep_len
            if ep_len == 0:
                continue
            episode_weights = _episode_start_weights(ep_len, self.batch_length)
            for t in range(ep_len):
                self.indices.append((ep_idx, t))
                self.sample_weights.append(float(episode_weights[t]))

        print(
            f"[RlDataset {mode}] Loaded {self.num_episodes} episodes, "
            f"{total_steps} total steps, {len(self.indices)} valid windows "
            f"(batch_length={self.batch_length})."
        )

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    # Cropping helpers
    # ------------------------------------------------------------------
    def _get_crop_coords(self) -> tuple[int, int]:
        """Return (top, left) crop coordinates.

        In ``train`` mode a random crop is sampled; in ``eval`` mode the
        center crop is used.
        """
        if not self.do_crop:
            return 0, 0
        if self.mode == "train":
            top = np.random.randint(0, self.orig_h - self.crop_h + 1)
            left = np.random.randint(0, self.orig_w - self.crop_w + 1)
        else:
            top = (self.orig_h - self.crop_h) // 2
            left = (self.orig_w - self.crop_w) // 2
        return top, left

    def _crop(self, img: np.ndarray, top: int, left: int) -> np.ndarray:
        """Crop ``(T, H, W, C)`` image tensor to ``(T, crop_h, crop_w, C)``."""
        if not self.do_crop:
            return img
        return img[:, top : top + self.crop_h, left : left + self.crop_w, :]

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, start = self.indices[idx]
        episode = self.episode_list[ep_idx]
        ep_len = len(episode["action"])

        # Wrap-around modular indices: when start + batch_length > ep_len
        # the window wraps back to the beginning of the same episode.
        indices = np.arange(start, start + self.batch_length) % ep_len

        # --- Image (with per-sample crop, consistent across timesteps) ---
        raw_imgs = episode["image"][indices]
        top, left = self._get_crop_coords()
        cropped_imgs = self._crop(raw_imgs, top, left)

        # --- Flags ---
        # Compute is_first from scratch: position 0 is always a sequence
        # boundary, and any wrap-around point (where index decreases) is
        # also a boundary so the RSSM resets its recurrent state.
        is_first = np.zeros(self.batch_length, dtype=np.float32)
        is_first[0] = 1.0
        wraps = indices[1:] <= indices[:-1]  # True wherever wrap occurs
        is_first[1:][wraps] = 1.0

        is_terminal = episode["is_terminal"][indices]

        # --- Action & reward ---
        action = episode["action"][indices]
        reward = episode["reward"][indices]

        # --- Discount (if present) ---
        discount = episode.get("discount")
        if discount is not None:
            discount = discount[indices]

        # --- Assemble output dict ---
        out: dict[str, torch.Tensor] = {
            "image": torch.from_numpy(np.asarray(cropped_imgs)),
            "action": torch.from_numpy(np.asarray(action, dtype=np.float32)),
            "reward": torch.from_numpy(np.asarray(reward, dtype=np.float32)),
            "is_first": torch.from_numpy(is_first),
            "is_terminal": torch.from_numpy(np.asarray(is_terminal, dtype=np.float32)),
        }
        if discount is not None:
            out["discount"] = torch.from_numpy(np.asarray(discount, dtype=np.float32))

        # --- Remaining observation keys (proprio, aux, etc.) ---
        skip_keys = {"image", "action", "reward", "discount", "is_first", "is_terminal"}
        for k, v in episode.items():
            if k in skip_keys or k.startswith("log_"):
                continue
            out[k] = torch.from_numpy(np.asarray(v[indices], dtype=np.float32))

        return out
