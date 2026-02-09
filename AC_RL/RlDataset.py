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
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

# Ensure the repo root is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import tools

_EXCLUDE_KEYS = {"action", "reward", "discount", "is_first", "is_terminal"}


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
        # Each pair identifies a contiguous window of ``batch_length`` steps
        # that fits entirely within a single episode.
        self.indices: list[tuple[int, int]] = []
        total_steps = 0
        for ep_idx, ep in enumerate(self.episode_list):
            ep_len = len(ep["action"])
            total_steps += ep_len
            if ep_len < self.batch_length:
                continue
            # The last valid start is (ep_len - batch_length).
            max_start = ep_len - self.batch_length
            for t in range(max_start + 1):
                self.indices.append((ep_idx, t))

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
        sl = slice(start, start + self.batch_length)

        # --- Image (with per-sample crop, consistent across timesteps) ---
        raw_imgs = episode["image"][sl]
        top, left = self._get_crop_coords()
        cropped_imgs = self._crop(raw_imgs, top, left)

        # --- Flags ---
        is_first = episode["is_first"][sl].copy()
        is_first[0] = True  # mark sequence boundary

        is_terminal = episode["is_terminal"][sl]

        # --- Action & reward ---
        action = episode["action"][sl]
        reward = episode["reward"][sl]

        # --- Discount (if present) ---
        discount = episode.get("discount")
        if discount is not None:
            discount = discount[sl]

        # --- Assemble output dict ---
        out: dict[str, torch.Tensor] = {
            "image": torch.from_numpy(np.asarray(cropped_imgs)),
            "action": torch.from_numpy(np.asarray(action, dtype=np.float32)),
            "reward": torch.from_numpy(np.asarray(reward, dtype=np.float32)),
            "is_first": torch.from_numpy(np.asarray(is_first, dtype=np.float32)),
            "is_terminal": torch.from_numpy(np.asarray(is_terminal, dtype=np.float32)),
        }
        if discount is not None:
            out["discount"] = torch.from_numpy(np.asarray(discount, dtype=np.float32))

        # --- Remaining observation keys (proprio, aux, etc.) ---
        skip_keys = {"image", "action", "reward", "discount", "is_first", "is_terminal"}
        for k, v in episode.items():
            if k in skip_keys or k.startswith("log_"):
                continue
            out[k] = torch.from_numpy(np.asarray(v[sl], dtype=np.float32))

        return out
