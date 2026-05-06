"""LUMOS-style dataset of fixed-length expert windows over precomputed posteriors.

Unlike ``RlDataset`` (which yields raw image windows and forces the RL training
loop to encode + observe every batch), ``LumosDataset`` runs the frozen world
model over **each full expert episode end-to-end** at construction time and
caches the resulting posterior trajectory ``post[ep, 0..ep_len-1]``. Every
cached posterior at timestep ``t`` has therefore been conditioned on the
entire expert sequence ``0..t`` -- the LUMOS / DITTO warm-up semantics, but
amortized to a one-time pre-encode (safe because the WM is frozen for the
entire RL fine-tune).

Each ``__getitem__`` returns a horizon-length window of cached tensors:

    post_<key>  : (H, ...) for each key returned by wm.dynamics.observe()
                  (typically stoch + deter + logit for discrete RSSM, or
                  stoch + deter + mean + std for continuous RSSM).
    action      : (H, A) raw expert actions for the same window.
    is_first    : (H,) -- 1.0 at index 0 (window start) and at every wrap
                  point; 0.0 elsewhere.

Wrap-around: every timestep in every episode is a valid window start. When
``start + H`` overruns the episode, the window indices wrap back to step 0 of
the same episode (``np.arange(start, start + H) % ep_len``) and ``is_first``
fires at the wrap so the boundary-aware ``_imagine_policy`` resets imagination
to the cached posterior at the wrapped index. This mirrors ``RlDataset``'s
sampling distribution -- late-episode timesteps are visited as often as any
other -- while preserving the LUMOS warm-up for the unwrapped portion of each
window.

The training loop forms ``start_state = {k: v[:, 0] for k, v in post.items()}``
and feeds it into ``_imagine_policy(..., is_first=is_first_horizon, post=post)``.
The unwrapped portion of every imagination rollout starts from a fully
history-conditioned posterior; at wrap points it splices to the cached
posterior at the wrapped (early-episode) index.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import tools


class LumosDataset(Dataset):
    def __init__(
        self,
        directory: str | pathlib.Path,
        config,
        world_model,
        mode: str = "train",
    ):
        super().__init__()
        self.config = config
        self.mode = mode
        self.directory = pathlib.Path(directory).expanduser()
        self.horizon = int(config.imag_horizon)
        if self.horizon <= 1:
            raise ValueError(
                f"LumosDataset requires imag_horizon > 1, got {self.horizon}."
            )

        self.cache_device = torch.device(
            str(getattr(config, "lumos_cache_device", "cpu"))
        )

        # Cropping params -- mirror RlDataset, but always center crop so cached
        # posteriors are deterministic across runs.
        self.crop_h = int(getattr(config, "image_crop_height", 0))
        self.crop_w = int(getattr(config, "image_crop_width", 0))
        self.do_crop = self.crop_h > 0 and self.crop_w > 0
        self.orig_h = 84
        self.orig_w = 84

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

        # Pre-encode every episode end-to-end with the frozen WM.
        # post_cache[ep_idx] is a dict of CPU/GPU tensors shaped (T, ...).
        # action_cache[ep_idx] is (T, A) float32.
        self.post_cache: list[dict[str, torch.Tensor]] = []
        self.action_cache: list[torch.Tensor] = []

        wm_device = next(world_model.parameters()).device
        use_amp = bool(
            int(getattr(config, "precision", 32)) == 16
            and torch.cuda.is_available()
            and "cuda" in str(wm_device)
        )
        amp_device = "cuda" if use_amp else "cpu"

        was_training = world_model.training
        world_model.eval()
        try:
            with torch.no_grad():
                for ep_idx, episode in enumerate(self.episode_list):
                    post, action = self._encode_episode(
                        episode, world_model, wm_device, use_amp, amp_device
                    )
                    self.post_cache.append(post)
                    self.action_cache.append(action)
        finally:
            if was_training:
                world_model.train()

        # Mirror RlDataset: every timestep in every episode is a valid window
        # start. Windows that overrun the episode wrap back to step 0 of the
        # same episode in __getitem__ and fire is_first at the wrap point.
        self.indices: list[tuple[int, int]] = []
        total_steps = 0
        for ep_idx, episode in enumerate(self.episode_list):
            ep_len = len(episode["action"])
            total_steps += ep_len
            for t in range(ep_len):
                self.indices.append((ep_idx, t))
        self.sample_weights = [1.0] * len(self.indices)

        cache_bytes = self._cache_bytes()
        print(
            f"[LumosDataset {mode}] Encoded {self.num_episodes} episodes, "
            f"{total_steps} total steps, {len(self.indices)} valid windows "
            f"(horizon={self.horizon}, wrap-around enabled, "
            f"cache={cache_bytes / (1024**2):.1f} MiB on {self.cache_device})."
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, start = self.indices[idx]
        H = self.horizon
        ep_len = self.action_cache[ep_idx].shape[0]

        # Wrap-around modular indices: matches RlDataset:149. The gather index
        # has to live on the same device as the cached tensors so index_select
        # works when lumos_cache_device='cuda'.
        indices_np = np.arange(start, start + H) % ep_len
        gather = torch.from_numpy(indices_np).to(
            device=self.cache_device, dtype=torch.long
        )

        out: dict[str, torch.Tensor] = {}
        for key, tensor in self.post_cache[ep_idx].items():
            out[f"post_{key}"] = tensor.index_select(0, gather).clone()
        out["action"] = self.action_cache[ep_idx].index_select(0, gather).clone()

        # is_first: position 0 is always a sequence boundary (matches
        # RlDataset:160-163), and any wrap-around point (where index decreases)
        # is also a boundary so the boundary-aware imagination resets there.
        is_first = torch.zeros(H, dtype=torch.float32, device=self.cache_device)
        is_first[0] = 1.0
        wraps = indices_np[1:] <= indices_np[:-1]
        if wraps.any():
            is_first[1:][torch.from_numpy(wraps).to(self.cache_device)] = 1.0
        out["is_first"] = is_first
        return out

    def _encode_episode(self, episode, world_model, wm_device, use_amp, amp_device):
        ep_len = len(episode["action"])

        obs: dict[str, torch.Tensor] = {}

        raw_image = episode["image"]
        top, left = self._center_crop_coords()
        cropped = self._apply_crop(raw_image, top, left)
        obs["image"] = torch.from_numpy(np.asarray(cropped)).unsqueeze(0)

        action_np = np.asarray(episode["action"], dtype=np.float32)
        obs["action"] = torch.from_numpy(action_np).unsqueeze(0)

        is_first = np.zeros(ep_len, dtype=np.float32)
        is_first[0] = 1.0
        obs["is_first"] = torch.from_numpy(is_first).unsqueeze(0)

        if "is_terminal" in episode:
            obs["is_terminal"] = torch.from_numpy(
                np.asarray(episode["is_terminal"], dtype=np.float32)
            ).unsqueeze(0)
        if "reward" in episode:
            obs["reward"] = torch.from_numpy(
                np.asarray(episode["reward"], dtype=np.float32)
            ).unsqueeze(0)
        if "discount" in episode:
            obs["discount"] = torch.from_numpy(
                np.asarray(episode["discount"], dtype=np.float32)
            ).unsqueeze(0)

        skip = {"image", "action", "reward", "discount", "is_first", "is_terminal"}
        for k, v in episode.items():
            if k in skip or k.startswith("log_"):
                continue
            obs[k] = torch.from_numpy(np.asarray(v, dtype=np.float32)).unsqueeze(0)

        with torch.amp.autocast(
            device_type=amp_device, enabled=use_amp, dtype=torch.float16
        ):
            data_wm = world_model.preprocess(obs)
            embed = world_model.encoder(data_wm)
            post, _ = world_model.dynamics.observe(
                embed, data_wm["action"], data_wm["is_first"]
            )

        post_cpu: dict[str, torch.Tensor] = {}
        for key, tensor in post.items():
            t = tensor.squeeze(0).detach().to(
                device=self.cache_device, dtype=torch.float32
            )
            post_cpu[key] = t.contiguous()

        action_cached = torch.from_numpy(action_np).to(
            device=self.cache_device, dtype=torch.float32
        ).contiguous()

        return post_cpu, action_cached

    def _center_crop_coords(self) -> tuple[int, int]:
        if not self.do_crop:
            return 0, 0
        top = (self.orig_h - self.crop_h) // 2
        left = (self.orig_w - self.crop_w) // 2
        return top, left

    def _apply_crop(self, img: np.ndarray, top: int, left: int) -> np.ndarray:
        if not self.do_crop:
            return img
        return img[:, top : top + self.crop_h, left : left + self.crop_w, :]

    def _cache_bytes(self) -> int:
        total = 0
        for post in self.post_cache:
            for tensor in post.values():
                total += tensor.element_size() * tensor.numel()
        for action in self.action_cache:
            total += action.element_size() * action.numel()
        return total
