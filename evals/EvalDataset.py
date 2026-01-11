import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
import os
import pathlib
import sys

# Ensure tools can be imported
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import tools

class EvalDataset(Dataset):
    def __init__(self, hdf5_path, eval_dir, config, warmup_len=5, horizon=10):
        """
        Args:
            hdf5_path (str): Path to the raw HDF5 (used ONLY for 'env_state' resets).
            eval_dir (str): Directory containing .npz files.
            config: Config object containing image_crop_height/width.
            warmup_len (int): Steps of history to feed RSSM.
            horizon (int): Steps of future to evaluate.
        """
        self.config = config
        self.warmup_len = warmup_len
        self.horizon = horizon
        self.hdf5_path = hdf5_path
        self.eval_dir = pathlib.Path(eval_dir).expanduser()
        
        # 1. Load Episodes
        raw_episodes = tools.load_episodes(self.eval_dir, limit=None)
        
        # 2. Re-key to "demo_X"
        self.episodes = {}
        for key, data in raw_episodes.items():
            demo_id = key.rsplit('-', 1)[0]
            self.episodes[demo_id] = data
        
        # 3. HDF5 handle (opened lazily per worker)
        self._file = None
        
        # 4. Build Valid Indices
        self.indices = []
        sorted_keys = sorted(self.episodes.keys(), key=lambda x: int(x.split('_')[1]))
        with h5py.File(self.hdf5_path, "r") as h5_file:
            for demo_key in sorted_keys:
                if demo_key not in h5_file["data"]:
                    print(f"Warning: {demo_key} found in NPZ but missing in HDF5. Skipping.")
                    continue

                episode = self.episodes[demo_key]
                ep_len = len(episode['action'])
                
                min_t0 = warmup_len
                max_t0 = ep_len - horizon - 1 
                
                if max_t0 > min_t0:
                    for t in range(min_t0, max_t0 + 1):
                        self.indices.append((demo_key, t))
                    
        print(f"EvalDataset loaded. Found {len(self.episodes)} episodes and {len(self.indices)} valid samples.")

    def _center_crop(self, image):
        """Helper to apply center cropping if config specifies it."""
        # image shape: (T, H, W, C)
        if self.config.image_crop_height <= 0 or self.config.image_crop_width <= 0:
            return image
            
        h, w = image.shape[1], image.shape[2]
        crop_h = self.config.image_crop_height
        crop_w = self.config.image_crop_width
        
        if h == crop_h and w == crop_w:
            return image
            
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        
        # Slice: all time steps, cropped height, cropped width, all channels
        return image[:, top:top+crop_h, left:left+crop_w, :]

    def __len__(self):
        return len(self.indices)

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __getitem__(self, idx):
        demo_key, t0 = self.indices[idx]
        episode = self.episodes[demo_key]
        
        # 1. Warmup Context
        w_start = t0 - self.warmup_len
        w_end = t0 + 1
        
        warmup_batch = {}
        for k, v in episode.items():
            data_slice = v[w_start:w_end]
            # Apply crop if this is an image
            if "image" in k:
                data_slice = self._center_crop(data_slice)
            warmup_batch[f"warmup_{k}"] = data_slice

        warmup_batch['warmup_is_first'] = np.array(warmup_batch['warmup_is_first'], copy=True)
        warmup_batch['warmup_is_first'][0] = 1.0

        # 2. Target Data
        h_start = t0 + 1
        h_end = t0 + 1 + self.horizon
        
        # Apply crop to target image as well
        target_img = episode['image'][h_start:h_end]
        target_img = self._center_crop(target_img)
        
        target_batch = {}
        target_batch["target_image"] = target_img
        target_batch["target_action"] = episode['action'][h_start:h_end]

        # 3. Env State
        h5_file = self._get_file()
        env_state = h5_file["data"][demo_key]["states"][t0]

        # 4. Construct Final Dict
        item = {}
        
        for k, v in warmup_batch.items():
            if v.dtype == np.uint8:
                item[k] = torch.tensor(v) 
            else:
                item[k] = torch.tensor(v, dtype=torch.float32)

        for k, v in target_batch.items():
            if v.dtype == np.uint8:
                item[k] = torch.tensor(v)
            else:
                item[k] = torch.tensor(v, dtype=torch.float32)

        item["env_state"] = torch.tensor(env_state, dtype=torch.float32)
        item["demo_key"] = demo_key
        item["t0"] = t0

        return item

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
