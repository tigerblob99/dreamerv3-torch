import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
import os
import re
import pathlib
import argparse
import yaml
import sys

# Ensure parent directory is in path to import 'tools'
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import tools  #

# Define REPO_ROOT for config loading
REPO_ROOT = pathlib.Path(__file__).resolve().parent

class EvalDataset(Dataset):
    def __init__(self, hdf5_path, config, eval_dir, warmup_len=5, horizon=10):
        """
        Args:
            hdf5_path (str): Path to the Robomimic HDF5 dataset.
            config: Configuration object.
            eval_dir (str): Directory containing .npz files defining which demos to use.
        """
        self.config = config
        self.warmup_len = warmup_len
        self.horizon = horizon
        self.filepath = hdf5_path
        
        # FIX: Call internal method with self
        self.eval_episodes = self.get_eval_demos(eval_dir)
        
        self._file = h5py.File(self.filepath, "r")
        
        self.indices = []
        
        for episode in self.eval_episodes:
            # Safety check: ensure demo exists in HDF5
            if episode not in self._file["data"]:
                print(f"Warning: {episode} found in eval_dir but missing in HDF5")
                continue

            demo_len = self._file["data"][episode]["actions"].shape[0]
            
            min_t0 = warmup_len
            max_t0 = demo_len - horizon
            
            if max_t0 > min_t0:
                for t in range(min_t0, max_t0):
                    self.indices.append((episode, t))
                    
        print(f"Dataset loaded. Found {len(self.indices)} valid evaluation points.")
        print("First few indices (demo_key, t0):")
        for idx_pair in self.indices[:5]:
            print(f"  - {idx_pair}")

    def __len__(self):
        return len(self.indices)

    # FIX: Added 'self' as first argument
    def get_eval_demos(self, eval_dir: str) -> list[str]:
        eval_path = pathlib.Path(eval_dir).expanduser()
        if not eval_path.exists():
            print(f"Warning: Evaluation directory {eval_path} does not exist.")
            return []

        present_demos = set()
        pattern = re.compile(r"^(demo_\d+)-\d+\.npz$")

        for filename in os.listdir(eval_path):
            match = pattern.match(filename)
            if match:
                present_demos.add(match.group(1))

        return sorted(list(present_demos), key=lambda x: int(x.split('_')[1]))

    def _get_obs_at_indices(self, demo_grp, t_start, t_end):
        imgs = []
        for key in self.config.camera_obs_keys:
            h5_key = key
            if key not in demo_grp["obs"]:
                h5_key = key.replace("_image", "")
            
            raw = demo_grp["obs"][h5_key][t_start:t_end]
            imgs.append(raw)
            
        cat_img = np.concatenate(imgs, axis=-1)
        
        h, w = cat_img.shape[1:3]
        th, tw = self.config.image_crop_height, self.config.image_crop_width
        if th > 0 and tw > 0:
            top = (h - th) // 2
            left = (w - tw) // 2
            cat_img = cat_img[:, top:top+th, left:left+tw, :]
            
        tensor_img = torch.tensor(cat_img, dtype=torch.float32) / 255.0
        return tensor_img

    def __getitem__(self, idx):
        demo_key, t0 = self.indices[idx]
        demo_grp = self._file["data"][demo_key]
        
        # 1. Warmup
        w_start = t0 - self.warmup_len
        w_end = t0 + 1
        warmup_obs = self._get_obs_at_indices(demo_grp, w_start, w_end)
        
        raw_warmup_actions = torch.tensor(demo_grp["actions"][w_start:w_end], dtype=torch.float32)
        warmup_actions = torch.zeros_like(raw_warmup_actions)
        warmup_actions[1:] = raw_warmup_actions[:-1]
        
        is_first = torch.zeros(len(warmup_obs))
        is_first[0] = 1.0

        # 2. Target
        h_start = t0 + 1
        h_end = t0 + 1 + self.horizon
        target_obs = self._get_obs_at_indices(demo_grp, h_start, h_end)
        target_actions = torch.tensor(demo_grp["actions"][t0 : t0 + self.horizon], dtype=torch.float32)

        # 3. State
        state_flat = torch.tensor(demo_grp["states"][t0], dtype=torch.float32)

        return {
            "demo_key": demo_key,
            "t0": t0,
            "warmup_image": warmup_obs,
            "warmup_action": warmup_actions,
            "is_first": is_first,
            "target_image": target_obs,
            "target_action": target_actions,
            "env_state": state_flat
        }

    def close(self):
        self._file.close()

def _parse_config():
    # ... (Your existing config parsing logic, kept for brevity) ...
    # Simplified for testing:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--configs", nargs="+")
    pre_args, remaining = pre_parser.parse_known_args()
    
    cfg_path = REPO_ROOT / "configs.yaml"
    configs = yaml.safe_load(cfg_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *(pre_args.configs or [])]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
        
    # Inject defaults needed for Dataset
    defaults.setdefault("camera_obs_keys", ["agentview_image", "robot0_eye_in_hand_image"])
    defaults.setdefault("image_crop_height", 84)
    defaults.setdefault("image_crop_width", 84)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    # Add dummy checkpoint arg if your parser requires it, or remove requirement for test
    parser.add_argument("--checkpoint", type=str, default="dummy.pt") 

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    config = parser.parse_args(remaining)
    return config

def main():
    config = _parse_config()
    
    # Paths based on your previous messages
    hdf5_path = "datasets/imagecanPH_v15.hdf5"
    eval_dir = "datasets/robomimic_data_MV/can_PH_eval" # Verify this path exists
    
    print(f"Testing Dataset Load...")
    print(f"HDF5: {hdf5_path}")
    print(f"Eval Dir: {eval_dir}")

    try:
        ds = EvalDataset(hdf5_path, config, eval_dir, warmup_len=5, horizon=10)
        
        # Test __getitem__
        if len(ds) > 0:
            print("\nFetching index 0...")
            item = ds[0]
            print("Keys in item:", item.keys())
            print("Warmup Image Shape:", item['warmup_image'].shape)
            print("Env State Shape:", item['env_state'].shape)
            print("Success!")
        else:
            print("Dataset empty. Check paths.")
            
        ds.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()