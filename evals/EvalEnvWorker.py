
import pathlib
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch



sys.path.append(str(pathlib.Path(__file__).parent.parent))



from bc_mlp.BC_MLP_eval import (
    _make_robomimic_env, 
    _extract_success
)

class EvalEnvWorker:
    def __init__(self, config, env_cfg, img_size, random_crop = False):
        self._cfg = env_cfg
        self._image_size = img_size
        self._env = _make_robomimic_env(self._cfg, self._image_size)
        self.config = config
        self.random_crop = random_crop

    def reset(self):
        return self._env.reset()

    def step(self, action,):
        obs, reward, done, info = self._env.step(action)
        obs = self._prepare_obs(obs, is_first=False, is_terminal=done)
        return obs, reward, done, info, _extract_success(info, self._env)

    def _stack_flip_cameras(self, obs: Dict[str, Any], camera_keys: Iterable[str], flip_keys: Iterable[str] = ()):
        frames: List[Any] = []
        flip_set = set(flip_keys)
        for key in camera_keys:
            if key not in obs:
                raise KeyError(f"Camera observation '{key}' missing from environment output")
            frame = np.asarray(obs[key])
            if frame.dtype != np.float32:
                frame = frame.astype(np.float32)
            if key in flip_set:
                frame = np.flip(frame, axis=0)
            frames.append(frame)
        return np.concatenate(frames, axis=-1)

    def _prepare_obs(
        self,
        raw_obs: Dict[str, Any],
        *,
        is_first: Optional[bool] = None,
        is_terminal: Optional[bool] = None,
    ) -> Dict[str, Any]:
        processed: Dict[str, Any] = OrderedDict()
        
        cnn_keys_order = self.config.bc_cnn_keys_order
        mlp_keys_order = self.config.bc_mlp_keys_order
        camera_keys = self.config.camera_obs_keys
        flip_keys = self.config.flip_camera_keys
        crop_height = self.config.image_crop_height
        crop_width = self.config.image_crop_width

        for key in cnn_keys_order:
            if key == "image":
                img = self._stack_flip_cameras(raw_obs, camera_keys, flip_keys)
                if crop_height and crop_width and img.shape[0] >= crop_height and img.shape[1] >= crop_width:
                    h, w = img.shape[:2]
                    if self.random_crop:
                        top = np.random.randint(0, h - crop_height + 1)
                        left = np.random.randint(0, w - crop_width + 1)
                    else:
                        top = (h - crop_height) // 2
                        left = (w - crop_width) // 2
                    img = img[top : top + crop_height, left : left + crop_width, :]
                processed["image"] = img
            elif key in raw_obs:
                processed[key] = np.asarray(raw_obs[key], dtype=np.float32)
        
        processed_to_raw = {
            "aux_robot0_joint_pos_sin": "robot0_joint_pos_sin",
            "aux_robot0_joint_pos_cos": "robot0_joint_pos_cos",
        }
        
        for key in mlp_keys_order:
            if key in processed_to_raw:
                raw_key = processed_to_raw[key]
                if raw_key in raw_obs:
                    processed[key] = np.asarray(raw_obs[raw_key], dtype=np.float32)
            elif key in raw_obs:
                processed[key] = np.asarray(raw_obs[key], dtype=np.float32)

        if is_first is not None:
            processed["is_first"] = np.array([1.0 if is_first else 0.0], dtype=np.float32)
        if is_terminal is not None:
            processed["is_terminal"] = np.array([1.0 if is_terminal else 0.0], dtype=np.float32)

        return processed
    
    def set_state(self,state):
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        self._env.reset()
        self._env.sim.set_state_from_flattened(state)
        self._env.sim.forward()
        raw_obs = self._env._get_observations(force_update=True)
        return self._prepare_obs(raw_obs, is_first=True, is_terminal=False)

    def close(self):
        if self._env: self._env.close()
