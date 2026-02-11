from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import gym
import numpy as np


class RobosuiteEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        task_name: str = "Lift",
        robots: Sequence[str] | str = ("Panda",),
        controller: str = "OSC_POSE",
        controller_configs: Optional[dict] = None,
        camera_obs_keys: Sequence[str] = ("agentview_image",),
        flip_camera_keys: Sequence[str] = (),
        image_size: Sequence[int] = (84, 84),
        mlp_keys_order: Sequence[str] = (
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
            "aux_robot0_joint_pos_sin",
            "aux_robot0_joint_pos_cos",
        ),
        aux_key_map: Optional[dict[str, str]] = None,
        reward_shaping: bool = False,
        control_freq: int = 20,
        horizon: int = 500,
        ignore_done: bool = False,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        use_camera_obs: bool = True,
        camera_depths: bool = False,
        render_gpu_device: int = -1,
        reward_shift: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()

        robot_list = [robots] if isinstance(robots, str) else list(robots)
        if not robot_list:
            raise ValueError("robosuite_robots must contain at least one robot name.")
        if not camera_obs_keys:
            raise ValueError("camera_obs_keys must include at least one camera key.")

        self._camera_obs_keys = tuple(camera_obs_keys)
        self._flip_camera_keys = set(flip_camera_keys)
        self._mlp_keys_order = tuple(mlp_keys_order)
        self._aux_key_map = aux_key_map or {
            "aux_robot0_joint_pos_sin": "robot0_joint_pos_sin",
            "aux_robot0_joint_pos_cos": "robot0_joint_pos_cos",
        }

        img_h, img_w = int(image_size[0]), int(image_size[1])
        camera_names = [self._camera_name_from_key(key) for key in self._camera_obs_keys]
        self._render_camera = camera_names[0]
        self._img_size = (img_h, img_w)
        self._reward_shift = float(reward_shift)

        # Import lazily to avoid an import cycle through offline_train -> dreamer.
        from bc_mlp.BC_MLP_eval import _make_robomimic_env

        env_cfg = SimpleNamespace(
            robosuite_task=task_name,
            robosuite_robots=tuple(robot_list),
            robosuite_controller=controller,
            controller_configs=controller_configs,
            camera_obs_keys=self._camera_obs_keys,
            use_camera_obs=use_camera_obs,
            camera_depths=camera_depths,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render=has_renderer,
            robosuite_reward_shaping=reward_shaping,
            robosuite_control_freq=control_freq,
            max_env_steps=int(horizon),
            ignore_done=ignore_done,
            seed=int(seed),
        )
        self._env = _make_robomimic_env(env_cfg, (img_h, img_w))

        act_dim = int(self._env.action_dim)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        sample_obs = self._env._get_observations(force_update=True)
        processed = self._process_obs(sample_obs, is_first=True, is_terminal=False)
        self.observation_space = gym.spaces.Dict(
            {
                key: gym.spaces.Box(
                    low=0 if value.dtype == np.uint8 else -np.inf,
                    high=255 if value.dtype == np.uint8 else np.inf,
                    shape=value.shape,
                    dtype=value.dtype,
                )
                for key, value in processed.items()
            }
        )

    @staticmethod
    def _camera_name_from_key(obs_key: str) -> str:
        return obs_key[:-6] if obs_key.endswith("_image") else obs_key

    @staticmethod
    def _to_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        if np.issubdtype(frame.dtype, np.floating):
            scale = 255.0 if frame.max() <= 1.0 else 1.0
            return np.clip(frame * scale, 0, 255).astype(np.uint8)
        return frame.astype(np.uint8)

    def _stack_cameras(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        frames = []
        for key in self._camera_obs_keys:
            if key not in obs:
                raise KeyError(
                    f"Camera observation '{key}' missing from robosuite output. "
                    f"Available keys: {list(obs.keys())}"
                )
            frame = np.asarray(obs[key])
            if key in self._flip_camera_keys:
                frame = np.flip(frame, axis=0)
            frames.append(self._to_uint8(frame))
        return np.concatenate(frames, axis=-1)

    def _extract_mlp_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        joint_pos = np.asarray(obs.get("robot0_joint_pos", []), dtype=np.float32)
        for key in self._mlp_keys_order:
            if key == "image":
                continue
            if key in result:
                continue
            if key in self._aux_key_map:
                raw_key = self._aux_key_map[key]
                if raw_key in obs:
                    result[key] = np.asarray(obs[raw_key], dtype=np.float32)
                    continue
                if raw_key == "robot0_joint_pos_sin" and joint_pos.size:
                    result[key] = np.sin(joint_pos).astype(np.float32)
                    continue
                if raw_key == "robot0_joint_pos_cos" and joint_pos.size:
                    result[key] = np.cos(joint_pos).astype(np.float32)
                    continue
                raise KeyError(
                    f"Observation key '{raw_key}' required by mlp key '{key}' is missing."
                )
            if key not in obs:
                raise KeyError(
                    f"Observation key '{key}' required by bc_mlp_keys_order is missing."
                )
            result[key] = np.asarray(obs[key], dtype=np.float32)
        return result

    def _process_obs(
        self, obs: Dict[str, np.ndarray], *, is_first: bool, is_terminal: bool
    ) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {"image": self._stack_cameras(obs)}
        result.update(self._extract_mlp_obs(obs))
        result["is_first"] = np.array(1.0 if is_first else 0.0, dtype=np.float32)
        result["is_terminal"] = np.array(1.0 if is_terminal else 0.0, dtype=np.float32)
        return result

    def reset(self):  # type: ignore[override]
        obs = self._env.reset()
        return self._process_obs(obs, is_first=True, is_terminal=False)

    def step(self, action):  # type: ignore[override]
        action = np.asarray(action, dtype=np.float32)
        obs, reward, done, info = self._env.step(action)
        processed = self._process_obs(obs, is_first=False, is_terminal=bool(done))
        info = info or {}
        info.setdefault("discount", np.array(0.0 if done else 1.0, dtype=np.float32))
        reward = float(np.float32(reward) + self._reward_shift)
        return processed, reward, bool(done), info

    def render(self, mode="rgb_array", width=None, height=None):  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only 'rgb_array' render mode is supported.")
        width = width or self._img_size[1]
        height = height or self._img_size[0]
        return self._env.render(
            camera_name=self._render_camera, width=width, height=height
        )

    def close(self):  # type: ignore[override]
        self._env.close()


RobosuiteLiftEnv = RobosuiteEnv


def make_lift_env(config, seed: int):
    size = tuple(getattr(config, "size", (84, 84)))
    env = RobosuiteEnv(
        task_name=getattr(config, "robosuite_task", "Lift"),
        robots=getattr(config, "robosuite_robots", ("Panda",)),
        controller=getattr(config, "robosuite_controller", "OSC_POSE"),
        controller_configs=getattr(config, "controller_configs", None),
        camera_obs_keys=tuple(getattr(config, "camera_obs_keys", ("agentview_image",))),
        flip_camera_keys=tuple(getattr(config, "flip_camera_keys", ("agentview_image",))),
        image_size=size,
        mlp_keys_order=tuple(
            getattr(
                config,
                "bc_mlp_keys_order",
                (
                    "robot0_joint_pos",
                    "robot0_joint_vel",
                    "robot0_gripper_qpos",
                    "robot0_gripper_qvel",
                    "aux_robot0_joint_pos_sin",
                    "aux_robot0_joint_pos_cos",
                ),
            )
        ),
        aux_key_map=getattr(config, "aux_key_map", None),
        reward_shaping=getattr(config, "robosuite_reward_shaping", False),
        control_freq=getattr(config, "robosuite_control_freq", 20),
        horizon=int(getattr(config, "max_env_steps", 500)),
        ignore_done=getattr(config, "ignore_done", False),
        has_renderer=getattr(config, "has_renderer", False),
        has_offscreen_renderer=getattr(config, "has_offscreen_renderer", True),
        use_camera_obs=getattr(config, "use_camera_obs", True),
        camera_depths=getattr(config, "camera_depths", False),
        render_gpu_device=getattr(config, "robosuite_render_device", -1),
        reward_shift=float(getattr(config, "robosuite_reward_shift", 0.0)),
        seed=seed,
    )
    return env
