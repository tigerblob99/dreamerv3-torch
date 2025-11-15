import json
import math
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import gym
import numpy as np


def _resolve_controller_loader(suite_module: object) -> Callable[..., dict]:
    """Find robosuite.load_controller_config across API variants."""

    candidates = (
        "controllers.config",             # robosuite>=1.5.0
        "controllers.controller_config",  # robosuite 1.5.x (internal refactor)
        "controllers.controller_factory", # some downstream forks
        "controllers",                    # robosuite<=1.4
    )
    for path in candidates:
        try:  # pragma: no cover - depends on installed robosuite version
            module = __import__(f"robosuite.{path}", fromlist=["load_controller_config"])
            loader = getattr(module, "load_controller_config", None)
            if loader is not None:
                return loader
        except ImportError:
            continue

    loader = getattr(suite_module, "load_controller_config", None)
    if loader is not None:
        return loader

    module_file = getattr(suite_module, "__file__", None)
    base_dir = Path(module_file).resolve().parent if module_file else None

    def _load_from_files(*, custom_fpath: Optional[str] = None, default_controller: Optional[str] = None, **_: object) -> dict:
        """Fallback loader that mimics robosuite<=1.4 behaviour using JSON configs."""

        candidates: list[Path] = []
        if custom_fpath:
            provided = Path(custom_fpath)
            candidates.append(provided if provided.is_absolute() else provided.resolve())
            if not provided.is_absolute() and base_dir is not None:
                candidates.append((base_dir / provided).resolve())
        elif default_controller:
            if base_dir is None:
                raise ImportError(
                    "Unable to locate robosuite installation to resolve controller config"
                )
            filename = f"{default_controller.lower()}.json"
            rel_paths = (
                Path("controllers") / "config" / filename,
                Path("controllers") / "config" / "default" / filename,
                Path("controllers") / "config" / "default" / "parts" / filename,
                Path("controllers") / "config" / "default" / "composite" / filename,
            )
            candidates.extend((base_dir / rel).resolve() for rel in rel_paths)
        else:
            raise ValueError("Either custom_fpath or default_controller must be provided")

        for config_path in candidates:
            if config_path.is_file():
                with config_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

        raise FileNotFoundError(
            "Controller config not found in candidate locations: "
            + ", ".join(str(path) for path in candidates)
        )

    return _load_from_files


def _ensure_composite_controller_config(controller_cfg: Optional[dict], robots: Sequence[str] | str) -> Optional[dict]:
    """Convert legacy part controller configs into composite configs when using robosuite>=1.5."""

    if controller_cfg is None or not isinstance(controller_cfg, dict):
        return controller_cfg

    try:  # pragma: no cover - only available in newer robosuite versions
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )
    except ImportError:
        return controller_cfg

    robot_names = [robots] if isinstance(robots, str) else list(robots)
    if not robot_names:
        return controller_cfg

    primary_robot = robot_names[0]

    arms: list[str] = []
    try:  # pragma: no cover - depends on robosuite internals
        from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

        registered = REGISTERED_ROBOTS.get(primary_robot)
        if registered is None and primary_robot.lower() in REGISTERED_ROBOTS:
            registered = REGISTERED_ROBOTS[primary_robot.lower()]
        if registered is not None:
            candidate_arms = getattr(registered, "arms", None)
            if candidate_arms:
                arms = list(candidate_arms)
    except Exception:
        arms = []

    if not arms:
        arms = ["right", "left"] if len(robot_names) > 1 else ["right"]

    try:
        return refactor_composite_controller_config(controller_cfg, primary_robot, arms)
    except Exception:
        return controller_cfg


class RobosuiteLiftEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        task_name: str = "Lift",
        robots: Sequence[str] | str = "Panda",
        controller: str = "OSC_POSE",
        camera: str = "agentview",
        image_size: Sequence[int] = (84, 84),
        lowdim_keys: Sequence[str] | None = None,
        add_joint_trig: bool = True,
        reward_shaping: bool = True,
        control_freq: int = 20,
        horizon: int | None = None,
        render_gpu_device: int = -1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        try:
            import robosuite as suite
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "robosuite is required for RobosuiteLiftEnv. Install via `pip install robosuite`"
            ) from exc

        load_controller_config = _resolve_controller_loader(suite)

        if isinstance(robots, str):
            robots = [robots]
        if lowdim_keys is None:
            lowdim_keys = (
                "robot0_joint_pos",
                "robot0_joint_vel",
                "robot0_gripper_qpos",
                "robot0_gripper_qvel",
            )

        controller_cfg = load_controller_config(default_controller=controller)
        controller_cfg = _ensure_composite_controller_config(controller_cfg, robots)
        img_h, img_w = image_size
        # Default horizon roughly matches robosuite Lift (100 steps). Fall back to product with control freq.
        self._horizon = horizon or int(math.ceil(control_freq * 5.0))

        self._env = suite.make(
            env_name=task_name,
            robots=list(robots),
            controller_configs=controller_cfg,
            use_object_obs=True,
            use_camera_obs=True,
            camera_names=[camera],
            camera_heights=img_h,
            camera_widths=img_w,
            has_renderer=False,
            has_offscreen_renderer=True,
            reward_shaping=reward_shaping,
            control_freq=control_freq,
            horizon=self._horizon,
            render_gpu_device_id=render_gpu_device,
        )

        
        try:  
            self._env.reset(seed=seed)
        except TypeError:  # upgrade / legacy mismatch in signature
            seed_fn = getattr(self._env, "seed", None)
            if callable(seed_fn):
                seed_fn(seed)
            self._env.reset()

        self._camera = camera
        self._img_size = (img_h, img_w)
        self._lowdim_keys = tuple(lowdim_keys)
        self._add_joint_trig = add_joint_trig
        self._last_was_reset = True

        # Action space in robosuite is already normalized to [-1, 1].
        act_dim = np.int32(self._env.action_dim)
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
        self._initial_obs = processed

    def _process_obs(self, obs: Dict[str, np.ndarray], *, is_first: bool, is_terminal: bool) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        image_key = f"{self._camera}_image"
        if image_key not in obs:
            raise KeyError(f"Camera '{self._camera}' not present in robosuite observations: {list(obs.keys())}")
        image = np.array(obs[image_key])
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        image = np.flipud(image)  # Flip vertically to align camera view
        result["image"] = image

        for key in self._lowdim_keys:
            if key not in obs:
                raise KeyError(f"Observation key '{key}' not provided by robosuite environment")
            result[key] = np.asarray(obs[key], dtype=np.float32)

        if self._add_joint_trig and "robot0_joint_pos" in obs:
            joint_pos = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
            result.setdefault("aux_robot0_joint_pos_sin", np.sin(joint_pos).astype(np.float32))
            result.setdefault("aux_robot0_joint_pos_cos", np.cos(joint_pos).astype(np.float32))

        result["is_first"] = np.array([1 if is_first else 0], dtype=np.uint8)
        result["is_terminal"] = np.array([1 if is_terminal else 0], dtype=np.uint8)
        return result

    def reset(self):  # type: ignore[override]
        obs = self._env.reset()
        self._last_was_reset = True
        processed = self._process_obs(obs, is_first=True, is_terminal=False)
        return processed

    def step(self, action):  # type: ignore[override]
        action = np.asarray(action, dtype=np.float32)
        obs, reward, done, info = self._env.step(action)
        processed = self._process_obs(obs, is_first=False, is_terminal=done)
        info = info or {}
        info.setdefault("discount", np.array(0.0 if done else 1.0, dtype=np.float32))
        return processed, float(reward), bool(done), info

    def render(self, mode="rgb_array", width=None, height=None):  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only 'rgb_array' render mode is supported.")
        width = width or self._img_size[1]
        height = height or self._img_size[0]
        return self._env.render(camera_name=self._camera, width=width, height=height)

    def close(self):  # type: ignore[override]
        self._env.close()


def make_lift_env(config, seed: int):
    size = config.size if hasattr(config, "size") else (84, 84)
    horizon = getattr(config, "robosuite_horizon", None)
    env = RobosuiteLiftEnv(
        task_name=getattr(config, "robosuite_task_name", "Lift"),
        robots=getattr(config, "robosuite_robots", "Panda"),
        controller=getattr(config, "robosuite_controller", "OSC_POSE"),
        camera=getattr(config, "robosuite_camera", "agentview"),
        image_size=tuple(size),
        lowdim_keys=getattr(
            config,
            "robosuite_lowdim_keys",
            (
                "robot0_joint_pos",
                "robot0_joint_vel",
                "robot0_gripper_qpos",
                "robot0_gripper_qvel",
            ),
        ),
        add_joint_trig=getattr(config, "robosuite_add_joint_trig", True),
        reward_shaping=getattr(config, "robosuite_reward_shaping", True),
        control_freq=getattr(config, "robosuite_control_freq", 20),
        horizon=horizon,
        render_gpu_device=getattr(config, "robosuite_render_device", -1),
        seed=seed,
    )
    return env
