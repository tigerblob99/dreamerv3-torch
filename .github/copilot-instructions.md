# Copilot Instructions for dreamerv3-torch

These instructions guide AI coding agents to be productive quickly in this repo. Focus on the concrete patterns and workflows used here.

## Big Picture
- This is a PyTorch implementation of DreamerV3. Core agent and training loop live in [dreamer.py](../dreamer.py), with model components in [models.py](../models.py) and [networks.py](../networks.py).
- Two modes of use:
  - Online RL training via environment wrappers (see [envs/](../envs)).
  - Offline training/evaluation from recorded episodes (NPZ/HDF5) via [offline_train.py](../offline_train.py) and behavior cloning in [BC_MLP_train.py](../BC_MLP_train.py) / [BC_MLP_eval.py](../BC_MLP_eval.py).
- World model components: `RSSM` latent dynamics, `MultiEncoder`/`MultiDecoder` for mixed CNN+MLP observations, `ImagBehavior` for actor-critic on imagined rollouts.

## Data & Episodes
- Episodes are dicts of numpy arrays with keys like `image`, `action`, `reward`, `discount`, `is_first`, `is_terminal`. Loader: `tools.load_episodes()` (see [tools.py](../tools.py)).
- Batch generation uses `tools.sample_episodes(..., batch_length)` → `tools.from_generator(..., batch_size)` in [dreamer.py](../dreamer.py#L145).
- Offline eval supports two styles in [offline_train.py](../offline_train.py):
  - Batched/padded via `_prepare_padded_episode()` respecting `batch_length`.
  - Sequential, length-preserving via `_prepare_full_episode()`.
- Image preprocessing: configurable standardization and optional center-cropping; see `image_standardize`, `image_standardize_dataset`, and `_crop_image_sequence()` usage.

## Config & Conventions
- All runs merge named blocks from [configs.yaml](../configs.yaml) on top of `defaults`. Scripts accept `--configs <name1> <name2> ...`.
- Observation key patterns: encoder/decoder use `mlp_keys` and `cnn_keys` regexes. For BC, explicit ordering is configured via `bc_cnn_keys_order` and `bc_mlp_keys_order` to ensure consistent concatenation.
- `MUJOCO_GL` is set to `osmesa` for headless rendering (see [dreamer.py](../dreamer.py) and BC scripts). Keep it for CI/headless runs.
- Logging: TensorBoard and Weights & Biases (wandb) are integrated; `WANDB_LOGDIR` defaults to the run `logdir`.

## Workflows (Commands)
- Install deps (Python 3.11):
  - `pip install -r requirements.txt`
- Online training (DMC Vision example):
  - `python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk`
- Offline Dreamer training/eval (robomimic NPZ):
  - `python offline_train.py --configs robomimic --offline_traindir ./datasets/robomimic_data_MV/can_MH_train --offline_evaldir ./datasets/robomimic_data_MV/can_MH_eval --logdir ./logdir/robomimic_offline_can_MH_cropped --image_crop_height 78 --image_crop_width 78 --image_crop_random False --image_standardize False --offline_eval_preserve_length True --image_standardize_dataset False`
- Behavior Cloning (BC) using frozen Dreamer encoder:
  - `python BC_MLP_train.py --configs bc_defaults bc_can_PH --checkpoint logdir/robomimic_offline_can_MH/latest.pt --offline_traindir datasets/robomimic_data_MV/can_PH_train --bc_epochs 1000 --bc_batch_size 512 --bc_lr 1e-4`
  - Evaluate trained BC policy in robosuite: see `evaluate_policy()` in [BC_MLP_eval.py](../BC_MLP_eval.py).
- Docker: see [Dockerfile](../Dockerfile) for environment setup.

## Patterns You Should Follow
- Gym spaces: build `gym.spaces.Dict` for observations and `gym.spaces.Box` for actions. `_infer_spaces()` in [offline_train.py](../offline_train.py) shows expected shapes and dtypes.
- Observations are channels-last images `(T, H, W, C)` and vector features as `(T, D)`. Encoders normalize images to `[0,1]` when configured.
- Parallel environments: use `Parallel`/`Damy` in [parallel.py](../parallel.py) to abstract process vs. in-thread envs.
- Metrics and logging: use `tools.Logger` and integrate with wandb (`wandb.init(...)` in [dreamer.py](../dreamer.py)). Avoid ad-hoc print-only logging for training loops.
- Keep actor/critic/loss scales consistent with `configs.yaml`. Symlog distributions are used for certain heads; match `reward_head`, `cont_head` config.

## Integration Points
- Robosuite/robomimic: environment setup in [envs/robosuite_env.py](../envs/robosuite_env.py) and BC eval helpers; camera stacking & flipping handled in `_stack_cameras()`.
- DMC/Atari/Crafter/Minecraft/Memory Maze: see `make_env()` in [dreamer.py](../dreamer.py) and [envs/](../envs).
- Weights & Biases: enabled by default; pass `WANDB_PROJECT`, `WANDB_ENTITY`, etc. via env vars.

## Gotchas
- Respect `batch_length` when preparing offline batches; padding must include correct `is_first` masks.
- When changing observation keys, update both encoder/decoder config and BC key ordering to avoid feature misalignment.
- Image cropping checks input size and returns contiguous crops; ensure dataset frames are large enough.
- If precision is 16, AMP is enabled in several modules; mind dtype casts in custom layers.

## Contributing in This Repo
- Prefer small, focused patches that match the current style; avoid reformatting unrelated code.
- Use the existing helpers: `_prepare_padded_episode`, `_prepare_full_episode`, `_build_ordered_obs_space`, `_build_encoder`, and `_infer_spaces` for new dataset/eval paths.
- When adding a new environment, extend `make_env()` and add wrappers under [envs/](../envs) as needed.

Feedback welcome: If any part of these instructions is unclear or incomplete for your task, tell us which workflow or component needs more detail and we’ll iterate.
