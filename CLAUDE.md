# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch implementation of DreamerV3 (world model-based RL), extended with offline training, behavior cloning (BC), joint world-model+BC training, and on-policy RL fine-tuning. Primary target environment is RoboSuite/RoboMimic manipulation tasks, with support for DMC, Atari, Crafter, Minecraft, and Memory Maze.

## Commands

### Setup
```bash
pip install -r requirements.txt   # Python 3.11+, CUDA GPU required
```

### Training

**Online RL (DMC Vision):**
```bash
python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```

**Offline world model training (RoboMimic):**
```bash
python offline_train.py --configs robomimic \
  --offline_traindir ./datasets/robomimic_data_MV/can_MH_train \
  --offline_evaldir ./datasets/robomimic_data_MV/can_MH_eval \
  --logdir ./logdir/robomimic_offline
```

**Joint training (world model + BC):**
```bash
python joint_train.py --configs joint_train robomimic \
  --offline_traindir datasets/robomimic_data_MV/can_PH_train \
  --offline_evaldir datasets/robomimic_data_MV/can_PH_eval \
  --offline_playdir datasets/robomimic_data_MV/can_MH_train \
  --logdir logdir/joint_run
```

**Behavior cloning with frozen Dreamer encoder:**
```bash
python BC_MLP_train.py --configs bc_defaults bc_can_PH \
  --checkpoint logdir/robomimic_offline/latest.pt \
  --offline_traindir datasets/robomimic_data_MV/can_PH_train
```

**On-policy RL fine-tuning:**
```bash
python AC_RL/RL_train.py --configs rl_train
```

**Monitoring:**
```bash
tensorboard --logdir ./logdir
```

## Architecture

### Core Training Loop (dreamer.py)
The `Dreamer` agent (nn.Module) owns two main components:
- **WorldModel** (`models.py`): Encoder → RSSM latent dynamics → decoder heads (reconstruction, reward, continuation)
- **ImagBehavior** (`models.py`): Actor-critic trained on imagined rollouts in latent space

Online loop: collect env experience → sample batches from replay → train world model → imagine trajectories → train actor-critic.

### World Model Data Flow
```
Observations → MultiEncoder (CNN for images, MLP for vectors)
  → RSSM.observe (GRU deterministic + stochastic latent)
  → get_feat (concat stoch + deter)
  → Decoder heads (image reconstruction, reward, continuation)
```

### Key Model Components (networks.py)
- **RSSM**: Recurrent state-space model with discrete latent (`dyn_discrete > 0` → multi-categorical) or continuous stochastic state. `is_first` masks reset state at episode boundaries.
- **MultiEncoder/MultiDecoder**: Handle mixed observation types. `cnn_keys` regex selects image inputs (CNN path), `mlp_keys` regex selects vector inputs (MLP path).
- **Symlog distributions**: Used for reward/value heads for stable learning across reward scales.

### Configuration System (configs.yaml)
- `defaults` block contains all hyperparameters
- Named config blocks overlay on defaults: `--configs config1 config2 ...`
- Configs merge left-to-right; later configs override earlier ones
- YAML anchors (`&robomimic_encoder` / `*robomimic_encoder`) share encoder/decoder definitions across configs

### Dataset Backends
- **ReplayStoreDataset** (`tools.py`): GPU-resident replay buffer for online training. Supports `'shorter'` (inverse-length weighting) or `'uniform'` episode sampling.
- **Generator-based** (`tools.load_episodes` → `tools.sample_episodes` → `tools.from_generator`): For offline training from .npz/.hdf5 files.

### Training Scripts Beyond dreamer.py
- `offline_train.py`: Offline world model training/eval. Two batch modes: `_prepare_padded_episode()` (fixed batch_length) and `_prepare_full_episode()` (preserve episode length).
- `joint_train.py`: Jointly trains world model + BC policy. Mixes expert and play data via `expert_data_fraction`.
- `BC_MLP_train.py` / `bc_mlp/BC_MLP_eval.py`: Train/evaluate MLP policy on frozen Dreamer encoder features.
- `AC_RL/RL_train.py`: On-policy actor-critic with frozen encoder+RSSM.

### Environment Integration
- Environment wrappers live in `envs/`. New environments extend `make_env()` in `dreamer.py`.
- RoboSuite wrapper (`envs/robosuite_env.py`): Multi-camera stacking (`_stack_cameras`), camera flipping, auxiliary sin/cos joint features.
- `parallel.py`: `Parallel` (multiprocess) and `Damy` (in-thread) for vectorized environments.

## Key Patterns

- **Observation key regexes**: Encoder/decoder configs use `mlp_keys` and `cnn_keys` as regex patterns to select observation keys. For BC, explicit ordering is set via `bc_mlp_keys_order` and `bc_cnn_keys_order` — these must stay consistent with encoder config.
- **Episode format**: Dicts of numpy arrays with keys `image` (T,H,W,C uint8), `action` (T,A float32), `reward`, `discount`, `is_first`, `is_terminal`.
- **Images are channels-last** (T, H, W, C) throughout. Encoders normalize to [0,1].
- **`is_first` masks**: Critical for RSSM — they reset the recurrent state at episode boundaries during `observe`.
- **AMP**: Enabled when `precision: 16`. `tools.Optimizer` wraps Adam with gradient clipping and optional mixed precision.
- **`tools.RequiresGrad`**: Context manager for selective gradient computation on model subsets.
- **`MUJOCO_GL=osmesa`**: Set in entry-point scripts for headless rendering. Keep for CI/server runs.
- **Logging**: `tools.Logger` integrates TensorBoard + wandb. Set `WANDB_PROJECT`/`WANDB_ENTITY` env vars.

## Gotchas

- When changing observation keys, update both encoder/decoder config regexes AND `bc_mlp_keys_order`/`bc_cnn_keys_order` — misalignment causes silent feature corruption.
- `batch_length` padding must include correct `is_first` masks or RSSM state will not reset properly.
- Image cropping validates input dimensions — ensure dataset frames are large enough for configured crop size.
- Don't mix float16 precision without understanding AMP casting in RSSM and decoder layers.
- The decoder's `mlp_keys` typically excludes `aux_*` keys (sin/cos features are encoder-only inputs, not reconstruction targets).
