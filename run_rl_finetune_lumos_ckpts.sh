#!/bin/bash
#SBATCH --job-name=rl-lumos-ckpts
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_sku:RTX-A6000'
#SBATCH --partition=short
#SBATCH --time=36:00:00
#SBATCH --account=engs-a2i
#SBATCH --qos=engs-a2i
#SBATCH --reservation=a2i2025
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --mem=31G
#SBATCH --output=logdir/rl_finetune_lumos_ckpts/slurm/array_%a/%x-%A.out
#SBATCH --error=logdir/rl_finetune_lumos_ckpts/slurm/array_%a/%x-%A.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sedm7084@ox.ac.uk

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# EDIT HERE — list of (BC checkpoint | expert_dir | env_config) entries.
# One SLURM array task per entry. Set --array=0-$((N-1)) when launching:
#     sbatch --array=0-$((N-1)) run_rl_finetune_lumos_ckpts.sh
# Format per line: "<ckpt path>|<expert_dir>|<env_config>"
# Pipes are the separator so paths-with-spaces and parens survive.
# ────────────────────────────────────────────────────────────────
CKPT_ENTRIES=(
  "logdir/joint_train_seeds_square_PH/seed_0_expert_only/latest.pt|datasets/robomimic_data_MV/Square_PH_Shaped_shifted_0-1|square_env_eval"
  "logdir/joint_train_seeds/seed_0_expert_only/latest.pt|datasets/robomimic_data_MV/can_PH_Shaped_shifted_0-1|can_env_eval"
  "logdir/joint_train_seeds/seed_0_expert_play-robomimic_data_MV__can_MH_Shaped_shifted_0-1/latest.pt|datasets/robomimic_data_MV/can_PH_Shaped_shifted_0-1|can_env_eval"
  "logdir/joint_train_seeds_can_warmstart/seed_0_expert_play-robomimic_data_MV__can_MH_Shaped_shifted_0-1/latest.pt|datasets/robomimic_data_MV/can_PH_Shaped_shifted_0-1|can_env_eval"
)

# ────────────────────────────────────────────────────────────────
# Sweep identity
# ────────────────────────────────────────────────────────────────
SWEEP_NAME="${SWEEP_NAME:-rl_finetune_lumos_ckpts}"
BASE_LOGDIR="${BASE_LOGDIR:-logdir/${SWEEP_NAME}}"
CONFIGS_ARR=(robomimic rl_finetune_lumos)        # passed to --configs
POLICY_INIT="${POLICY_INIT:-checkpoint}"         # checkpoint | random

# ────────────────────────────────────────────────────────────────
# Runtime / environment
# ────────────────────────────────────────────────────────────────
SIF="${CONTAINER:-containerv5.sif}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.secrets/wandb.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "$WANDB_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$WANDB_ENV_FILE"
fi

# New wandb project for this sweep — overrides anything in wandb.env so all
# checkpoint runs land in one bucket.
export WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-dreamerv3-rl-finetune-lumos-ckpts}"

WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE,,}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set for an online wandb run."
    echo "Create $WANDB_ENV_FILE with your wandb exports, or export WANDB_API_KEY before launching."
    exit 1
fi

if [[ -n "${CONTAINER:-}" && ! -f "$SIF" ]]; then
    echo "ERROR: Container image '$SIF' not found."
    exit 1
fi

mkdir -p "$BASE_LOGDIR" "$BASE_LOGDIR/slurm"

# ────────────────────────────────────────────────────────────────
# Pick entry for this array task (or run all sequentially if not in an array)
# ────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX=$SLURM_ARRAY_TASK_ID
    if (( IDX >= ${#CKPT_ENTRIES[@]} )); then
        echo "Array index $IDX exceeds number of entries (${#CKPT_ENTRIES[@]}). Exiting."
        exit 0
    fi
    ENTRY_LIST=("${CKPT_ENTRIES[$IDX]}")
else
    ENTRY_LIST=("${CKPT_ENTRIES[@]}")
fi

EXTRA_ARGS=("$@")

# Build the python invocation prefix once
if [[ -f "$SIF" ]]; then
    RUNNER=(
        apptainer exec --nv
        --bind "$PWD":"$PWD"
        --pwd "$PWD"
        "$SIF"
        "$PYTHON_BIN"
        "$SCRIPT_DIR/RL_finetune_lumos.py"
    )
else
    RUNNER=(
        "$PYTHON_BIN"
        "$SCRIPT_DIR/RL_finetune_lumos.py"
    )
fi

echo "=== Node: $(hostname) ==="
echo "=== GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A') ==="
echo "=== Sweep:         $SWEEP_NAME ==="
echo "=== Base logdir:   $BASE_LOGDIR ==="
echo "=== WANDB project: $WANDB_PROJECT ==="
echo "=== Entries: ${#ENTRY_LIST[@]} (array task ${SLURM_ARRAY_TASK_ID:-<none>}) ==="

# ────────────────────────────────────────────────────────────────
# Per-entry launch
# ────────────────────────────────────────────────────────────────
for ENTRY in "${ENTRY_LIST[@]}"; do
    IFS='|' read -r CKPT EXPERT_DIR ENV_CONFIG <<< "$ENTRY"

    if [[ -z "$CKPT" || -z "$EXPERT_DIR" || -z "$ENV_CONFIG" ]]; then
        echo "ERROR: malformed entry (need ckpt|expert_dir|env_config): $ENTRY"
        exit 1
    fi
    if [[ ! -f "$CKPT" ]]; then
        echo "ERROR: checkpoint not found: $CKPT"
        exit 1
    fi
    if [[ ! -d "$EXPERT_DIR" ]]; then
        echo "ERROR: expert_dir not found: $EXPERT_DIR"
        exit 1
    fi

    # Tag = "<sweep parent>__<run dir>" derived from the checkpoint path so
    # both the on-disk logdir and the wandb run name say which BC checkpoint
    # this is. e.g.:
    #   logdir/joint_train_seeds_square/seed_0_expert_play-.../latest.pt
    #     -> joint_train_seeds_square__seed_0_expert_play-...
    CKPT_DIR_NAME=$(basename "$(dirname "$CKPT")")
    SWEEP_PARENT=$(basename "$(dirname "$(dirname "$CKPT")")")
    RAW_TAG="${SWEEP_PARENT}__${CKPT_DIR_NAME}"
    # Sanitize: spaces, slashes, parens -> underscores so the path is safe.
    SAFE_TAG=$(echo "$RAW_TAG" | tr ' ()/' '____')

    LOGDIR="${BASE_LOGDIR}/${SAFE_TAG}"
    RUN_NAME="lumos_${SAFE_TAG}"

    # WANDB_NAME is read by RL_finetune_lumos.py at wandb.init() time
    # (RL_finetune_lumos.py:875). Re-export per entry so each run gets its
    # own name; APPTAINERENV_* propagates it into the container.
    export WANDB_NAME="$RUN_NAME"
    export APPTAINERENV_WANDB_NAME="$WANDB_NAME"
    for env_name in WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_MODE; do
        if [[ -n "${!env_name:-}" ]]; then
            export "APPTAINERENV_${env_name}=${!env_name}"
        fi
    done

    CMD=(
        "${RUNNER[@]}"
        --configs "${CONFIGS_ARR[@]}"
        --env_config "$ENV_CONFIG"
        --logdir "$LOGDIR"
        --checkpoint "$CKPT"
        --policy_init "$POLICY_INIT"
        --expert_dir "$EXPERT_DIR"
    )

    if (( ${#EXTRA_ARGS[@]} )); then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo ""
    echo "────────────────────────────────────────"
    echo "  Run name:    $RUN_NAME"
    echo "  Checkpoint:  $CKPT"
    echo "  Expert dir:  $EXPERT_DIR"
    echo "  Env config:  $ENV_CONFIG"
    echo "  Logdir:      $LOGDIR"
    echo "────────────────────────────────────────"

    "${CMD[@]}"

    echo "  Run '$RUN_NAME' finished with exit code $?."
done

echo ""
echo "=== All runs complete ==="
