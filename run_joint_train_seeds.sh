#!/bin/bash
#SBATCH --job-name=joint-train-seeds
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_sku:RTX-A6000'
#SBATCH --partition=short
#SBATCH --time=36:00:00
#SBATCH --account=engs-a2i
#SBATCH --qos=engs-a2i
#SBATCH --reservation=a2i2025
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --mem=85G
#SBATCH --output=logdir/joint_train_seeds/joint_train_seeds-%A_%a.out
#SBATCH --error=logdir/joint_train_seeds/joint_train_seeds-%A_%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sedm7084@ox.ac.uk

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Data dirs (edit here to change dataset for all seeds)
# ────────────────────────────────────────────────────────────────
OFFLINE_TRAINDIR="${OFFLINE_TRAINDIR:-datasets/robomimic_data_MV/Square_PH_Shaped_shifted_0-1_train}"
OFFLINE_EVALDIR="${OFFLINE_EVALDIR:-logdir/robosuite_square_reset_every_5e4_run2 (larger latent)/eval_eps}"
OFFLINE_PLAYDIR="${OFFLINE_PLAYDIR:-logdir/robosuite_square_reset_every_5e4_run2 (larger latent)/train_eps}"

# ────────────────────────────────────────────────────────────────
# Seeds for the sweep (one per array index)
# ────────────────────────────────────────────────────────────────
SEEDS=(0 1 2 3 4)

# ────────────────────────────────────────────────────────────────
# Runtime / environment
# ────────────────────────────────────────────────────────────────
SIF="${CONTAINER:-containerv5.sif}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
BASE_LOGDIR="${BASE_LOGDIR:-logdir/joint_train_seeds}"
ENV_CONFIG="${ENV_CONFIG:-square_env_eval}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.secrets/wandb.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "$WANDB_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$WANDB_ENV_FILE"
fi

WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE,,}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set for an online wandb run."
    echo "Create $WANDB_ENV_FILE with your wandb exports, or export WANDB_API_KEY before launching."
    exit 1
fi

for env_name in WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_MODE; do
    if [[ -n "${!env_name:-}" ]]; then
        export "APPTAINERENV_${env_name}=${!env_name}"
    fi
done

mkdir -p "$BASE_LOGDIR"

# ────────────────────────────────────────────────────────────────
# Pick seed for this array task (fall back to running all if not in an array)
# ────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX=$SLURM_ARRAY_TASK_ID
    if (( IDX >= ${#SEEDS[@]} )); then
        echo "Array index $IDX exceeds number of seeds (${#SEEDS[@]}). Exiting."
        exit 0
    fi
    SEED_LIST=("${SEEDS[$IDX]}")
else
    SEED_LIST=("${SEEDS[@]}")
fi

EXTRA_ARGS=("$@")

if [[ -n "${CONTAINER:-}" && ! -f "$SIF" ]]; then
    echo "ERROR: Container image '$SIF' not found."
    exit 1
fi

if [[ -f "$SIF" ]]; then
    RUNNER=(
        apptainer exec --nv
        --bind "$PWD":"$PWD"
        --pwd "$PWD"
        "$SIF"
        "$PYTHON_BIN"
        "$SCRIPT_DIR/joint_train.py"
    )
else
    RUNNER=(
        "$PYTHON_BIN"
        "$SCRIPT_DIR/joint_train.py"
    )
fi

echo "=== Node: $(hostname) ==="
echo "=== GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A') ==="
echo "=== Seeds: ${SEED_LIST[*]} ==="
echo "=== Base logdir: $BASE_LOGDIR ==="
echo "=== Env config: ${ENV_CONFIG:-<none>} ==="
echo "=== Train dir: $OFFLINE_TRAINDIR ==="
echo "=== Eval  dir: $OFFLINE_EVALDIR ==="
echo "=== Play  dir: $OFFLINE_PLAYDIR ==="

for SEED in "${SEED_LIST[@]}"; do
    RUN_NAME="seed_${SEED}"
    LOGDIR="${BASE_LOGDIR}/${RUN_NAME}"

    CMD=(
        "${RUNNER[@]}"
        --configs joint_train robomimic
        --logdir "$LOGDIR"
        --run_name "$RUN_NAME"
        --seed "$SEED"
        --offline_traindir "$OFFLINE_TRAINDIR"
        --offline_evaldir "$OFFLINE_EVALDIR"
        --offline_playdir "$OFFLINE_PLAYDIR"
    )

    if [[ -n "$ENV_CONFIG" ]]; then
        CMD+=(--env_config "$ENV_CONFIG")
    fi
    if (( ${#EXTRA_ARGS[@]} )); then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo ""
    echo "────────────────────────────────────────"
    echo "  Run:    $RUN_NAME"
    echo "  Seed:   $SEED"
    echo "  Logdir: $LOGDIR"
    echo "────────────────────────────────────────"

    "${CMD[@]}"

    echo "  Run '$RUN_NAME' finished with exit code $?."
done

echo ""
echo "=== All runs complete ==="
