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
#SBATCH --output=logdir/joint_train_seeds/slurm/array_%a/%x-%A.out
#SBATCH --error=logdir/joint_train_seeds/slurm/array_%a/%x-%A.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sedm7084@ox.ac.uk

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Data dirs (edit here to change dataset for all seeds)
# ────────────────────────────────────────────────────────────────
OFFLINE_TRAINDIR="${OFFLINE_TRAINDIR:-datasets/robomimic_data_MV/can_PH_Shaped_shifted_0-1}"
OFFLINE_EVALDIR="${OFFLINE_EVALDIR:-datasets/robomimic_data_MV/can_MH_train}"
OFFLINE_PLAYDIR="${OFFLINE_PLAYDIR:-}"

# ────────────────────────────────────────────────────────────────
# Seeds for the sweep (one per array index)
# ────────────────────────────────────────────────────────────────
SEEDS=(0 3 5 7 9)

# ────────────────────────────────────────────────────────────────
# Sweep identity — downstream finetune scripts should reuse SWEEP_NAME
# and resolve checkpoints as: logdir/${SWEEP_NAME}/seed_${SEED}/latest.pt
# ────────────────────────────────────────────────────────────────
SWEEP_NAME="${SWEEP_NAME:-joint_train_seeds}"

# ────────────────────────────────────────────────────────────────
# Runtime / environment
# ────────────────────────────────────────────────────────────────
SIF="${CONTAINER:-containerv5.sif}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
BASE_LOGDIR="${BASE_LOGDIR:-logdir/${SWEEP_NAME}}"
ENV_CONFIG="${ENV_CONFIG:-can_env_eval}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.secrets/wandb.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "$WANDB_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$WANDB_ENV_FILE"
fi

# Override wandb project for this sweep (after sourcing the env file, so it wins).
export WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-dreamerv3-joint-train-seeds}"

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

mkdir -p "$BASE_LOGDIR" "$BASE_LOGDIR/slurm"

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
echo "=== Play  dir: ${OFFLINE_PLAYDIR:-<none — expert-only mode>} ==="
echo "=== WANDB project: $WANDB_PROJECT ==="

sanitize_path_tag() {
    # Lowercase alnum / dash / underscore only; collapse runs; trim edges.
    local s="$1"
    s="${s// /_}"
    s="$(printf '%s' "$s" | tr -c 'A-Za-z0-9._-' '_' | tr -s '_' | sed 's/^_//;s/_$//')"
    printf '%s' "$s"
}

if [[ -n "$OFFLINE_PLAYDIR" ]]; then
    # Last two path components of the play dir, e.g.
    #   logdir/robosuite_square_reset_every_5e4_run2 (larger latent)/train_eps
    # becomes: robosuite_square_reset_every_5e4_run2_larger_latent__train_eps
    PLAY_TRIMMED="${OFFLINE_PLAYDIR%/}"
    PLAY_PARENT="$(sanitize_path_tag "$(basename "$(dirname "$PLAY_TRIMMED")")")"
    PLAY_LEAF="$(sanitize_path_tag "$(basename "$PLAY_TRIMMED")")"
    DATA_TAG="expert_play-${PLAY_PARENT}__${PLAY_LEAF}"
else
    DATA_TAG="expert_only"
fi

for SEED in "${SEED_LIST[@]}"; do
    RUN_NAME="seed_${SEED}_${DATA_TAG}"
    LOGDIR="${BASE_LOGDIR}/${RUN_NAME}"
    CKPT_PATH="${LOGDIR}/latest.pt"

    CMD=(
        "${RUNNER[@]}"
        --configs joint_train robomimic
        --logdir "$LOGDIR"
        --run_name "$RUN_NAME"
        --seed "$SEED"
        --offline_traindir "$OFFLINE_TRAINDIR"
        --offline_evaldir "$OFFLINE_EVALDIR"
    )

    if [[ -n "$OFFLINE_PLAYDIR" ]]; then
        CMD+=(--offline_playdir "$OFFLINE_PLAYDIR")
    else
        CMD+=(--expert_data_fraction 1.0)
    fi

    if [[ -n "$ENV_CONFIG" ]]; then
        CMD+=(--env_config "$ENV_CONFIG")
    fi
    if (( ${#EXTRA_ARGS[@]} )); then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo ""
    echo "────────────────────────────────────────"
    echo "  Sweep:  $SWEEP_NAME"
    echo "  Run:    $RUN_NAME"
    echo "  Seed:   $SEED"
    echo "  Logdir: $LOGDIR"
    echo "  Ckpt:   $CKPT_PATH  (expected after training)"
    echo "────────────────────────────────────────"

    "${CMD[@]}"

    echo "  Run '$RUN_NAME' finished with exit code $?."
done

echo ""
echo "=== All runs complete ==="
