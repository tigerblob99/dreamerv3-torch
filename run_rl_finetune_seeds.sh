#!/bin/bash
#SBATCH --job-name=rl-finetune-seeds
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
#SBATCH --output=logdir/rl_finetune_seeds/slurm/array_%a/%x-%A.out
#SBATCH --error=logdir/rl_finetune_seeds/slurm/array_%a/%x-%A.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sedm7084@ox.ac.uk

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# EDIT HERE — source of joint_train checkpoints
#   Each seed's checkpoint is resolved as:
#     ${JOINT_BASE_LOGDIR}/seed_${SEED}_${DATA_TAG}/${CKPT_NAME}
#   which matches the layout produced by run_joint_train_seeds.sh.
# ────────────────────────────────────────────────────────────────
JOINT_SWEEP_NAME="${JOINT_SWEEP_NAME:-joint_train_seeds}"
DATA_TAG="${DATA_TAG:-expert_only}"           # must match joint_train's DATA_TAG
CKPT_NAME="${CKPT_NAME:-latest.pt}"           # e.g. latest.pt | wm_pretrain_end.pt | step_50000.pt
JOINT_BASE_LOGDIR="${JOINT_BASE_LOGDIR:-logdir/${JOINT_SWEEP_NAME}}"

# ────────────────────────────────────────────────────────────────
# EDIT HERE — RL finetune run identity & data
# ────────────────────────────────────────────────────────────────
SWEEP_NAME="${SWEEP_NAME:-rl_finetune_seeds}"
BASE_LOGDIR="${BASE_LOGDIR:-logdir/${SWEEP_NAME}}"
EXPERT_DIR="${EXPERT_DIR:-datasets/robomimic_data_MV/can_PH_Shaped_shifted_0-1}"
ENV_CONFIG="${ENV_CONFIG:-can_env_eval}"
CONFIGS_ARR=(robomimic rl_finetune)           # passed to --configs
POLICY_INIT="${POLICY_INIT:-checkpoint}"      # checkpoint | random

# ────────────────────────────────────────────────────────────────
# Seeds for the sweep (one per array index) — must match run_joint_train_seeds.sh
# ────────────────────────────────────────────────────────────────
SEEDS=(0 3 5 7 9)

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

# Override wandb project for this sweep (after sourcing the env file, so it wins).
export WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-dreamerv3-rl-finetune-seeds}"

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
        "$SCRIPT_DIR/RL_finetune.py"
    )
else
    RUNNER=(
        "$PYTHON_BIN"
        "$SCRIPT_DIR/RL_finetune.py"
    )
fi

echo "=== Node: $(hostname) ==="
echo "=== GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A') ==="
echo "=== Seeds:            ${SEED_LIST[*]} ==="
echo "=== RL base logdir:   $BASE_LOGDIR ==="
echo "=== Joint sweep:      $JOINT_SWEEP_NAME  (base=$JOINT_BASE_LOGDIR)"
echo "=== Data tag:         $DATA_TAG ==="
echo "=== Checkpoint file:  $CKPT_NAME ==="
echo "=== Configs:          ${CONFIGS_ARR[*]} ==="
echo "=== Env config:       ${ENV_CONFIG:-<none>} ==="
echo "=== Expert dir:       $EXPERT_DIR ==="
echo "=== Policy init:      $POLICY_INIT ==="
echo "=== WANDB project:    $WANDB_PROJECT ==="

# ────────────────────────────────────────────────────────────────
# Per-seed launches
# ────────────────────────────────────────────────────────────────
for SEED in "${SEED_LIST[@]}"; do
    JOINT_RUN_NAME="seed_${SEED}_${DATA_TAG}"
    JOINT_LOGDIR="${JOINT_BASE_LOGDIR}/${JOINT_RUN_NAME}"
    JOINT_CKPT="${JOINT_LOGDIR}/${CKPT_NAME}"

    RUN_NAME="seed_${SEED}_${DATA_TAG}_${CKPT_NAME%.pt}"
    LOGDIR="${BASE_LOGDIR}/${RUN_NAME}"

    if [[ ! -f "$JOINT_CKPT" ]]; then
        echo ""
        echo "ERROR: joint_train checkpoint not found: $JOINT_CKPT"
        echo "       Check JOINT_SWEEP_NAME ($JOINT_SWEEP_NAME),"
        echo "             DATA_TAG ($DATA_TAG),"
        echo "             CKPT_NAME ($CKPT_NAME)."
        exit 1
    fi

    CMD=(
        "${RUNNER[@]}"
        --configs "${CONFIGS_ARR[@]}"
        --env_config "$ENV_CONFIG"
        --logdir "$LOGDIR"
        --seed "$SEED"
        --checkpoint "$JOINT_CKPT"
        --policy_init "$POLICY_INIT"
        --expert_dir "$EXPERT_DIR"
    )

    if (( ${#EXTRA_ARGS[@]} )); then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo ""
    echo "────────────────────────────────────────"
    echo "  Sweep:      $SWEEP_NAME"
    echo "  Run:        $RUN_NAME"
    echo "  Seed:       $SEED"
    echo "  Logdir:     $LOGDIR"
    echo "  Joint ckpt: $JOINT_CKPT"
    echo "────────────────────────────────────────"

    "${CMD[@]}"

    echo "  Run '$RUN_NAME' finished with exit code $?."
done

echo ""
echo "=== All runs complete ==="
