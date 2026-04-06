#!/bin/bash
#SBATCH --job-name=joint-train-sweep
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_sku:RTX-A6000'
#SBATCH --partition=short
#SBATCH --time=36:00:00
#SBATCH --account=engs-a2i
#SBATCH --qos=engs-a2i
#SBATCH --reservation=a2i2025
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8
#SBATCH --mem=85G
#SBATCH --output=logdir/joint_train_sweep/joint_train_sweep-%A_%a.out
#SBATCH --error=logdir/joint_train_sweep/joint_train_sweep-%A_%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sedm7084@ox.ac.uk

set -euo pipefail

SIF="${CONTAINER:-containerv5.sif}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
BASE_LOGDIR="${BASE_LOGDIR:-logdir/joint_train_sweep}"
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

MODEL_LRS=("1e-4" "5e-4" "1e-3")
WM_LOSS_SCALES=("1" "0.1" "0.01")

RUNS=()
for MODEL_LR in "${MODEL_LRS[@]}"; do
    SAFE_MODEL_LR="${MODEL_LR//./p}"
    for WM_LOSS_SCALE in "${WM_LOSS_SCALES[@]}"; do
        SAFE_WM_LOSS_SCALE="${WM_LOSS_SCALE//./p}"
        RUN_NAME="lr${SAFE_MODEL_LR}_wm${SAFE_WM_LOSS_SCALE}"
        RUNS+=("${RUN_NAME}|${MODEL_LR}|${WM_LOSS_SCALE}")
    done
done

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX=$SLURM_ARRAY_TASK_ID
    if (( IDX >= ${#RUNS[@]} )); then
        echo "Array index $IDX exceeds number of runs (${#RUNS[@]}). Exiting."
        exit 0
    fi
    RUN_LIST=("${RUNS[$IDX]}")
else
    RUN_LIST=("${RUNS[@]}")
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
echo "=== Runs: ${#RUN_LIST[@]} ==="
echo "=== Base logdir: $BASE_LOGDIR ==="
echo "=== Env config: ${ENV_CONFIG:-<none>} ==="

for entry in "${RUN_LIST[@]}"; do
    IFS='|' read -r RUN_NAME MODEL_LR WM_LOSS_SCALE <<< "$entry"
    LOGDIR="${BASE_LOGDIR}/${RUN_NAME}"

    CMD=(
        "${RUNNER[@]}"
        --configs joint_train robomimic
        --logdir "$LOGDIR"
        --run_name "$RUN_NAME"
        --model_lr "$MODEL_LR"
        --wm_loss_scale "$WM_LOSS_SCALE"
    )

    if [[ -n "$ENV_CONFIG" ]]; then
        CMD+=(--env_config "$ENV_CONFIG")
    fi
    if (( ${#EXTRA_ARGS[@]} )); then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo ""
    echo "────────────────────────────────────────"
    echo "  Run: $RUN_NAME"
    echo "  Logdir: $LOGDIR"
    echo "  model_lr: $MODEL_LR"
    echo "  wm_loss_scale: $WM_LOSS_SCALE"
    echo "────────────────────────────────────────"

    "${CMD[@]}"

    echo "  Run '$RUN_NAME' finished with exit code $?."
done

echo ""
echo "=== All runs complete ==="
