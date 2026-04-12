#!/bin/bash
#SBATCH --job-name=dreamer
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_sku:RTX-A6000'
#SBATCH --time=36:00:00
#SBATCH --partition=short
#SBATCH --account=engs-a2i
#SBATCH --reservation=a2i2025
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --output=logdir/dreamer-%j.out
#SBATCH --error=logdir/dreamer-%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sedm7084@ox.ac.uk

set -euo pipefail

SIF="${CONTAINER:-containerv5.sif}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
TASK="${TASK:-robosuite_NutAssemblySquare}"
LOGDIR="${LOGDIR:-logdir/${TASK}}"
EXPTDIR="${EXPTDIR:-datasets/robomimic_data_MV/Square_PH_Shaped_shifted_0-1}"
ROBOSUITE_RENDER_DEVICE="${ROBOSUITE_RENDER_DEVICE:-0}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.secrets/wandb.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "$WANDB_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$WANDB_ENV_FILE"
fi

WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE,,}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set for an online wandb run."
    echo "Create $WANDB_ENV_FILE with your wandb exports, or export WANDB_API_KEY before sbatch."
    exit 1
fi

for env_name in WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_MODE; do
    if [[ -n "${!env_name:-}" ]]; then
        export "APPTAINERENV_${env_name}=${!env_name}"
    fi
done

mkdir -p "$(dirname "$LOGDIR")"

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
        "$SCRIPT_DIR/dreamer.py"
    )
else
    RUNNER=(
        "$PYTHON_BIN"
        "$SCRIPT_DIR/dreamer.py"
    )
fi

echo "=== Node: $(hostname) ==="
echo "=== GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A') ==="
echo "=== Task: $TASK ==="
echo "=== Logdir: $LOGDIR ==="
echo "=== RoboSuite render device: $ROBOSUITE_RENDER_DEVICE ==="

CMD=(
    "${RUNNER[@]}"
    --configs robosuite
    --task "$TASK"
    --logdir "$LOGDIR"
    --exptdir "$EXPTDIR"
    --robosuite_render_device "$ROBOSUITE_RENDER_DEVICE"
)

if (( ${#EXTRA_ARGS[@]} )); then
    CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

echo "=== Run complete ==="
