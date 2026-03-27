#!/bin/bash
#SBATCH --job-name=dreamer-sweep
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --account=engs-a2i
#SBATCH --qos=engs-a2i
#SBATCH --reservation=a2i2025
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3
#SBATCH --mem=56G
#SBATCH --output=logdir/sweep/slurm-%A_%a.out
#SBATCH --error=logdir/sweep/slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sedm7084@ox.ac.uk

set -euo pipefail

SIF="${CONTAINER:-container.sif}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
TASK="${TASK:-robosuite_NutAssemblySquare}"
BASE_LOGDIR="${BASE_LOGDIR:-logdir/sweep}"
EXPTDIR="${EXPTDIR:-datasets/robomimic_data_MV/Square_PH_Shaped_shifted_0-1}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.secrets/wandb.env}"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Container image '$SIF' not found."
    exit 1
fi

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

MUJOCO_GL="${MUJOCO_GL:-egl}"

for env_name in MUJOCO_GL WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_MODE; do
    if [[ -n "${!env_name:-}" ]]; then
        export "APPTAINERENV_${env_name}=${!env_name}"
    fi
done

# ── Define sweep configurations ──────────────────────────────────
# Each entry: "run_name|extra_flags"
CONFIGS=(
    "baseline|"
    "tr2048|--train_ratio 2048"
    "prefill10k|--prefill 10000"
    "tr2048_prefill10k|--train_ratio 2048 --prefill 10000"
)

# Pick config based on SLURM array index, or run all sequentially
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX=$SLURM_ARRAY_TASK_ID
    if (( IDX >= ${#CONFIGS[@]} )); then
        echo "Array index $IDX exceeds number of configs (${#CONFIGS[@]}). Exiting."
        exit 0
    fi
    RUN_LIST=("${CONFIGS[$IDX]}")
else
    RUN_LIST=("${CONFIGS[@]}")
fi

echo "=== Node: $(hostname) ==="
echo "=== GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A') ==="
echo "=== Task: $TASK ==="
echo "=== Runs: ${#RUN_LIST[@]} ==="

for entry in "${RUN_LIST[@]}"; do
    IFS='|' read -r RUN_NAME EXTRA_FLAGS <<< "$entry"
    LOGDIR="${BASE_LOGDIR}/${TASK}_${RUN_NAME}"

    echo ""
    echo "────────────────────────────────────────"
    echo "  Run: $RUN_NAME"
    echo "  Logdir: $LOGDIR"
    echo "  Extra flags: ${EXTRA_FLAGS:-<none>}"
    echo "────────────────────────────────────────"

    apptainer exec --nv \
        --bind "$PWD":"$PWD" \
        --pwd "$PWD" \
        "$SIF" \
        python "$SCRIPT_DIR/dreamer.py" \
            --configs robosuite \
            --task "$TASK" \
            --logdir "$LOGDIR" \
            --exptdir "$EXPTDIR" \
            $EXTRA_FLAGS

    echo "  Run '$RUN_NAME' finished with exit code $?."
done

echo ""
echo "=== All runs complete ==="
