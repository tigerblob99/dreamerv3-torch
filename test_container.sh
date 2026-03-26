#!/bin/bash
#SBATCH --job-name=container-test
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --account=engs-a2i
#SBATCH --qos=engs-a2i
#SBATCH --reservation=a2i2025

set -euo pipefail

SIF="${1:-container.sif}"
SCRIPT_DIR="$SLURM_SUBMIT_DIR"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Container image '$SIF' not found."
    exit 1
fi

echo "=== Testing container: $SIF ==="
echo "=== Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

apptainer exec --nv --bind "$PWD":"$PWD" --pwd "$PWD" "$SIF" python "$SCRIPT_DIR/test_container.py"
