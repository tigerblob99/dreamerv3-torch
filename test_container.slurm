#!/bin/bash
#SBATCH --job-name=container-test
#SBATCH --output=test_container_%j.out
#SBATCH --error=test_container_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

SIF="${1:-container.sif}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Container image '$SIF' not found."
    exit 1
fi

echo "=== Testing container: $SIF ==="
echo "=== Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

apptainer exec --nv "$SIF" python "$SCRIPT_DIR/test_container.py"
