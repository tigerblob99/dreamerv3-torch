#!/usr/bin/env bash
# Run the container smoke-test inside Apptainer.
# Usage:  ./test_container.sh [container.sif]

set -euo pipefail

SIF="${1:-container.sif}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Container image '$SIF' not found."
    echo "Usage: $0 [path/to/container.sif]"
    exit 1
fi

echo "=== Testing container: $SIF ==="
apptainer exec --nv "$SIF" python "$SCRIPT_DIR/test_container.py"
