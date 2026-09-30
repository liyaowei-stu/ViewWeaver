#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
exec "${PYTHON:-$project_root/.venv/bin/python}" prepare.py \
  --vggt-model "${VGGT_MODEL:-checkpoints/vggt}" \
  --output-dir "${PREPARED_DIR:-outputs/prepared}" "$@"
