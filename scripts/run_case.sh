#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

exec "${PYTHON:-$project_root/.venv/bin/python}" infer.py \
  --mode end-to-end \
  --vggt-model "${VGGT_MODEL:-checkpoints/vggt}" \
  --base-model "${FLUX_MODEL:-checkpoints/flux-kontext}" \
  --checkpoint "${VIEWWEAVER_CHECKPOINT:-checkpoints/viewweaver}" \
  --output-dir "${OUTPUT_DIR:-outputs}" \
  "$@"
