#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AEOS_ENV_BIN="${AEOS_ENV_BIN:-/home/hy/miniconda3/envs/aeos/bin}"
PYTHON_BIN="${AEOS_ENV_BIN}/python"
AUTO_TORCHRUN="${AEOS_ENV_BIN}/auto_torchrun"
export PATH="${AEOS_ENV_BIN}:${PATH}"
export PYTHONPATH=".:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

RUN_NAME="${RUN_NAME:-timemodel_duration_head_reset_pilot_2k}"
GPU_ID="${GPU_ID:-0}"
CALIBRATION_DEVICE="${CALIBRATION_DEVICE:-cpu}"
CALIBRATION_SCENES="${CALIBRATION_SCENES:-8}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth}"
CONFIG="constellation/new_transformers/config_timemodel_scale_pilot.py"
CALIBRATION_DIR="work_dirs/${RUN_NAME}/calibration"

mkdir -p "$CALIBRATION_DIR"

calibrate() {
  local checkpoint="$1"
  local tag="$2"
  local split="$3"
  "$PYTHON_BIN" tools/calibrate_timemodel_feasibility.py \
    "$checkpoint" \
    --split "$split" \
    --max-scenes "$CALIBRATION_SCENES" \
    --thresholds 0.01 0.03 0.1 0.3 0.5 \
    --device "$CALIBRATION_DEVICE" \
    --output "${CALIBRATION_DIR}/${tag}_${split}.json"
}

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "missing baseline checkpoint: $BASE_CHECKPOINT" >&2
  exit 1
fi

for split in val_seen val_unseen; do
  calibrate "$BASE_CHECKPOINT" baseline "$split"
done

CUDA_VISIBLE_DEVICES="$GPU_ID" "$AUTO_TORCHRUN" \
  -m constellation.new_transformers.train \
  "$RUN_NAME" \
  "$CONFIG" \
  --load-model-from "$BASE_CHECKPOINT"

for iteration in 500 1000 2000; do
  checkpoint="work_dirs/${RUN_NAME}/checkpoints/iter_${iteration}/model.pth"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing pilot checkpoint: $checkpoint" >&2
    exit 1
  fi
  for split in val_seen val_unseen; do
    calibrate "$checkpoint" "iter_${iteration}" "$split"
  done
done
