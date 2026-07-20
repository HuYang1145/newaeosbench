#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/hy/data/newaeosbench"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CHECKPOINT="${ROOT_DIR}/work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth"
OUTPUT_ROOT="${ROOT_DIR}/work_dirs/event_actor_m1_smoke"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[error] aeos python not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] Stage3 checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON}" tools/rollout_model_trajectories.py \
  "${CHECKPOINT}" \
  "${OUTPUT_ROOT}/baseline" \
  --split train \
  --limit 1 \
  --device cpu \
  --strategy greedy \
  --overwrite

"${PYTHON}" tools/rollout_model_trajectories.py \
  "${CHECKPOINT}" \
  "${OUTPUT_ROOT}/event_5s" \
  --split train \
  --limit 1 \
  --device cpu \
  --strategy greedy \
  --event-actor \
  --event-commitment-seconds 5 \
  --event-idle-commitment-seconds 1 \
  --overwrite

echo "[done] baseline=${OUTPUT_ROOT}/baseline"
echo "[done] event_5s=${OUTPUT_ROOT}/event_5s"
