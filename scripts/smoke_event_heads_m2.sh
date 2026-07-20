#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/hy/data/newaeosbench"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CHECKPOINT="${ROOT_DIR}/work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth"
ANNOTATION="${ROOT_DIR}/data/annotations/train_paper_stage3_tau_e_existing.json"
OUTPUT="${ROOT_DIR}/work_dirs/event_supervision_m2_preflight/train_step_scene0.json"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"

"${PYTHON}" tools/smoke_event_heads_m2.py \
  "${CHECKPOINT}" \
  --annotation-file "${ANNOTATION}" \
  --split train \
  --scene-index 0 \
  --batch-size 2 \
  --output "${OUTPUT}"
