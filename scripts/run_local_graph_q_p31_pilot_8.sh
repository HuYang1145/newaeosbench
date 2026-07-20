#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/hy/data/newaeosbench"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CHECKPOINT="${ROOT_DIR}/work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth"
REFERENCE_ROOT="${ROOT_DIR}/work_dirs/same_scene_candidates_stage3_200k_256/candidate_000_greedy"
DATA_ROOT="${ROOT_DIR}/work_dirs/local_graph_q_p31_pilot_8"
CRITIC_ROOT="${ROOT_DIR}/work_dirs/local_graph_q_p31_critic_pilot_8"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"

"${PYTHON}" tools/generate_local_graph_q_dataset.py \
  "${CHECKPOINT}" \
  "${REFERENCE_ROOT}" \
  "${DATA_ROOT}" \
  --split train \
  --limit 8 \
  --horizons 180 300 600 \
  --primary-horizon 300 \
  --max-decisions 1 \
  --top-k 3 \
  --device cpu \
  --scene-workers 2 \
  --threads-per-scene 24

"${PYTHON}" tools/train_local_graph_q_critic.py \
  "${DATA_ROOT}" \
  --output-dir "${CRITIC_ROOT}" \
  --num-folds 4 \
  --hidden-dim 64 \
  --epochs 150 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --outcome-loss-weight 0.2 \
  --min-cost-margin 0.001 \
  --primary-horizon 300 \
  --check-horizon 600 \
  --device cpu \
  --num-threads 16
