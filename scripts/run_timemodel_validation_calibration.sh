#!/usr/bin/env bash
set -euo pipefail

cd /home/hy/data/newaeosbench

export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH=":${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR=/tmp/matplotlib

checkpoint="${CHECKPOINT:-work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth}"
max_scenes="${MAX_SCENES:-64}"
output_dir="${OUTPUT_DIR:-work_dirs/timemodel_calibration}"

mkdir -p "${output_dir}" work_dirs/eval_logs

for split in val_seen val_unseen; do
  output="${output_dir}/${split}_stage3_200k_${max_scenes}.json"
  log="work_dirs/eval_logs/timemodel_calibration_${split}_${max_scenes}.log"
  python tools/calibrate_timemodel_feasibility.py \
    "${checkpoint}" \
    --split "${split}" \
    --max-scenes "${max_scenes}" \
    --thresholds 0.001 0.01 0.03 0.05 0.1 0.2 0.3 0.5 \
    --hard-negative-threshold 0.7 \
    --device cuda:0 \
    --output "${output}" 2>&1 | tee "${log}"
done
