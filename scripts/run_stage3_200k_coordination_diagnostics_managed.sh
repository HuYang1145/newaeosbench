#!/usr/bin/env bash
set -euo pipefail

cd /home/hy/data/newaeosbench

export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="/home/hy/data/newaeosbench:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WORLD_SIZE=1
export RANK=0
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/aeos_cache

workers="${WORKERS:-16}"
max_scenes="${MAX_SCENES:-64}"
top_k="${TOP_K:-5}"
model_path="work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth"

mkdir -p work_dirs/eval_logs

for split in val_seen val_unseen; do
  run_name="stage3_200k_coordination_top${top_k}_${split}_${max_scenes}"
  log_path="work_dirs/eval_logs/${run_name}.log"
  echo "===== $(date '+%F %T') start ${split} =====" | tee -a "${log_path}"
  /home/hy/miniconda3/envs/aeos/bin/python \
    -m constellation.rl.eval_all \
    "${run_name}" \
    constellation/rl/config_eval.py \
    --override \
    "[\"environment\"][\"world_size\"]:${workers}" \
    "[\"environment\"][\"split\"]:\"${split}\"" \
    --max-scenes "${max_scenes}" \
    --coordination-diagnostics-top-k "${top_k}" \
    --load-model-from "${model_path}" \
    2>&1 | tee -a "${log_path}"
  echo "===== $(date '+%F %T') done ${split} =====" | tee -a "${log_path}"
done
