#!/usr/bin/env bash
set -euo pipefail

cd /home/hy/data/newaeosbench

export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH=":${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/aeos_cache
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p work_dirs/eval_logs

run_stage() {
  local name="$1"
  local config="$2"
  shift 2
  local log_path="work_dirs/eval_logs/${name}_train.log"
  echo "===== $(date '+%F %T') start ${name} =====" | tee -a "${log_path}"
  auto_torchrun -m constellation.new_transformers.train \
    "${name}" \
    "${config}" \
    --auto-resume \
    "$@" 2>&1 | tee -a "${log_path}"
  echo "===== $(date '+%F %T') done ${name} =====" | tee -a "${log_path}"
}

run_stage \
  paper_joint_stage1_200k \
  constellation/new_transformers/config_paper_stage1_200k.py

run_stage \
  paper_joint_stage2_200k \
  constellation/new_transformers/config_paper_stage2_200k.py \
  --load-model-from work_dirs/paper_joint_stage1_200k/checkpoints/iter_200000/model.pth

run_stage \
  paper_joint_stage3_200k \
  constellation/new_transformers/config_paper_stage3_200k.py \
  --load-model-from work_dirs/paper_joint_stage2_200k/checkpoints/iter_200000/model.pth
