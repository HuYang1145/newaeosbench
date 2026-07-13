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

max_scenes="${1:-2}"
world_size="${2:-${max_scenes}}"
continuation_bonus=0.25
checkpoint="work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth"
summary="work_dirs/eval_summaries/owner_assignment_p01_b025_val${max_scenes}.json"

mkdir -p work_dirs/eval_logs work_dirs/eval_summaries

for split in val_seen val_unseen; do
  run_name="owner_assignment_p01_b025_${split}_${max_scenes}"
  log_path="work_dirs/eval_logs/${run_name}.log"
  /home/hy/miniconda3/envs/aeos/bin/python \
    -m constellation.rl.eval_all \
    "${run_name}" \
    constellation/rl/config_eval.py \
    --override \
    "[\"environment\"][\"world_size\"]:${world_size}" \
    "[\"environment\"][\"split\"]:\"${split}\"" \
    --max-scenes "${max_scenes}" \
    --owner-assignment \
    --owner-continuation-bonus "${continuation_bonus}" \
    --coordination-diagnostics-top-k 5 \
    --load-model-from "${checkpoint}" \
    2>&1 | tee -a "${log_path}"
done

/home/hy/miniconda3/envs/aeos/bin/python tools/summarize_eval.py \
  --output "${summary}" \
  "work_dirs/rl_eval_owner_assignment_p01_b025_val_seen_${max_scenes}/val_seen" \
  "work_dirs/rl_eval_owner_assignment_p01_b025_val_unseen_${max_scenes}/val_unseen"
