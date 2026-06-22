#!/usr/bin/env bash
set -euo pipefail

cd /home/hy/data/newaeosbench

export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH=":${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WORLD_SIZE=1
export RANK=0
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/aeos_cache

mkdir -p work_dirs/eval_logs work_dirs/eval_summaries

model_path="work_dirs/paper_joint_stage3_30k/checkpoints/iter_30000/model.pth"

for split in val_seen val_unseen test; do
  run_name="paper_joint_stage3_30k_96core_${split}_restored"
  log_path="work_dirs/eval_logs/${run_name}.log"
  echo "===== $(date '+%F %T') start ${split} =====" | tee -a "${log_path}"
  python -m constellation.rl.eval_all \
    "${run_name}" \
    constellation/rl/config_eval.py \
    --override '["environment"]["world_size"]:96' "[\"environment\"][\"split\"]:\"${split}\"" \
    --load-model-from "${model_path}" 2>&1 | tee -a "${log_path}"
  echo "===== $(date '+%F %T') done ${split} =====" | tee -a "${log_path}"
done

python tools/summarize_no_tat_eval.py \
  --output work_dirs/eval_summaries/paper_joint_stage3_30k_no_tat_96core_restored.json \
  work_dirs/rl_eval_paper_joint_stage3_30k_96core_val_seen_restored/val_seen \
  work_dirs/rl_eval_paper_joint_stage3_30k_96core_val_unseen_restored/val_unseen \
  work_dirs/rl_eval_paper_joint_stage3_30k_96core_test_restored/test \
  2>&1 | tee -a work_dirs/eval_logs/paper_joint_stage3_30k_96core_summary_restored.log
