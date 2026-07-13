#!/usr/bin/env bash
#SBATCH --job-name=aeos_assign_eval8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/assignment_head_p0_eval8_slurm.log

set -euo pipefail

cd /home/hy/data/newaeosbench
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="/home/hy/data/newaeosbench:${PYTHONPATH:-}"
export WORLD_SIZE=1
export RANK=0
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/aeos_cache

checkpoint="work_dirs/assignment_head_p0_c020_cov010_10k/checkpoints/iter_10000/model.pth"
max_scenes=8
world_size=8

for split in val_seen val_unseen; do
  run_name="assignment_head_p0_10k_${split}_8"
  /home/hy/miniconda3/envs/aeos/bin/python \
    -m constellation.rl.eval_all \
    "${run_name}" \
    constellation/rl/config_eval.py \
    --override \
    "[\"environment\"][\"world_size\"]:${world_size}" \
    "[\"environment\"][\"split\"]:\"${split}\"" \
    --max-scenes "${max_scenes}" \
    --use-assignment-head \
    --assignment-head-hidden-width 32 \
    --coordination-diagnostics-top-k 5 \
    --load-model-from "${checkpoint}"
done

/home/hy/miniconda3/envs/aeos/bin/python tools/summarize_eval.py \
  --output work_dirs/eval_summaries/assignment_head_p0_10k_val8.json \
  work_dirs/rl_eval_assignment_head_p0_10k_val_seen_8/val_seen \
  work_dirs/rl_eval_assignment_head_p0_10k_val_unseen_8/val_unseen
