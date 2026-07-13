#!/usr/bin/env bash
#SBATCH --job-name=aeos_assign_p0
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/assignment_head_p0_train_slurm.log

set -euo pipefail

cd /home/hy/data/newaeosbench
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="/home/hy/data/newaeosbench:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/aeos_cache

run_name="assignment_head_p0_c020_cov010_10k"
baseline="work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth"

auto_torchrun -m constellation.new_transformers.train \
  "${run_name}" \
  constellation/new_transformers/config_assignment_head_p0.py \
  --auto-resume \
  --load-model-from "${baseline}"
