#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_m2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_heads_m2_10k_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
CHECKPOINT="${ROOT_DIR}/work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_heads_m2.py"
RUN_NAME="event_heads_m2_10k"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] Stage3 checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] M2 config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/work_dirs/eval_logs"
exec auto_torchrun \
  -m constellation.new_transformers.train \
  "${RUN_NAME}" \
  "${CONFIG}" \
  --auto-resume \
  --load-model-from "${CHECKPOINT}"
