#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_2_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_2_smoke_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo_full.py"
BOOTSTRAP="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/checkpoint_update_000101.pth"
OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/smoke_${SLURM_JOB_ID:-manual}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "${BOOTSTRAP}" ]]; then
  echo "[error] V2-1 bootstrap checkpoint not found: ${BOOTSTRAP}" >&2
  exit 1
fi

mkdir -p "${OUTPUT}"

exec "${PYTHON}" tools/train_event_v2_sync_ppo.py \
  --config "${CONFIG}" \
  --bootstrap-checkpoint "${BOOTSTRAP}" \
  --device cuda \
  --seed 4407 \
  --scene-ids 4 \
  --max-time-step 120 \
  --max-updates 1 \
  --ppo-epochs 1 \
  --output "${OUTPUT}"
