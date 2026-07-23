#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_sync_ppo_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo.py"
WARM_START="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_0_warm_start/checkpoint_step_010000.pth"
OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_1_sync_ppo"
PREFLIGHT_OUTPUT="${OUTPUT}/synthetic_preflight_${SLURM_JOB_ID:-manual}"
REAL_PREFLIGHT_OUTPUT="${OUTPUT}/real_preflight_${SLURM_JOB_ID:-manual}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "${WARM_START}" ]]; then
  echo "[error] V2-0 checkpoint not found: ${WARM_START}" >&2
  exit 1
fi

mkdir -p "${OUTPUT}" "${PREFLIGHT_OUTPUT}" "${REAL_PREFLIGHT_OUTPUT}"

"${PYTHON}" tools/train_event_v2_sync_ppo.py \
  --config "${CONFIG}" \
  --synthetic-preflight \
  --device cpu \
  --max-updates 2 \
  --output "${PREFLIGHT_OUTPUT}"

"${PYTHON}" tools/train_event_v2_sync_ppo.py \
  --config "${CONFIG}" \
  --warm-start-checkpoint "${WARM_START}" \
  --device cuda \
  --scene-ids 0 \
  --max-time-step 60 \
  --max-updates 1 \
  --ppo-epochs 1 \
  --output "${REAL_PREFLIGHT_OUTPUT}"

exec "${PYTHON}" tools/train_event_v2_sync_ppo.py \
  --config "${CONFIG}" \
  --warm-start-checkpoint "${WARM_START}" \
  --device cuda \
  --max-time-step 3600 \
  --output "${OUTPUT}"
