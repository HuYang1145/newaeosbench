#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_warm_start_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_warm_start.py"
CHECKPOINT="${ROOT_DIR}/work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth"
OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_0_warm_start"
PREFLIGHT_OUTPUT="${OUTPUT}/preflight_${SLURM_JOB_ID:-manual}"

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
  echo "[error] V2-0 config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT}" "${PREFLIGHT_OUTPUT}"

"${PYTHON}" tools/train_event_v2_warm_start.py \
  --config "${CONFIG}" \
  --stage3-checkpoint "${CHECKPOINT}" \
  --output "${PREFLIGHT_OUTPUT}" \
  --max-steps 1 \
  --device cuda

exec "${PYTHON}" tools/train_event_v2_warm_start.py \
  --config "${CONFIG}" \
  --stage3-checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --device cuda
