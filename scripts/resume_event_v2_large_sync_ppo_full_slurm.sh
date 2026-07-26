#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_large_sync_resume
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=72
#SBATCH --mem=70G
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_large_sync_resume_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_large_sync_ppo.py"
BOOTSTRAP="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"
BASE_OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo"
OUTPUT_A="${BASE_OUTPUT}/seed_5408"
OUTPUT_B="${BASE_OUTPUT}/seed_5409"
LATEST_A="${BASE_OUTPUT}/seed_5408/checkpoint_latest.pth"
LATEST_B="${BASE_OUTPUT}/seed_5409/checkpoint_latest.pth"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#ALLOCATED_GPUS[@]} < 2 )); then
  echo "[error] large sync resume requires two allocated GPUs" >&2
  exit 1
fi
GPU_A="${ALLOCATED_GPUS[0]}"
GPU_B="${ALLOCATED_GPUS[1]}"

is_accepted() {
  local summary_path="$1"
  if [[ ! -f "${summary_path}" ]]; then
    return 1
  fi
  "${PYTHON}" - "${summary_path}" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
raise SystemExit(0 if summary.get('accepted') is True else 1)
PY
}

pids=()
labels=()
if ! is_accepted "${OUTPUT_A}/summary.json"; then
  resume_checkpoint="${LATEST_A}"
  if [[ ! -f "${resume_checkpoint}" ]]; then
    echo "[error] seed 5408 has no resumable checkpoint" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU_A}" \
    "${PYTHON}" tools/train_event_v2_large_sync_ppo.py \
      --config "${CONFIG}" \
      --bootstrap-checkpoint "${BOOTSTRAP}" \
      --seed 5408 \
      --learner-device cuda:0 \
      --actor-devices cuda:0 \
      --actors 12 \
      --active-environments 60 \
      --scene-start 205 \
      --scene-end 324 \
      --max-time-step 3600 \
      --max-updates 100000 \
      --checkpoint-every-updates 100 \
      --resume "${resume_checkpoint}" \
      --output-dir "${OUTPUT_A}" \
      >"${OUTPUT_A}/resume_${SLURM_JOB_ID:-manual}.log" 2>&1 &
  pids+=("$!")
  labels+=("5408")
fi

if ! is_accepted "${OUTPUT_B}/summary.json"; then
  resume_checkpoint="${LATEST_B}"
  if [[ ! -f "${resume_checkpoint}" ]]; then
    echo "[error] seed 5409 has no resumable checkpoint" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU_B}" \
    "${PYTHON}" tools/train_event_v2_large_sync_ppo.py \
      --config "${CONFIG}" \
      --bootstrap-checkpoint "${BOOTSTRAP}" \
      --seed 5409 \
      --learner-device cuda:0 \
      --actor-devices cuda:0 \
      --actors 12 \
      --active-environments 60 \
      --scene-start 205 \
      --scene-end 324 \
      --max-time-step 3600 \
      --max-updates 100000 \
      --checkpoint-every-updates 100 \
      --resume "${resume_checkpoint}" \
      --output-dir "${OUTPUT_B}" \
      >"${OUTPUT_B}/resume_${SLURM_JOB_ID:-manual}.log" 2>&1 &
  pids+=("$!")
  labels+=("5409")
fi

status=0
for index in "${!pids[@]}"; do
  pid="${pids[$index]}"
  if ! wait "${pid}"; then
    echo "[error] seed ${labels[$index]} resume failed" >&2
    status=1
  fi
done
if (( status != 0 )); then
  exit "${status}"
fi

if ! is_accepted "${OUTPUT_A}/summary.json" || ! is_accepted "${OUTPUT_B}/summary.json"; then
  exit 75
fi
