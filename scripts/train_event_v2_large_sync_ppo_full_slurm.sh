#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_large_sync_full
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=72
#SBATCH --mem=70G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_large_sync_full_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_large_sync_ppo.py"
BOOTSTRAP="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"
BASE_OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo"
SEEDS=(5408 5409)
: "${SMOKE_SUMMARY:?SMOKE_SUMMARY must point to an accepted large-sync smoke summary}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

if [[ ! -f "${BOOTSTRAP}" || ! -f "${SMOKE_SUMMARY}" ]]; then
  echo "[error] bootstrap or smoke summary is missing" >&2
  exit 1
fi
"${PYTHON}" - "${SMOKE_SUMMARY}" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert summary['accepted'] is True
PY

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#ALLOCATED_GPUS[@]} < 2 )); then
  echo "[error] large sync full training requires two allocated GPUs" >&2
  exit 1
fi
GPU_A="${ALLOCATED_GPUS[0]}"
GPU_B="${ALLOCATED_GPUS[1]}"

OUTPUT_A="${BASE_OUTPUT}/seed_${SEEDS[0]}"
OUTPUT_B="${BASE_OUTPUT}/seed_${SEEDS[1]}"
if [[ -e "${OUTPUT_A}/checkpoint_latest.pth" || -e "${OUTPUT_B}/checkpoint_latest.pth" ]]; then
  echo "[error] existing checkpoints require the dedicated resume script" >&2
  exit 1
fi
mkdir -p "${OUTPUT_A}" "${OUTPUT_B}"

pids=()
checkpoint_before_timeout() {
  trap - USR1
  echo "[info] time limit approaching; requesting barrier checkpoints" >&2
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -USR1 "${pid}"
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit 75
}
trap checkpoint_before_timeout USR1

CUDA_VISIBLE_DEVICES="${GPU_A}" \
  "${PYTHON}" tools/train_event_v2_large_sync_ppo.py \
    --config "${CONFIG}" \
    --bootstrap-checkpoint "${BOOTSTRAP}" \
    --seed "${SEEDS[0]}" \
    --learner-device cuda:0 \
    --actor-devices cuda:0 \
    --actors 12 \
    --active-environments 60 \
    --scene-start 205 \
    --scene-end 324 \
    --max-time-step 3600 \
    --max-updates 100000 \
    --checkpoint-every-updates 100 \
    --output-dir "${OUTPUT_A}" \
    >"${OUTPUT_A}/train_${SLURM_JOB_ID:-manual}.log" 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES="${GPU_B}" \
  "${PYTHON}" tools/train_event_v2_large_sync_ppo.py \
    --config "${CONFIG}" \
    --bootstrap-checkpoint "${BOOTSTRAP}" \
    --seed "${SEEDS[1]}" \
    --learner-device cuda:0 \
    --actor-devices cuda:0 \
    --actors 12 \
    --active-environments 60 \
    --scene-start 205 \
    --scene-end 324 \
    --max-time-step 3600 \
    --max-updates 100000 \
    --checkpoint-every-updates 100 \
    --output-dir "${OUTPUT_B}" \
    >"${OUTPUT_B}/train_${SLURM_JOB_ID:-manual}.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  exit "${status}"
fi

"${PYTHON}" - "${OUTPUT_A}/summary.json" "${OUTPUT_B}/summary.json" <<'PY'
import json
import pathlib
import sys

summaries = [
    json.loads(pathlib.Path(path).read_text())
    for path in sys.argv[1:]
]
if not all(summary['accepted'] is True for summary in summaries):
    raise SystemExit(75)
PY
