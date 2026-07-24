#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_selected
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_selected_smoke_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo_full.py"
SELECTION="${SELECTION:-${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_heldout/heldout_2212/selection.json}"
OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_selected_smoke/smoke_${SLURM_JOB_ID:-manual}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "${SELECTION}" ]]; then
  echo "[error] held-out selection not found: ${SELECTION}" >&2
  exit 1
fi
SELECTED_CHECKPOINT=$(
  "${PYTHON}" - "${SELECTION}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))['selected']['checkpoint'])
PY
)
if [[ ! -f "${SELECTED_CHECKPOINT}" ]]; then
  echo "[error] selected checkpoint not found: ${SELECTED_CHECKPOINT}" >&2
  exit 1
fi

free_gpu_indices=()
while IFS=',' read -r gpu_index memory_used; do
  gpu_index="${gpu_index// /}"
  memory_used="${memory_used// /}"
  if (( memory_used < 4096 )); then
    free_gpu_indices+=("${gpu_index}")
  fi
done < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
)
if (( ${#free_gpu_indices[@]} < 1 )); then
  echo "[error] selected smoke needs one physically free GPU" >&2
  exit 1
fi
GPU_INDEX="${free_gpu_indices[0]}"

mkdir -p "${OUTPUT}"
exec env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
  "${PYTHON}" tools/evaluate_event_v2_policy.py \
    --config "${CONFIG}" \
    --checkpoint "${SELECTED_CHECKPOINT}" \
    --label v2_2_selected \
    --split train \
    --scene-ids 204 \
    --max-time-step 3600 \
    --device cuda \
    --output "${OUTPUT}"
