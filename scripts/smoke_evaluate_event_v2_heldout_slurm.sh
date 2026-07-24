#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_eval_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_heldout_smoke_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo_full.py"
OUTPUT_ROOT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_heldout/smoke_${SLURM_JOB_ID:-manual}"
BASELINE="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/checkpoint_update_000101.pth"
CANDIDATE="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for checkpoint in "${BASELINE}" "${CANDIDATE}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] checkpoint not found: ${checkpoint}" >&2
    exit 1
  fi
done

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
  echo "[error] held-out smoke needs one physically free GPU" >&2
  exit 1
fi
GPU_INDEX="${free_gpu_indices[0]}"

mkdir -p "${OUTPUT_ROOT}/v2_1" "${OUTPUT_ROOT}/v2_2_replica_0"
pids=()
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
  "${PYTHON}" tools/evaluate_event_v2_policy.py \
    --config "${CONFIG}" \
    --checkpoint "${BASELINE}" \
    --label v2_1 \
    --split train \
    --scene-ids 196 \
    --max-time-step 3600 \
    --device cuda \
    --output "${OUTPUT_ROOT}/v2_1" \
    >"${OUTPUT_ROOT}/v2_1/evaluate.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
  "${PYTHON}" tools/evaluate_event_v2_policy.py \
    --config "${CONFIG}" \
    --checkpoint "${CANDIDATE}" \
    --label v2_2_replica_0 \
    --split train \
    --scene-ids 196 \
    --max-time-step 3600 \
    --device cuda \
    --output "${OUTPUT_ROOT}/v2_2_replica_0" \
    >"${OUTPUT_ROOT}/v2_2_replica_0/evaluate.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
