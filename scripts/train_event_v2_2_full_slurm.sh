#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_2_full
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=96
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_2_full_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo_full.py"
BOOTSTRAP="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/checkpoint_update_000101.pth"
OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo"
SEEDS=(4407 4408 4409 4410)
SHARDS=(
  "$(seq 4 51 | tr '\n' ' ')"
  "$(seq 52 99 | tr '\n' ' ')"
  "$(seq 100 147 | tr '\n' ' ')"
  "$(seq 148 195 | tr '\n' ' ')"
)

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
if (( ${#free_gpu_indices[@]} < 3 )); then
  echo "[error] V2-2 needs at least three physically free GPUs" >&2
  exit 1
fi
if (( ${#free_gpu_indices[@]} >= 4 )); then
  GPU_ASSIGNMENTS=(
    "${free_gpu_indices[0]}"
    "${free_gpu_indices[1]}"
    "${free_gpu_indices[2]}"
    "${free_gpu_indices[3]}"
  )
else
  GPU_ASSIGNMENTS=(
    "${free_gpu_indices[0]}"
    "${free_gpu_indices[1]}"
    "${free_gpu_indices[2]}"
    "${free_gpu_indices[0]}"
  )
fi
echo "[info] replica GPU assignments: ${GPU_ASSIGNMENTS[*]}"

mkdir -p "${OUTPUT}"
pids=()
for replica in 0 1 2 3; do
  replica_output="${OUTPUT}/replica_${replica}"
  replica_log="${replica_output}/train_${SLURM_JOB_ID:-manual}.log"
  gpu_index="${GPU_ASSIGNMENTS[$replica]}"
  mkdir -p "${replica_output}"
  read -r -a scene_ids <<< "${SHARDS[$replica]}"
  CUDA_VISIBLE_DEVICES="${gpu_index}" \
    "${PYTHON}" tools/train_event_v2_sync_ppo.py \
      --config "${CONFIG}" \
      --bootstrap-checkpoint "${BOOTSTRAP}" \
      --device cuda \
      --seed "${SEEDS[$replica]}" \
      --scene-ids "${scene_ids[@]}" \
      --max-time-step 3600 \
      --max-updates 1400 \
      --output "${replica_output}" \
      >"${replica_log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
