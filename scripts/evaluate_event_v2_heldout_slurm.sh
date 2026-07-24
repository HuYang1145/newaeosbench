#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_heldout
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_heldout_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo_full.py"
OUTPUT_ROOT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_heldout/heldout_${SLURM_JOB_ID:-manual}"
SCENE_IDS=($(seq 196 203))
LABELS=(
  "v2_1"
  "v2_2_replica_0"
  "v2_2_replica_1"
  "v2_2_replica_2"
  "v2_2_replica_3"
)
CHECKPOINTS=(
  "${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/checkpoint_update_000101.pth"
  "${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"
  "${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_1/checkpoint_update_000950.pth"
  "${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_2/checkpoint_update_000924.pth"
  "${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_3/checkpoint_update_000914.pth"
)
GPU_ASSIGNMENTS=(0 1 2 0 1)

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] checkpoint not found: ${checkpoint}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"
pids=()
for index in 0 1 2 3 4; do
  label="${LABELS[$index]}"
  checkpoint="${CHECKPOINTS[$index]}"
  gpu_index="${GPU_ASSIGNMENTS[$index]}"
  candidate_output="${OUTPUT_ROOT}/${label}"
  mkdir -p "${candidate_output}"
  CUDA_VISIBLE_DEVICES="${gpu_index}" \
    "${PYTHON}" tools/evaluate_event_v2_policy.py \
      --config "${CONFIG}" \
      --checkpoint "${checkpoint}" \
      --label "${label}" \
      --split train \
      --scene-ids "${SCENE_IDS[@]}" \
      --max-time-step 3600 \
      --device cuda \
      --output "${candidate_output}" \
      >"${candidate_output}/evaluate.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  exit "${status}"
fi

"${PYTHON}" tools/select_event_v2_heldout.py \
  --baseline "${OUTPUT_ROOT}/v2_1/summary.json" \
  --candidates \
    "${OUTPUT_ROOT}/v2_2_replica_0/summary.json" \
    "${OUTPUT_ROOT}/v2_2_replica_1/summary.json" \
    "${OUTPUT_ROOT}/v2_2_replica_2/summary.json" \
    "${OUTPUT_ROOT}/v2_2_replica_3/summary.json" \
  --expected-scene-ids "${SCENE_IDS[@]}" \
  --output "${OUTPUT_ROOT}/selection.json"
