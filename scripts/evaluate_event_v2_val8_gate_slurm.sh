#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_val8
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_val8_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo_full.py"
SELECTION="${SELECTION:-${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_heldout/heldout_2212/selection.json}"
BASELINE="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/checkpoint_update_000101.pth"
OUTPUT_ROOT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_val8/val_${SLURM_JOB_ID:-manual}"
SCENE_IDS=($(seq 0 7))

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "${SELECTION}" || ! -f "${BASELINE}" ]]; then
  echo "[error] selection or baseline checkpoint is missing" >&2
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
if (( ${#free_gpu_indices[@]} < 3 )); then
  echo "[error] Val 8+8 needs three physically free GPUs" >&2
  exit 1
fi

LABELS=(
  "v2_1_val_seen"
  "v2_2_val_seen"
  "v2_1_val_unseen"
  "v2_2_val_unseen"
)
SPLITS=("val_seen" "val_seen" "val_unseen" "val_unseen")
CHECKPOINTS=(
  "${BASELINE}"
  "${SELECTED_CHECKPOINT}"
  "${BASELINE}"
  "${SELECTED_CHECKPOINT}"
)
GPU_ASSIGNMENTS=(
  "${free_gpu_indices[0]}"
  "${free_gpu_indices[1]}"
  "${free_gpu_indices[2]}"
  "${free_gpu_indices[0]}"
)

mkdir -p "${OUTPUT_ROOT}"
pids=()
for index in 0 1 2 3; do
  label="${LABELS[$index]}"
  split="${SPLITS[$index]}"
  checkpoint="${CHECKPOINTS[$index]}"
  gpu_index="${GPU_ASSIGNMENTS[$index]}"
  output="${OUTPUT_ROOT}/${label}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu_index}" \
    "${PYTHON}" tools/evaluate_event_v2_policy.py \
      --config "${CONFIG}" \
      --checkpoint "${checkpoint}" \
      --label "${label}" \
      --split "${split}" \
      --scene-ids "${SCENE_IDS[@]}" \
      --max-time-step 3600 \
      --device cuda \
      --output "${output}" \
      >"${output}/evaluate.log" 2>&1 &
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

"${PYTHON}" tools/compare_event_v2_val_gate.py \
  --baseline-seen "${OUTPUT_ROOT}/v2_1_val_seen/summary.json" \
  --candidate-seen "${OUTPUT_ROOT}/v2_2_val_seen/summary.json" \
  --baseline-unseen "${OUTPUT_ROOT}/v2_1_val_unseen/summary.json" \
  --candidate-unseen "${OUTPUT_ROOT}/v2_2_val_unseen/summary.json" \
  --expected-scene-ids "${SCENE_IDS[@]}" \
  --minimum-q-improvement 0.005 \
  --output "${OUTPUT_ROOT}/gate.json"
