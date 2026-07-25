#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_large_gate
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_large_gate_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
BASE_CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo_full.py"
LARGE_CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_large_sync_ppo.py"
BASE_CHECKPOINT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"
BASE_OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo"
OUTPUT="${BASE_OUTPUT}/val_gate_${SLURM_JOB_ID:-manual}"
SCENE_IDS=($(seq 8 15))
: "${SELECTION_JSON:?SELECTION_JSON must point to the locked heldout selection}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

if [[ ! -f "${BASE_CHECKPOINT}" || ! -f "${SELECTION_JSON}" ]]; then
  echo "[error] baseline checkpoint or heldout selection is missing" >&2
  exit 1
fi
CANDIDATE_CHECKPOINT=$(
  "${PYTHON}" - "${SELECTION_JSON}" <<'PY'
import json
import pathlib
import sys

selection = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(selection['selected']['checkpoint'])
PY
)
if [[ ! -f "${CANDIDATE_CHECKPOINT}" ]]; then
  echo "[error] locked candidate checkpoint is missing" >&2
  exit 1
fi

SPLITS=("val_seen" "val_seen" "val_unseen" "val_unseen")
LABELS=(
  "v2_2_seen"
  "v2_2_large_seen"
  "v2_2_unseen"
  "v2_2_large_unseen"
)
CONFIGS=(
  "${BASE_CONFIG}"
  "${LARGE_CONFIG}"
  "${BASE_CONFIG}"
  "${LARGE_CONFIG}"
)
CHECKPOINTS=(
  "${BASE_CHECKPOINT}"
  "${CANDIDATE_CHECKPOINT}"
  "${BASE_CHECKPOINT}"
  "${CANDIDATE_CHECKPOINT}"
)

mkdir -p "${OUTPUT}"
pids=()
for index in 0 1 2 3; do
  split="${SPLITS[$index]}"
  label="${LABELS[$index]}"
  CUDA_VISIBLE_DEVICES="${index}" \
    "${PYTHON}" tools/evaluate_event_v2_policy.py \
      --config "${CONFIGS[$index]}" \
      --checkpoint "${CHECKPOINTS[$index]}" \
      --label "${label}" \
      --split "${split}" \
      --scene-ids "${SCENE_IDS[@]}" \
      --max-time-step 3600 \
      --device cuda:0 \
      --output "${OUTPUT}/${label}.json" \
      >"${OUTPUT}/${label}.log" 2>&1 &
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
  --baseline-seen "${OUTPUT}/v2_2_seen.json" \
  --candidate-seen "${OUTPUT}/v2_2_large_seen.json" \
  --baseline-unseen "${OUTPUT}/v2_2_unseen.json" \
  --candidate-unseen "${OUTPUT}/v2_2_large_unseen.json" \
  --expected-scene-ids "${SCENE_IDS[@]}" \
  --minimum-q-improvement 0.005 \
  --baseline-stage V2-2 \
  --candidate-stage V2-2-Large \
  --output "${OUTPUT}/gate.json"
