#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_large_test_once
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=120
#SBATCH --mem=220G
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_large_test_once_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_large_sync_ppo.py"
BASE_OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo"
OUTPUT="${BASE_OUTPUT}/test_once"
ALL_SCENE_IDS=($(seq 0 63))
SHARD_STARTS=(0 16 32 48)
SHARD_ENDS=(15 31 47 63)
: "${SELECTION_JSON:?SELECTION_JSON must point to the locked heldout selection}"
: "${FULL_VAL_RESULT:?FULL_VAL_RESULT must point to the passed full Val result}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

"${PYTHON}" - "${FULL_VAL_RESULT}" <<'PY'
import json
import pathlib
import sys

full_val = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert full_val['passed'] is True
PY
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
  echo "[error] locked Test checkpoint is missing" >&2
  exit 1
fi
if [[ -f "${OUTPUT}/summary.json" ]]; then
  echo "[error] the one allowed Test evaluation already exists" >&2
  exit 1
fi

mkdir -p "${OUTPUT}/shards"
pids=()
inputs=()
for shard_index in 0 1 2 3; do
  start="${SHARD_STARTS[$shard_index]}"
  end="${SHARD_ENDS[$shard_index]}"
  scene_ids=($(seq "${start}" "${end}"))
  shard_output="${OUTPUT}/shards/shard_${shard_index}.json"
  inputs+=("${shard_output}")
  CUDA_VISIBLE_DEVICES="${shard_index}" \
    "${PYTHON}" tools/evaluate_event_v2_policy.py \
      --config "${CONFIG}" \
      --checkpoint "${CANDIDATE_CHECKPOINT}" \
      --label v2_2_large_test \
      --split test \
      --scene-ids "${scene_ids[@]}" \
      --max-time-step 3600 \
      --device cuda:0 \
      --output "${shard_output}" \
      >"${OUTPUT}/shards/shard_${shard_index}.log" 2>&1 &
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

"${PYTHON}" tools/merge_event_v2_eval_summaries.py \
  --inputs "${inputs[@]}" \
  --expected-scene-ids "${ALL_SCENE_IDS[@]}" \
  --output "${OUTPUT}/summary.json"
