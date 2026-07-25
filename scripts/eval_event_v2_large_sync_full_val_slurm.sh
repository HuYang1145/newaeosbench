#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_large_full_val
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=120
#SBATCH --mem=220G
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_large_full_val_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
BASE_CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo_full.py"
LARGE_CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_large_sync_ppo.py"
BASE_CHECKPOINT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"
BASE_OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo"
OUTPUT="${BASE_OUTPUT}/full_val_${SLURM_JOB_ID:-manual}"
ALL_SCENE_IDS=($(seq 0 63))
GROUP_STARTS=(0 8 16)
GROUP_ENDS=(7 15 63)
GROUP_NAMES=("history_0_7" "gate_8_15" "rest_16_63")
: "${SELECTION_JSON:?SELECTION_JSON must point to the locked heldout selection}"
: "${GATE_JSON:?GATE_JSON must point to the passed new Val 8+8 gate}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

"${PYTHON}" - "${GATE_JSON}" <<'PY'
import json
import pathlib
import sys

gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert gate['passed'] is True
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
if [[ ! -f "${BASE_CHECKPOINT}" || ! -f "${CANDIDATE_CHECKPOINT}" ]]; then
  echo "[error] full Val checkpoint is missing" >&2
  exit 1
fi

MODEL_NAMES=("baseline" "candidate")
MODEL_LABELS=("v2_2" "v2_2_large")
MODEL_CONFIGS=("${BASE_CONFIG}" "${LARGE_CONFIG}")
MODEL_CHECKPOINTS=("${BASE_CHECKPOINT}" "${CANDIDATE_CHECKPOINT}")
SPLITS=("val_seen" "val_unseen")
mkdir -p "${OUTPUT}/shards" "${OUTPUT}/merged"

pids=()
task_index=0
wait_batch() {
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  pids=()
  if (( status != 0 )); then
    return 1
  fi
}

for model_index in 0 1; do
  for split in "${SPLITS[@]}"; do
    for group_index in 0 1 2; do
      start="${GROUP_STARTS[$group_index]}"
      end="${GROUP_ENDS[$group_index]}"
      group="${GROUP_NAMES[$group_index]}"
      scene_ids=($(seq "${start}" "${end}"))
      label="${MODEL_LABELS[$model_index]}_${split}_${group}"
      output_path="${OUTPUT}/shards/${label}.json"
      gpu_index=$(( task_index % 4 ))
      CUDA_VISIBLE_DEVICES="${gpu_index}" \
        "${PYTHON}" tools/evaluate_event_v2_policy.py \
          --config "${MODEL_CONFIGS[$model_index]}" \
          --checkpoint "${MODEL_CHECKPOINTS[$model_index]}" \
          --label "${MODEL_LABELS[$model_index]}" \
          --split "${split}" \
          --scene-ids "${scene_ids[@]}" \
          --max-time-step 3600 \
          --device cuda:0 \
          --output "${output_path}" \
          >"${OUTPUT}/shards/${label}.log" 2>&1 &
      pids+=("$!")
      task_index=$(( task_index + 1 ))
      if (( ${#pids[@]} == 4 )); then
        wait_batch
      fi
    done
  done
done
if (( ${#pids[@]} > 0 )); then
  wait_batch
fi

for model_index in 0 1; do
  for split in "${SPLITS[@]}"; do
    inputs=()
    for group in "${GROUP_NAMES[@]}"; do
      inputs+=(
        "${OUTPUT}/shards/${MODEL_LABELS[$model_index]}_${split}_${group}.json"
      )
    done
    "${PYTHON}" tools/merge_event_v2_eval_summaries.py \
      --inputs "${inputs[@]}" \
      --expected-scene-ids "${ALL_SCENE_IDS[@]}" \
      --output "${OUTPUT}/merged/${MODEL_NAMES[$model_index]}_${split}.json"
  done
done

"${PYTHON}" - "${OUTPUT}/merged/candidate_val_seen.json" "${OUTPUT}/merged/candidate_val_unseen.json" <<'PY'
import json
import pathlib
import sys

for path in sys.argv[1:]:
    summary = json.loads(pathlib.Path(path).read_text())
    for metric in ('TAT_s', 'PC_Wh', 'CS_paper'):
        assert metric in summary['aggregate']
PY

"${PYTHON}" tools/compare_event_v2_full_val.py \
  --baseline-seen "${OUTPUT}/merged/baseline_val_seen.json" \
  --candidate-seen "${OUTPUT}/merged/candidate_val_seen.json" \
  --baseline-unseen "${OUTPUT}/merged/baseline_val_unseen.json" \
  --candidate-unseen "${OUTPUT}/merged/candidate_val_unseen.json" \
  --expected-scene-ids "${ALL_SCENE_IDS[@]}" \
  --minimum-q-improvement 0.005 \
  --baseline-stage V2-2 \
  --candidate-stage V2-2-Large \
  --output "${OUTPUT}/full_val_result.json"
