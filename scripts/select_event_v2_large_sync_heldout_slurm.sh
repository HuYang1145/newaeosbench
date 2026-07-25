#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_large_heldout
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_large_heldout_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
BASE_CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_sync_ppo_full.py"
LARGE_CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_large_sync_ppo.py"
BASE_CHECKPOINT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"
BASE_OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo"
OUTPUT="${BASE_OUTPUT}/heldout_${SLURM_JOB_ID:-manual}"
BEST_LINK="${BASE_OUTPUT}/checkpoint_best.pth"
SCENE_IDS=($(seq 196 203))

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "[error] V2-2 baseline checkpoint is missing" >&2
  exit 1
fi
for seed in 5408 5409; do
  "${PYTHON}" - "${BASE_OUTPUT}/seed_${seed}/summary.json" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
if not summary_path.is_file():
    raise SystemExit(f'missing training summary: {summary_path}')
summary = json.loads(summary_path.read_text())
assert summary['accepted'] is True
PY
done

mkdir -p "${OUTPUT}/baseline" "${OUTPUT}/candidates" "${OUTPUT}/logs"
CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON}" tools/evaluate_event_v2_policy.py \
    --config "${BASE_CONFIG}" \
    --checkpoint "${BASE_CHECKPOINT}" \
    --label v2_2_replica_0 \
    --split train \
    --scene-ids "${SCENE_IDS[@]}" \
    --max-time-step 3600 \
    --device cuda:0 \
    --output "${OUTPUT}/baseline/summary.json" \
    >"${OUTPUT}/logs/baseline.log" 2>&1

readarray -t CHECKPOINTS < <(
  find "${BASE_OUTPUT}/seed_5408" "${BASE_OUTPUT}/seed_5409" \
    -maxdepth 1 -type f \
    \( -name 'checkpoint_update_*.pth' \
       -o -name 'checkpoint_final_update_*.pth' \) \
    | sort
)
if (( ${#CHECKPOINTS[@]} < 2 )); then
  echo "[error] heldout selection needs at least two permanent checkpoints" >&2
  exit 1
fi

candidate_summaries=()
pids=()

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

for index in "${!CHECKPOINTS[@]}"; do
  checkpoint="${CHECKPOINTS[$index]}"
  seed_directory="$(basename "$(dirname "${checkpoint}")")"
  seed="${seed_directory#seed_}"
  checkpoint_stem="$(basename "${checkpoint}" .pth)"
  label="seed_${seed}_${checkpoint_stem#checkpoint_}"
  candidate_dir="${OUTPUT}/candidates/${label}"
  candidate_summary="${candidate_dir}/summary.json"
  gpu_index=$(( index % 4 ))
  mkdir -p "${candidate_dir}"
  candidate_summaries+=("${candidate_summary}")

  CUDA_VISIBLE_DEVICES="${gpu_index}" \
    "${PYTHON}" tools/evaluate_event_v2_policy.py \
      --config "${LARGE_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --label "${label}" \
      --split train \
      --scene-ids "${SCENE_IDS[@]}" \
      --max-time-step 3600 \
      --device cuda:0 \
      --output "${candidate_summary}" \
      >"${OUTPUT}/logs/${label}.log" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} == 4 )); then
    wait_batch
  fi
done
if (( ${#pids[@]} > 0 )); then
  wait_batch
fi

"${PYTHON}" tools/select_event_v2_large_sync_heldout.py \
  --baseline "${OUTPUT}/baseline/summary.json" \
  --candidates "${candidate_summaries[@]}" \
  --expected-scene-ids "${SCENE_IDS[@]}" \
  --output "${OUTPUT}/selection.json" \
  --best-link "${BEST_LINK}"
