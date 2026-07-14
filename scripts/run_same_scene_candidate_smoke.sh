#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/aeos/bin/python}"
CHECKPOINT="${CHECKPOINT:-work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-work_dirs/same_scene_candidates_stage3_200k_smoke}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-2}"
DEVICE="${DEVICE:-cpu}"
NUM_THREADS="${NUM_THREADS:-8}"
SCENE_WORKERS="${SCENE_WORKERS:-1}"
TOP_K="${TOP_K:-3}"
TEMPERATURE="${TEMPERATURE:-0.7}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/aeos-matplotlib}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] aeos python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"

candidates=(
  candidate_000_greedy
  candidate_001_topk_seed3408
  candidate_002_topk_seed3409
  candidate_003_topk_seed3410
)
strategies=(greedy top_k_sample top_k_sample top_k_sample)
seeds=(3407 3408 3409 3410)
pids=()
labels=()

echo "[info] checkpoint=${CHECKPOINT}"
echo "[info] output_root=${OUTPUT_ROOT} split=${SPLIT} limit=${LIMIT}"
echo "[info] device=${DEVICE} threads_per_worker=${NUM_THREADS} scene_workers=${SCENE_WORKERS}"

for index in "${!candidates[@]}"; do
  candidate="${candidates[$index]}"
  strategy="${strategies[$index]}"
  seed="${seeds[$index]}"
  for ((rank = 0; rank < SCENE_WORKERS; rank++)); do
    label="${candidate}_rank${rank}"
    log_path="${OUTPUT_ROOT}/logs/${label}.log"
    echo "[start] ${label} strategy=${strategy} seed=${seed} log=${log_path}"
    (
      export OMP_NUM_THREADS="${NUM_THREADS}"
      export MKL_NUM_THREADS="${NUM_THREADS}"
      export WORLD_SIZE="${SCENE_WORKERS}"
      export RANK="${rank}"
      nice -n 10 "${PYTHON_BIN}" tools/rollout_model_trajectories.py \
        "${CHECKPOINT}" \
        "${OUTPUT_ROOT}/${candidate}" \
        --split "${SPLIT}" \
        --limit "${LIMIT}" \
        --device "${DEVICE}" \
        --strategy "${strategy}" \
        --top-k "${TOP_K}" \
        --temperature "${TEMPERATURE}" \
        --seed "${seed}"
    ) >"${log_path}" 2>&1 &
    pids+=("$!")
    labels+=("${label}")
  done
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[error] ${labels[$index]} failed; inspect its log" >&2
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${PYTHON_BIN}" tools/summarize_same_scene_candidates.py \
  "${OUTPUT_ROOT}" \
  --split "${SPLIT}" \
  --greedy-candidate candidate_000_greedy \
  --output "${OUTPUT_ROOT}/summary.json"

echo "[done] summary=${OUTPUT_ROOT}/summary.json"
