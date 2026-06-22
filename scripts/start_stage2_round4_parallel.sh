#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AEOS_ENV_BIN="${AEOS_ENV_BIN:-/home/hy/miniconda3/envs/aeos/bin}"
export PATH="${AEOS_ENV_BIN}:${PATH}"

OLD_MODEL="${OLD_MODEL:-work_dirs/table3_base/checkpoints/iter_200000/model.pth}"
RUN_NAME="${RUN_NAME:-stage4_parallel_from_old_200k}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
CANDIDATE_EPOCH="${CANDIDATE_EPOCH:-4}"
TAU_E="${TAU_E:-4.5}"
MODEL_DEVICE="${MODEL_DEVICE:-0}"
ROLLOUT_DEVICE="${ROLLOUT_DEVICE:-cpu}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-}"
ROLLOUT_NICE="${ROLLOUT_NICE:-19}"
STOP_AFTER_ROLLOUT="${STOP_AFTER_ROLLOUT:-0}"

BASE_ANNOTATION="data/annotations/${TRAIN_SPLIT}.json"
BACKUP_ANNOTATION="data/annotations/${TRAIN_SPLIT}.before_${RUN_NAME}.json"
CANDIDATE_ROOT="data/trajectories.${CANDIDATE_EPOCH}"
ROUND_ANNOTATION="data/annotations/${TRAIN_SPLIT}_${RUN_NAME}.json"
ROUND_SUMMARY="work_dirs/${RUN_NAME}/${RUN_NAME}_summary.json"
ROLLOUT_LOG_DIR="work_dirs/${RUN_NAME}/rollout_logs"

mkdir -p "work_dirs/${RUN_NAME}" "${ROLLOUT_LOG_DIR}"

if [[ ! -f "${OLD_MODEL}" ]]; then
  echo "[error] old model not found: ${OLD_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${BACKUP_ANNOTATION}" ]]; then
  cp "${BASE_ANNOTATION}" "${BACKUP_ANNOTATION}"
fi

if [[ -z "${ROLLOUT_WORKERS}" ]]; then
  CPU_TOTAL="$(nproc)"
  if (( CPU_TOTAL > 1 )); then
    ROLLOUT_WORKERS="${CPU_TOTAL}"
  else
    ROLLOUT_WORKERS=1
  fi
fi

echo "[info] old model: ${OLD_MODEL}"
echo "[info] candidate trajectory root: ${CANDIDATE_ROOT}"
echo "[info] tau_e: ${TAU_E}"
echo "[info] rollout workers: ${ROLLOUT_WORKERS}"
echo "[info] rollout device: ${ROLLOUT_DEVICE}"
echo "[info] rollout nice: ${ROLLOUT_NICE}"
echo "[info] stop after rollout: ${STOP_AFTER_ROLLOUT}"

ROLLOUT_ARGS=()
if [[ -n "${ROLLOUT_LIMIT}" ]]; then
  ROLLOUT_ARGS+=(--limit "${ROLLOUT_LIMIT}")
fi

echo "[1/3] parallel rollout to ${CANDIDATE_ROOT}"
PIDS=()
for ((RANK=0; RANK<ROLLOUT_WORKERS; RANK++)); do
  LOG_PATH="${ROLLOUT_LOG_DIR}/rank_${RANK}.log"
  (
    echo "[start] $(date -Is) rank=${RANK} world_size=${ROLLOUT_WORKERS} device=${ROLLOUT_DEVICE}"
    set +e
    env \
      WORLD_SIZE="${ROLLOUT_WORKERS}" \
      RANK="${RANK}" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      PYTHONPATH=":${PYTHONPATH:-}" \
      ionice -c3 nice -n "${ROLLOUT_NICE}" \
      stdbuf -oL -eL \
      python3 tools/rollout_model_trajectories.py \
        "${OLD_MODEL}" \
        "${CANDIDATE_ROOT}" \
        --split "${TRAIN_SPLIT}" \
        --annotation-file "${BASE_ANNOTATION}" \
        --device "${ROLLOUT_DEVICE}" \
        "${ROLLOUT_ARGS[@]}"
    STATUS=$?
    set -e
    if (( STATUS != 0 )); then
      echo "[error] $(date -Is) rank=${RANK} exit_code=${STATUS}"
      exit "${STATUS}"
    fi
    echo "[done] $(date -Is) rank=${RANK} exit_code=0"
  ) > "${LOG_PATH}" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

if (( FAIL != 0 )); then
  echo "[error] at least one rollout worker failed; inspect ${ROLLOUT_LOG_DIR}" >&2
  exit 1
fi

if [[ "${STOP_AFTER_ROLLOUT}" == "1" ]]; then
  echo "[done] rollout finished and preserved for later filtering/training"
  echo "[done] rollout logs: ${ROLLOUT_LOG_DIR}"
  exit 0
fi

echo "[2/3] build stage-2 annotation with tau_e filter"
python3 tools/build_tau_e_annotation.py \
  "${BASE_ANNOTATION}" \
  "${CANDIDATE_ROOT}" \
  "${ROUND_ANNOTATION}" \
  --split "${TRAIN_SPLIT}" \
  --candidate-epoch "${CANDIDATE_EPOCH}" \
  --tau-e "${TAU_E}" \
  --summary-path "${ROUND_SUMMARY}"

cp "${ROUND_ANNOTATION}" "${BASE_ANNOTATION}"

echo "[3/3] continue training from old model on updated annotation"
CUDA_VISIBLE_DEVICES="${MODEL_DEVICE}" \
PYTHONPATH=":${PYTHONPATH:-}" \
auto_torchrun -m constellation.new_transformers.train \
  "${RUN_NAME}" \
  constellation/new_transformers/config.py \
  --load-model-from "${OLD_MODEL}"

echo "[done] new annotation: ${BASE_ANNOTATION}"
echo "[done] summary: ${ROUND_SUMMARY}"
echo "[done] rollout logs: ${ROLLOUT_LOG_DIR}"
echo "[done] new model: work_dirs/${RUN_NAME}/checkpoints/iter_200000/model.pth"
