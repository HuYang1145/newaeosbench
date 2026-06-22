#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AEOS_ENV_BIN="${AEOS_ENV_BIN:-/home/hy/miniconda3/envs/aeos/bin}"
export PATH="${AEOS_ENV_BIN}:${PATH}"

OLD_MODEL="${OLD_MODEL:-work_dirs/table3_base/checkpoints/iter_200000/model.pth}"
RUN_NAME="${RUN_NAME:-stage4_from_old_200k}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
CANDIDATE_EPOCH="${CANDIDATE_EPOCH:-4}"
TAU_E="${TAU_E:-4.5}"
MODEL_DEVICE="${MODEL_DEVICE:-0}"
ROLLOUT_DEVICE="${ROLLOUT_DEVICE:-cuda:0}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-}"

BASE_ANNOTATION="data/annotations/${TRAIN_SPLIT}.json"
BACKUP_ANNOTATION="data/annotations/${TRAIN_SPLIT}.before_${RUN_NAME}.json"
CANDIDATE_ROOT="data/trajectories.${CANDIDATE_EPOCH}"
ROUND_ANNOTATION="data/annotations/${TRAIN_SPLIT}_${RUN_NAME}.json"
ROUND_SUMMARY="work_dirs/${RUN_NAME}/${RUN_NAME}_summary.json"

mkdir -p "work_dirs/${RUN_NAME}"

if [[ ! -f "${OLD_MODEL}" ]]; then
  echo "[error] old model not found: ${OLD_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${BACKUP_ANNOTATION}" ]]; then
  cp "${BASE_ANNOTATION}" "${BACKUP_ANNOTATION}"
fi

echo "[info] old model: ${OLD_MODEL}"
echo "[info] candidate trajectory root: ${CANDIDATE_ROOT}"
echo "[info] tau_e: ${TAU_E}"
echo "[info] python: $(command -v python3)"
echo "[info] auto_torchrun: $(command -v auto_torchrun)"

ROLLOUT_ARGS=()
if [[ -n "${ROLLOUT_LIMIT}" ]]; then
  ROLLOUT_ARGS+=(--limit "${ROLLOUT_LIMIT}")
fi

echo "[1/3] rollout old model to ${CANDIDATE_ROOT}"
WORLD_SIZE="${WORLD_SIZE:-1}" RANK="${RANK:-0}" PYTHONPATH=":${PYTHONPATH:-}" \
  python3 tools/rollout_model_trajectories.py \
  "${OLD_MODEL}" \
  "${CANDIDATE_ROOT}" \
  --split "${TRAIN_SPLIT}" \
  --annotation-file "${BASE_ANNOTATION}" \
  --device "${ROLLOUT_DEVICE}" \
  "${ROLLOUT_ARGS[@]}"

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
CUDA_VISIBLE_DEVICES="${MODEL_DEVICE}" PYTHONPATH=":${PYTHONPATH:-}" \
  auto_torchrun -m constellation.new_transformers.train \
  "${RUN_NAME}" \
  constellation/new_transformers/config.py \
  --load-model-from "${OLD_MODEL}"

echo "[done] new annotation: ${BASE_ANNOTATION}"
echo "[done] summary: ${ROUND_SUMMARY}"
echo "[done] new model: work_dirs/${RUN_NAME}/checkpoints/iter_200000/model.pth"
