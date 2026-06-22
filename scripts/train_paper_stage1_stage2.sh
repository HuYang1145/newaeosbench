#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AEOS_ENV_BIN="${AEOS_ENV_BIN:-/home/hy/miniconda3/envs/aeos/bin}"
export PATH="${AEOS_ENV_BIN}:${PATH}"

RUN_NAME="${RUN_NAME:-paper_joint}"
TIME_RUN_NAME="${TIME_RUN_NAME:-${RUN_NAME}_time}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${RUN_NAME}_stage1}"
STAGE2_ROUNDS="${STAGE2_ROUNDS:-1}"
TAU_E="${TAU_E:-4.5}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TIME_DEVICE="${TIME_DEVICE:-0}"
MODEL_DEVICE="${MODEL_DEVICE:-0}"
ROLLOUT_DEVICE="${ROLLOUT_DEVICE:-cuda:0}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-}"

BASE_ANNOTATION="data/annotations/${TRAIN_SPLIT}.json"
BACKUP_ANNOTATION="data/annotations/${TRAIN_SPLIT}.before_${RUN_NAME}.json"

if [[ ! -f "$BACKUP_ANNOTATION" ]]; then
  cp "$BASE_ANNOTATION" "$BACKUP_ANNOTATION"
fi

echo "[stage1] train time model"
CUDA_VISIBLE_DEVICES="${TIME_DEVICE}" PYTHONPATH=":${PYTHONPATH:-}" \
  auto_torchrun -m constellation.new_transformers.train \
  "${TIME_RUN_NAME}" \
  constellation/new_transformers/config_timemodel.py

TIME_CKPT="work_dirs/${TIME_RUN_NAME}/checkpoints/iter_50000/model.pth"
WRAPPED_TIME_CKPT="work_dirs/${TIME_RUN_NAME}/checkpoints/iter_50000/model_for_main.pth"

echo "[stage1] wrap time-model checkpoint for nested loading"
python3 tools/wrap_time_model_checkpoint.py \
  "$TIME_CKPT" \
  "$WRAPPED_TIME_CKPT"

echo "[stage1] train actor with pretrained time model"
CUDA_VISIBLE_DEVICES="${MODEL_DEVICE}" PYTHONPATH=":${PYTHONPATH:-}" \
  auto_torchrun -m constellation.new_transformers.train \
  "${STAGE1_RUN_NAME}" \
  constellation/new_transformers/config.py \
  --load-model-from "$WRAPPED_TIME_CKPT"

CURRENT_RUN_NAME="$STAGE1_RUN_NAME"
CURRENT_MODEL_CKPT="work_dirs/${CURRENT_RUN_NAME}/checkpoints/iter_200000/model.pth"
NEXT_EPOCH="$(find data -maxdepth 1 -type d -name 'trajectories.*' | sed 's#^.*/trajectories\.##' | sort -n | tail -n 1)"
if [[ -z "${NEXT_EPOCH}" ]]; then
  NEXT_EPOCH=1
fi
NEXT_EPOCH="$((NEXT_EPOCH + 1))"

for ROUND in $(seq 1 "${STAGE2_ROUNDS}"); do
  CANDIDATE_EPOCH="${NEXT_EPOCH}"
  CANDIDATE_ROOT="data/trajectories.${CANDIDATE_EPOCH}"
  ROUND_TAG="${RUN_NAME}_stage2_round${ROUND}"
  ROUND_ANNOTATION="data/annotations/${TRAIN_SPLIT}_${ROUND_TAG}.json"
  ROUND_SUMMARY="work_dirs/${RUN_NAME}/${ROUND_TAG}_summary.json"
  mkdir -p "work_dirs/${RUN_NAME}"

  echo "[stage2][round=${ROUND}] rollout current model into ${CANDIDATE_ROOT}"
  ROLLOUT_ARGS=()
  if [[ -n "${ROLLOUT_LIMIT}" ]]; then
    ROLLOUT_ARGS+=(--limit "${ROLLOUT_LIMIT}")
  fi
  WORLD_SIZE="${WORLD_SIZE:-1}" RANK="${RANK:-0}" PYTHONPATH=":${PYTHONPATH:-}" \
    python3 tools/rollout_model_trajectories.py \
    "${CURRENT_MODEL_CKPT}" \
    "${CANDIDATE_ROOT}" \
    --split "${TRAIN_SPLIT}" \
    --annotation-file "${BASE_ANNOTATION}" \
    --device "${ROLLOUT_DEVICE}" \
    "${ROLLOUT_ARGS[@]}"

  echo "[stage2][round=${ROUND}] build tau_e annotation"
  python3 tools/build_tau_e_annotation.py \
    "${BASE_ANNOTATION}" \
    "${CANDIDATE_ROOT}" \
    "${ROUND_ANNOTATION}" \
    --split "${TRAIN_SPLIT}" \
    --candidate-epoch "${CANDIDATE_EPOCH}" \
    --tau-e "${TAU_E}" \
    --summary-path "${ROUND_SUMMARY}"

  cp "${ROUND_ANNOTATION}" "${BASE_ANNOTATION}"

  CURRENT_RUN_NAME="${RUN_NAME}_stage2_round${ROUND}"
  echo "[stage2][round=${ROUND}] continue training from ${CURRENT_MODEL_CKPT}"
  CUDA_VISIBLE_DEVICES="${MODEL_DEVICE}" PYTHONPATH=":${PYTHONPATH:-}" \
    auto_torchrun -m constellation.new_transformers.train \
    "${CURRENT_RUN_NAME}" \
    constellation/new_transformers/config.py \
    --load-model-from "${CURRENT_MODEL_CKPT}" "${WRAPPED_TIME_CKPT}"

  CURRENT_MODEL_CKPT="work_dirs/${CURRENT_RUN_NAME}/checkpoints/iter_200000/model.pth"
  NEXT_EPOCH="$((NEXT_EPOCH + 1))"
done

echo "[done] latest model: ${CURRENT_MODEL_CKPT}"
echo "[done] active train annotation: ${BASE_ANNOTATION}"
