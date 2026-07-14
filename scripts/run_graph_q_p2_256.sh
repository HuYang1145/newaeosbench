#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/aeos/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-work_dirs/same_scene_candidates_stage3_200k_64}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-work_dirs/same_scene_candidates_stage3_200k_256}"
GRAPH_ROOT="${GRAPH_ROOT:-work_dirs/first_divergence_graph_q_256}"
DIVERGENCE_PATH="${DIVERGENCE_PATH:-${GRAPH_ROOT}/first_divergence_preferences.json}"
LIMIT="${LIMIT:-256}"
SCENE_WORKERS="${SCENE_WORKERS:-5}"
NUM_THREADS="${NUM_THREADS:-6}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/aeos-matplotlib}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] aeos python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_ROOT}/summary.json" ]]; then
  echo "[error] 64-scene source summary not found: ${SOURCE_ROOT}/summary.json" >&2
  exit 1
fi

mkdir -p "${CANDIDATE_ROOT}" "${GRAPH_ROOT}"
for candidate in \
  candidate_000_greedy \
  candidate_001_topk_seed3408 \
  candidate_002_topk_seed3409 \
  candidate_003_topk_seed3410; do
  mkdir -p "${CANDIDATE_ROOT}/${candidate}"
  # 已有 64 场只读复用硬链接；新目录的 logs 独立创建，避免覆盖历史日志。
  cp -aln "${SOURCE_ROOT}/${candidate}/." "${CANDIDATE_ROOT}/${candidate}/"
done

echo "[phase] generate candidates limit=${LIMIT} workers=${SCENE_WORKERS} threads=${NUM_THREADS}"
LIMIT="${LIMIT}" \
SCENE_WORKERS="${SCENE_WORKERS}" \
NUM_THREADS="${NUM_THREADS}" \
OUTPUT_ROOT="${CANDIDATE_ROOT}" \
bash scripts/run_same_scene_candidate_smoke.sh

echo "[phase] build first-divergence preferences"
"${PYTHON_BIN}" tools/build_first_divergence_preferences.py \
  "${CANDIDATE_ROOT}/summary.json" \
  --min-cost-margin 0.05 \
  --output "${DIVERGENCE_PATH}"

echo "[phase] train four-fold Graph-Q"
"${PYTHON_BIN}" tools/train_first_divergence_graph_q.py \
  "${DIVERGENCE_PATH}" \
  --output-dir "${GRAPH_ROOT}" \
  --num-folds 4 \
  --hidden-dim 32 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --margin-clip 1.0 \
  --device cpu \
  --num-threads 16

echo "[done] graph_q_summary=${GRAPH_ROOT}/summary.json"
