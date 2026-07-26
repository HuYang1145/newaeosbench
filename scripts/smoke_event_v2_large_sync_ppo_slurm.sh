#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_large_sync_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=70G
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_large_sync_smoke_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_large_sync_ppo.py"
BOOTSTRAP="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"
OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo/smoke_${SLURM_JOB_ID:-manual}"
LATEST="${OUTPUT}/checkpoint_latest.pth"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

if [[ ! -f "${BOOTSTRAP}" ]]; then
  echo "[error] V2-2 bootstrap checkpoint not found: ${BOOTSTRAP}" >&2
  exit 1
fi
if [[ -e "${LATEST}" ]]; then
  echo "[error] smoke output already contains a checkpoint: ${LATEST}" >&2
  exit 1
fi

mkdir -p "${OUTPUT}"

"${PYTHON}" tools/train_event_v2_large_sync_ppo.py \
  --config "${CONFIG}" \
  --bootstrap-checkpoint "${BOOTSTRAP}" \
  --seed 5408 \
  --learner-device cuda:1 \
  --actor-devices cuda:0 \
  --actors 1 \
  --active-environments 1 \
  --scene-start 205 \
  --scene-end 205 \
  --max-time-step 3600 \
  --max-updates 1 \
  --checkpoint-every-updates 100 \
  --output-dir "${OUTPUT}" \
  >"${OUTPUT}/phase1_train.log" 2>&1

"${PYTHON}" - "${OUTPUT}/summary.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert summary['resumable'] is True
assert summary['accepted'] is False
PY

if [[ ! -f "${LATEST}" ]]; then
  echo "[error] smoke phase 1 did not create checkpoint_latest.pth" >&2
  exit 1
fi

"${PYTHON}" tools/train_event_v2_large_sync_ppo.py \
  --config "${CONFIG}" \
  --bootstrap-checkpoint "${BOOTSTRAP}" \
  --seed 5408 \
  --learner-device cuda:1 \
  --actor-devices cuda:0 \
  --actors 1 \
  --active-environments 1 \
  --scene-start 205 \
  --scene-end 205 \
  --max-time-step 3600 \
  --max-updates 100000 \
  --checkpoint-every-updates 100 \
  --resume "${LATEST}" \
  --output-dir "${OUTPUT}" \
  >"${OUTPUT}/phase2_resume.log" 2>&1

"${PYTHON}" - "${OUTPUT}/summary.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert summary['accepted'] is True
assert summary['all_scenes_finished'] is True
assert summary['stale_rollout_events'] == 0
assert summary['frozen_parameter_changed_count'] == 0
assert summary['logprob_replay_max_error'] <= 1e-6
assert summary['reward_reconstruction_max_error'] <= 1e-6
PY
