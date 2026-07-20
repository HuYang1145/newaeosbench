#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_m2_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_actor_m2_smoke_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CHECKPOINT="${ROOT_DIR}/work_dirs/event_heads_m2_10k/checkpoints/iter_10000/model.pth"
OUTPUT_ROOT="${ROOT_DIR}/work_dirs/event_actor_m2_smoke/learned_t050"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] M2 checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${ROOT_DIR}/work_dirs/eval_logs"
"${PYTHON}" tools/rollout_model_trajectories.py \
  "${CHECKPOINT}" \
  "${OUTPUT_ROOT}" \
  --split train \
  --limit 1 \
  --device cpu \
  --strategy greedy \
  --event-actor \
  --event-learned-commitment \
  --event-continue-threshold 0.5 \
  --event-idle-commitment-seconds 1 \
  --overwrite

echo "[done] output=${OUTPUT_ROOT}"
