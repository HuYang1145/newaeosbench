#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_m2_offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_heads_m2_offline_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
RUN_ROOT="${ROOT_DIR}/work_dirs/event_heads_m2_10k"
OUTPUT_ROOT="${ROOT_DIR}/work_dirs/event_heads_m2_offline"
LIMIT="${LIMIT:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"

mkdir -p "${OUTPUT_ROOT}" "${ROOT_DIR}/work_dirs/eval_logs"

iterations=(1000 2000 5000 10000)
splits=(val_seen val_unseen)
for iteration in "${iterations[@]}"; do
  checkpoint="${RUN_ROOT}/checkpoints/iter_${iteration}/model.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] M2 checkpoint not found: ${checkpoint}" >&2
    exit 1
  fi
  for split in "${splits[@]}"; do
    annotation="${ROOT_DIR}/data/annotations/${split}.json"
    output="${OUTPUT_ROOT}/iter_${iteration}_${split}_${LIMIT}.json"
    "${PYTHON}" -m tools.evaluate_event_heads_m2 \
      "${checkpoint}" \
      --annotation-file "${annotation}" \
      --split "${split}" \
      --limit "${LIMIT}" \
      --batch-size "${BATCH_SIZE}" \
      --device cpu \
      --output "${output}"
  done
done

echo "[done] output_root=${OUTPUT_ROOT}"
