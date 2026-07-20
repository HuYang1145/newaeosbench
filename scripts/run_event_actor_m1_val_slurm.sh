#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_m1_val
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_actor_m1_val_%j.log

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="/home/hy/data/newaeosbench"
fi
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CHECKPOINT="${ROOT_DIR}/work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth"
OUTPUT_ROOT="${ROOT_DIR}/work_dirs/event_actor_m1_val8"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[error] aeos python not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] Stage3 checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${ROOT_DIR}/work_dirs/eval_logs"

splits=(val_seen val_unseen)
commitments=(1 5 15 30 60)

# 便于静态审计：val_seen val_unseen；1 5 15 30 60。
for split in "${splits[@]}"; do
  for commitment in "${commitments[@]}"; do
    output="${OUTPUT_ROOT}/${split}_event_${commitment}s"
    echo "[start] split=${split} commitment=${commitment} output=${output}"
    srun --exclusive --ntasks=8 --cpus-per-task=6 bash -c '
      export RANK="${SLURM_PROCID}"
      export WORLD_SIZE="${SLURM_NTASKS}"
      export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
      export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
      exec "$@"
    ' _ \
      "${PYTHON}" tools/rollout_model_trajectories.py \
      "${CHECKPOINT}" \
      "${output}" \
      --split "${split}" \
      --limit 8 \
      --device cpu \
      --strategy greedy \
      --event-actor \
      --event-commitment-seconds "${commitment}" \
      --event-idle-commitment-seconds 1
  done
done

echo "[done] output_root=${OUTPUT_ROOT}"
