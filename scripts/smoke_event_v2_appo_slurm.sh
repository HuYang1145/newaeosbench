#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_appo_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_appo_smoke_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_appo.py"
BOOTSTRAP="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"
OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_3_appo/smoke_${SLURM_JOB_ID:-manual}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "${BOOTSTRAP}" ]]; then
  echo "[error] V2-2 selected checkpoint not found: ${BOOTSTRAP}" >&2
  exit 1
fi

free_gpu_indices=()
while IFS=',' read -r gpu_index memory_used; do
  gpu_index="${gpu_index// /}"
  memory_used="${memory_used// /}"
  if (( memory_used < 4096 )); then
    free_gpu_indices+=("${gpu_index}")
  fi
done < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
)
if (( ${#free_gpu_indices[@]} < 2 )); then
  echo "[error] V2-3 smoke needs two physically free GPUs" >&2
  exit 1
fi

ACTOR_DEVICE="cuda:${free_gpu_indices[0]}"
LEARNER_DEVICE="cuda:${free_gpu_indices[1]}"
echo "[info] actor=${ACTOR_DEVICE} learner=${LEARNER_DEVICE}"

"${PYTHON}" tools/train_event_v2_appo.py \
  --config "${CONFIG}" \
  --bootstrap-checkpoint "${BOOTSTRAP}" \
  --device "${LEARNER_DEVICE}" \
  --actor-devices "${ACTOR_DEVICE}" \
  --scene-ids 205 \
  --max-time-step 3600 \
  --max-updates 1000 \
  --output "${OUTPUT}"
