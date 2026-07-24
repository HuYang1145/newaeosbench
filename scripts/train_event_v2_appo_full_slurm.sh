#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_appo_full
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=120
#SBATCH --mem=200G
#SBATCH --time=28:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_appo_full_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_appo.py"
BOOTSTRAP="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth"
OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_3_appo/full_${SLURM_JOB_ID:-manual}"
: "${SMOKE_SUMMARY:?SMOKE_SUMMARY must point to the accepted V2-3 smoke summary}"

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "${BOOTSTRAP}" || ! -f "${SMOKE_SUMMARY}" ]]; then
  echo "[error] V2-3 bootstrap or smoke summary is missing" >&2
  exit 1
fi
SMOKE_ACCEPTED=$(
  "${PYTHON}" - "${SMOKE_SUMMARY}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))['accepted'])
PY
)
if [[ "${SMOKE_ACCEPTED}" != "True" ]]; then
  echo "[error] V2-3 full training requires accepted=true smoke" >&2
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
  echo "[error] V2-3 full training needs two physically free GPUs" >&2
  exit 1
fi

ACTOR_DEVICES=()
for gpu_index in "${free_gpu_indices[@]}"; do
  ACTOR_DEVICES+=("cuda:${gpu_index}")
done
last_index=$(( ${#free_gpu_indices[@]} - 1 ))
LEARNER_DEVICE="cuda:${free_gpu_indices[$last_index]}"
SCENE_IDS=($(seq 205 324))
echo "[info] actors=${ACTOR_DEVICES[*]} learner=${LEARNER_DEVICE}"

"${PYTHON}" tools/train_event_v2_appo.py \
  --config "${CONFIG}" \
  --bootstrap-checkpoint "${BOOTSTRAP}" \
  --device "${LEARNER_DEVICE}" \
  --actor-devices "${ACTOR_DEVICES[@]}" \
  --scene-ids "${SCENE_IDS[@]}" \
  --max-time-step 3600 \
  --max-updates 5000 \
  --output "${OUTPUT}"
