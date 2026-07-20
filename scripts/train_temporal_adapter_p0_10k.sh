#!/usr/bin/env bash

set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${AEOS_STATE_ROOT:-/home/hy/data/newaeosbench}"
RUN_NAME="${RUN_NAME:-temporal_adapter_p0_10k}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${STATE_ROOT}/work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth}"
GPU_MEMORY_LIMIT_MB="${GPU_MEMORY_LIMIT_MB:-2048}"
DRY_RUN="${DRY_RUN:-0}"

export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/aeos_cache}"

config="${CODE_ROOT}/constellation/new_transformers/config_temporal_adapter_p0.py"
log_dir="${STATE_ROOT}/work_dirs/eval_logs"
log_path="${log_dir}/${RUN_NAME}_train.log"

command=(
  auto_torchrun
  -m constellation.new_transformers.train
  "${RUN_NAME}"
  "${config}"
  --auto-resume
  --load-model-from "${BASE_CHECKPOINT}"
)

printf '[preflight] code_root=%s\n' "${CODE_ROOT}"
printf '[preflight] state_root=%s\n' "${STATE_ROOT}"
printf '[preflight] baseline=%s\n' "${BASE_CHECKPOINT}"
printf '[preflight] command='
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  printf '[error] missing baseline checkpoint: %s\n' "${BASE_CHECKPOINT}" >&2
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  mapfile -t gpu_memory < <(
    nvidia-smi "--id=${CUDA_VISIBLE_DEVICES}" \
      --query-gpu=memory.used --format=csv,noheader,nounits
  )
else
  mapfile -t gpu_memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
fi
for used_mb in "${gpu_memory[@]}"; do
  if (( used_mb > GPU_MEMORY_LIMIT_MB )); then
    printf \
      '[error] GPU busy: %s MiB used exceeds %s MiB; training not started\n' \
      "${used_mb}" "${GPU_MEMORY_LIMIT_MB}" >&2
    exit 2
  fi
done

if [[ ! -e "${CODE_ROOT}/work_dirs" ]]; then
  ln -s "${STATE_ROOT}/work_dirs" "${CODE_ROOT}/work_dirs"
fi
mkdir -p "${log_dir}"
cd "${CODE_ROOT}"
"${command[@]}" 2>&1 | tee -a "${log_path}"
