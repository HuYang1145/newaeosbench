#!/usr/bin/env bash

set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${AEOS_STATE_ROOT:-/home/hy/data/newaeosbench}"
CHECKPOINT="${CHECKPOINT:-${STATE_ROOT}/work_dirs/temporal_adapter_p0_10k/checkpoints/iter_10000/model.pth}"
DRY_RUN="${DRY_RUN:-0}"
max_scenes=8
world_size=8

export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
export WORLD_SIZE=1
export RANK=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/aeos_cache}"

if [[ "${DRY_RUN}" != "1" && ! -f "${CHECKPOINT}" ]]; then
  printf '[error] missing Temporal Adapter checkpoint: %s\n' \
    "${CHECKPOINT}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" && ! -e "${CODE_ROOT}/work_dirs" ]]; then
  ln -s "${STATE_ROOT}/work_dirs" "${CODE_ROOT}/work_dirs"
fi
mkdir -p "${STATE_ROOT}/work_dirs/eval_logs" \
  "${STATE_ROOT}/work_dirs/eval_summaries"
cd "${CODE_ROOT}"

for split in val_seen val_unseen; do
  run_name="temporal_adapter_p0_10k_${split}_8"
  command=(
    /home/hy/miniconda3/envs/aeos/bin/python
    -m constellation.rl.eval_all
    "${run_name}"
    constellation/rl/config_eval.py
    --override
    "[\"environment\"][\"world_size\"]:${world_size}"
    "[\"environment\"][\"split\"]:\"${split}\""
    --max-scenes "${max_scenes}"
    --use-temporal-adapter
    --temporal-adapter-hidden-width 64
    --temporal-residual-scale 0.25
    --coordination-diagnostics-top-k 5
    --load-model-from "${CHECKPOINT}"
  )
  printf '[preflight] command='
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    log_path="${STATE_ROOT}/work_dirs/eval_logs/${run_name}.log"
    "${command[@]}" 2>&1 | tee -a "${log_path}"
  fi
done

summary_command=(
  /home/hy/miniconda3/envs/aeos/bin/python
  tools/summarize_eval.py
  --output
  "${STATE_ROOT}/work_dirs/eval_summaries/temporal_adapter_p0_10k_val8.json"
  "${STATE_ROOT}/work_dirs/rl_eval_temporal_adapter_p0_10k_val_seen_8/val_seen"
  "${STATE_ROOT}/work_dirs/rl_eval_temporal_adapter_p0_10k_val_unseen_8/val_unseen"
)
printf '[preflight] command='
printf ' %q' "${summary_command[@]}"
printf '\n'
if [[ "${DRY_RUN}" != "1" ]]; then
  "${summary_command[@]}"
fi
