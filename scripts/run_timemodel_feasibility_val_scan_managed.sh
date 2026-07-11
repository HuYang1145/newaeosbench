#!/usr/bin/env bash
set -euo pipefail

cd /home/hy/data/newaeosbench

export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH=":${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WORLD_SIZE=1
export RANK=0
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/aeos_cache

checkpoint="${CHECKPOINT:-work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth}"
max_scenes="${MAX_SCENES:-8}"
eval_world_size="${EVAL_WORLD_SIZE:-${max_scenes}}"

mkdir -p work_dirs/eval_logs work_dirs/eval_summaries

run_eval() {
  local split="$1"
  local label="$2"
  local threshold="$3"
  local run_name="stage3_200k_feasibility_${label}_${split}_${max_scenes}"
  local log_path="work_dirs/eval_logs/${run_name}.log"
  local summary_path="work_dirs/eval_summaries/${run_name}.json"
  local threshold_args=()

  if [[ "${threshold}" != "none" ]]; then
    threshold_args=(--feasibility-threshold "${threshold}")
  fi

  /home/hy/miniconda3/envs/aeos/bin/python -m constellation.rl.eval_all \
    "${run_name}" \
    constellation/rl/config_eval.py \
    --override "[\"environment\"][\"world_size\"]:${eval_world_size}" "[\"environment\"][\"split\"]:\"${split}\"" \
    --max-scenes "${max_scenes}" \
    --load-model-from "${checkpoint}" \
    "${threshold_args[@]}" 2>&1 | tee "${log_path}"

  /home/hy/miniconda3/envs/aeos/bin/python tools/summarize_no_tat_eval.py \
    --output "${summary_path}" \
    "work_dirs/rl_eval_${run_name}/${split}" 2>&1 | tee -a "${log_path}"
}

for split in val_seen val_unseen; do
  run_eval "${split}" baseline none
  run_eval "${split}" threshold_010 0.1
  run_eval "${split}" threshold_020 0.2
  run_eval "${split}" threshold_030 0.3
done
