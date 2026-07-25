#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_appo_u800
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_appo_u800_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
CONFIG="${ROOT_DIR}/constellation/new_transformers/config_event_v2_appo.py"
CANDIDATE="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_3_appo/full_2229/checkpoint_latest.pth"
BASELINE_SUMMARY="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_2_heldout/heldout_2212/v2_2_replica_0/summary.json"
OUTPUT_ROOT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_3_checkpoint_diagnostic/u800_${SLURM_JOB_ID:-manual}"
SCENE_IDS=($(seq 196 203))

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl"
export XDG_CACHE_HOME="/tmp/aeos_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "${CANDIDATE}" || ! -f "${BASELINE_SUMMARY}" ]]; then
  echo "[error] checkpoint or baseline summary is missing" >&2
  exit 1
fi

"${PYTHON}" - "${CANDIDATE}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
if checkpoint.get('stage') != 'V2-3' or checkpoint.get('updates') != 800:
    raise SystemExit('checkpoint_latest is not the expected V2-3 update 800')
PY

mkdir -p "${OUTPUT_ROOT}/v2_3_update_800"
"${PYTHON}" tools/evaluate_event_v2_policy.py \
  --config "${CONFIG}" \
  --checkpoint "${CANDIDATE}" \
  --label v2_3_update_800 \
  --split train \
  --scene-ids "${SCENE_IDS[@]}" \
  --max-time-step 3600 \
  --device cuda \
  --output "${OUTPUT_ROOT}/v2_3_update_800" \
  >"${OUTPUT_ROOT}/v2_3_update_800/evaluate.log" 2>&1

"${PYTHON}" - \
  "${BASELINE_SUMMARY}" \
  "${OUTPUT_ROOT}/v2_3_update_800/summary.json" \
  "${OUTPUT_ROOT}/comparison.json" <<'PY'
import json
import pathlib
import sys

baseline = json.load(open(sys.argv[1]))
candidate = json.load(open(sys.argv[2]))
metric_names = ('CR', 'PCR', 'WCR', 'Q')
delta = {
    name: candidate['aggregate'][name] - baseline['aggregate'][name]
    for name in metric_names
}
result = {
    'protocol': {
        'purpose': 'diagnostic_only',
        'split': 'train',
        'scene_ids': list(range(196, 204)),
        'max_time_step': 3600,
        'deterministic': True,
    },
    'baseline': {
        'label': baseline['label'],
        'checkpoint': baseline['checkpoint'],
        'aggregate': baseline['aggregate'],
    },
    'candidate': {
        'label': candidate['label'],
        'checkpoint': candidate['checkpoint'],
        'aggregate': candidate['aggregate'],
    },
    'delta': delta,
    'candidate_better_q': delta['Q'] > 0,
    'all_completion_metrics_non_decreasing': all(
        delta[name] >= 0 for name in ('CR', 'PCR', 'WCR')
    ),
}
output = pathlib.Path(sys.argv[3])
output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True))
PY
