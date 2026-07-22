#!/usr/bin/env bash
#SBATCH --job-name=aeos_event_v2_unseen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=lab_team
#SBATCH --partition=local-10
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/event_v2_unseen_offline_%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/hy/data/newaeosbench}"
PYTHON="/home/hy/miniconda3/envs/aeos/bin/python"
TOOL="${ROOT_DIR}/tools/evaluate_event_v2_unseen_offline.py"
ANNOTATION="val_unseen.json"
OUTPUT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_0_unseen_offline/summary.json"
JOB_ID="${SLURM_JOB_ID:-manual}"
PROBE_ROOT="${ROOT_DIR}/work_dirs/event_joint_transformer_v2/v2_0_unseen_offline/probes"
MAX_RESERVED_FRACTION="${MAX_RESERVED_FRACTION:-0.90}"
BATCH_CANDIDATES=(8 16 32 64 128 256 512)

cd "${ROOT_DIR}"
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export MPLCONFIGDIR="/tmp/aeos_mpl_${JOB_ID}"
export XDG_CACHE_HOME="/tmp/aeos_cache_${JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "${PROBE_ROOT}" "${ROOT_DIR}/work_dirs/eval_logs"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if [[ ! -f "${TOOL}" ]]; then
  echo "[error] evaluator not found: ${TOOL}" >&2
  exit 1
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "[error] formal output already exists: ${OUTPUT}" >&2
  exit 1
fi

gpu_preflight="$(${PYTHON} -c 'import json, torch; assert torch.cuda.is_available(); free, total = torch.cuda.mem_get_info(0); print(json.dumps({"free": free, "total": total, "used": total - free, "name": torch.cuda.get_device_name(0)}))')"
echo "[gpu-preflight] ${gpu_preflight}"
gpu_is_clean="$(${PYTHON} -c 'import json, sys; value=json.loads(sys.argv[1]); print("true" if value["used"] <= 2 * 1024**3 else "false")' "${gpu_preflight}")"
if [[ "${gpu_is_clean}" != "true" ]]; then
  echo "[error] allocated GPU already uses more than 2 GiB" >&2
  exit 1
fi

selected_batch=""
for batch_size in "${BATCH_CANDIDATES[@]}"; do
  probe_output="${PROBE_ROOT}/job_${JOB_ID}_batch_${batch_size}.json"
  echo "[probe] event_batch_size=${batch_size}"
  if "${PYTHON}" "${TOOL}" \
    --annotation-file "${ANNOTATION}" \
    --event-batch-size "${batch_size}" \
    --limit 1 \
    --device cuda \
    --output "${probe_output}" \
    --overwrite; then
    probe_is_safe="$(${PYTHON} -c 'import json, sys; value=json.load(open(sys.argv[1])); fraction=value["resources"]["max_reserved_fraction"]; limit=float(sys.argv[2]); print("true" if fraction is not None and fraction <= limit else "false")' "${probe_output}" "${MAX_RESERVED_FRACTION}")"
    if [[ "${probe_is_safe}" == "true" ]]; then
      selected_batch="${batch_size}"
    else
      echo "[probe-stop] reserved fraction exceeds ${MAX_RESERVED_FRACTION}"
      break
    fi
  else
    echo "[probe-stop] batch ${batch_size} failed, treating it as unsafe"
    break
  fi
done

if [[ -z "${selected_batch}" ]]; then
  echo "[error] no safe event batch size was found" >&2
  exit 1
fi

echo "[formal] locked_event_batch_size=${selected_batch}"
exec "${PYTHON}" "${TOOL}" \
  --annotation-file "${ANNOTATION}" \
  --event-batch-size "${selected_batch}" \
  --device cuda \
  --formal \
  --output "${OUTPUT}"
