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

mkdir -p work_dirs/eval_logs work_dirs/eval_summaries

model_path="work_dirs/paper_joint_stage3_30k/checkpoints/iter_30000/model.pth"
retry_csv="tmp/tau_s_val_seen8_retry.csv"

run_eval() {
  local label="$1"
  local tau="$2"
  local run_name="paper_joint_stage3_tau_s_${label}_val_seen8"
  local log_path="work_dirs/eval_logs/${run_name}.log"

  echo "===== $(date '+%F %T') start ${label} =====" | tee "${log_path}"
  if [[ "${tau}" == "none" ]]; then
    unset AEOS_TAU_S
  else
    export AEOS_TAU_S="${tau}"
  fi

  python -m constellation.rl.eval_all \
    "${run_name}" \
    constellation/rl/config_eval.py \
    --override '["environment"]["world_size"]:8' '["environment"]["split"]:"val_seen"' \
    --retry-from "${retry_csv}" \
    --load-model-from "${model_path}" 2>&1 | tee -a "${log_path}"

  python tools/summarize_no_tat_eval.py \
    --output "work_dirs/eval_summaries/${run_name}_no_tat.json" \
    "work_dirs/rl_eval_${run_name}/val_seen" 2>&1 | tee -a "${log_path}"

  echo "===== $(date '+%F %T') done ${label} =====" | tee -a "${log_path}"
}

run_eval baseline none
run_eval tau0001 0.001
run_eval tau001 0.01
run_eval tau005 0.05
run_eval tau01 0.1
run_eval tau03 0.3
run_eval tau05 0.5

python - <<'PY'
import json
from pathlib import Path

labels = ["baseline", "tau0001", "tau001", "tau005", "tau01", "tau03", "tau05"]
rows = []
for label in labels:
    path = Path(f"work_dirs/eval_summaries/paper_joint_stage3_tau_s_{label}_val_seen8_no_tat.json")
    data = json.loads(path.read_text())
    split = next(iter(data["splits"].values()))
    table = split["table"]
    rows.append({
        "label": label,
        "scene_count": split["scene_count"],
        **table,
    })

out = Path("work_dirs/eval_summaries/paper_joint_stage3_tau_s_scan_val_seen8.json")
out.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n")
print(out)
for row in rows:
    print(
        row["label"],
        row["scene_count"],
        f"CS_no_TAT={row['CS_no_TAT']:.4f}",
        f"CR={row['CR_percent']:.2f}",
        f"PCR={row['PCR_percent']:.2f}",
        f"WCR={row['WCR_percent']:.2f}",
        f"PC_Wh={row['PC_Wh']:.2f}",
    )
PY
