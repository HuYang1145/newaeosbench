#!/usr/bin/env bash
#SBATCH --job-name=aeos_temporal_eval8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=lab_team
#SBATCH --output=/home/hy/data/newaeosbench/work_dirs/eval_logs/temporal_adapter_p0_eval8_slurm_%j.log

set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${AEOS_STATE_ROOT:-/home/hy/data/newaeosbench}"

cd "${CODE_ROOT}"
export AEOS_STATE_ROOT="${STATE_ROOT}"

exec bash scripts/eval_temporal_adapter_p0_8.sh
