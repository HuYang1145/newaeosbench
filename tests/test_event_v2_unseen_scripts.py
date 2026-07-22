import os
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / 'scripts' / 'evaluate_event_v2_unseen_offline_slurm.sh'


def test_unseen_acceptance_slurm_script_has_fixed_scope_and_gpu_probe() -> None:
    assert SCRIPT.is_file(), 'V2 unseen acceptance Slurm script is missing'
    script = SCRIPT.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:1' in script
    assert '/home/hy/miniconda3/envs/aeos/bin/python' in script
    assert 'evaluate_event_v2_unseen_offline.py' in script
    assert 'val_unseen.json' in script
    assert 'event_v2_unseen_offline_%j.log' in script
    assert 'v2_0_unseen_offline/summary.json' in script
    assert 'BATCH_CANDIDATES=(8 16 32 64 128 256 512)' in script
    assert 'PROBE_SCENE_INDEX="${PROBE_SCENE_INDEX:-36}"' in script
    assert 'MAX_RESERVED_FRACTION="${MAX_RESERVED_FRACTION:-0.90}"' in script
    assert '--limit 1' in script
    assert '--scene-index "${PROBE_SCENE_INDEX}"' in script
    assert '--formal' in script
    assert 'memory.used' in script or 'mem_get_info' in script
    assert os.access(SCRIPT, os.X_OK)


def test_unseen_acceptance_script_does_not_read_forbidden_online_inputs() -> None:
    script = SCRIPT.read_text().lower()

    assert 'data/annotations/test' not in script
    assert 'basilisk' not in script
    assert 'srun ' not in script
