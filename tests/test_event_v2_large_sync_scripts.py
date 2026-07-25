from __future__ import annotations

import os
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'scripts/smoke_event_v2_large_sync_ppo_slurm.sh'
FULL = ROOT / 'scripts/train_event_v2_large_sync_ppo_full_slurm.sh'
RESUME = ROOT / 'scripts/resume_event_v2_large_sync_ppo_full_slurm.sh'


def test_large_sync_cli_exposes_resource_and_resume_boundaries() -> None:
    result = subprocess.run(
        [
            '/home/hy/miniconda3/envs/aeos/bin/python',
            'tools/train_event_v2_large_sync_ppo.py',
            '--help',
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for argument in (
        '--synthetic-preflight',
        '--learner-device',
        '--actor-devices',
        '--actors',
        '--active-environments',
        '--scene-start',
        '--scene-end',
        '--max-time-step',
        '--max-updates',
        '--checkpoint-every-updates',
        '--resume',
        '--output-dir',
    ):
        assert argument in result.stdout


def test_large_sync_smoke_runs_one_real_3600_scene_and_exact_resume() -> None:
    script = SMOKE.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --account=lab_team' in script
    assert '#SBATCH --gres=gpu:2' in script
    assert '#SBATCH --cpus-per-task=32' in script
    assert '#SBATCH --time=' not in script
    assert 'config_event_v2_large_sync_ppo.py' in script
    assert 'checkpoint_update_001046.pth' in script
    assert '--scene-start 205' in script
    assert '--scene-end 205' in script
    assert '--max-time-step 3600' in script
    assert '--actors 1' in script
    assert '--active-environments 1' in script
    assert '--max-updates 1' in script
    assert '--resume "${LATEST}"' in script
    assert '--max-updates 100000' in script
    assert "summary['resumable'] is True" in script
    assert "summary['accepted'] is True" in script
    assert 'val_seen' not in script.lower()
    assert 'val_unseen' not in script.lower()
    assert '--split test' not in script.lower()
    assert os.access(SMOKE, os.X_OK)


def test_large_sync_full_uses_two_seeds_four_gpus_and_at_most_120_cpus(
) -> None:
    script = FULL.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --account=lab_team' in script
    assert '#SBATCH --gres=gpu:4' in script
    assert '#SBATCH --cpus-per-task=120' in script
    assert '#SBATCH --time=' not in script
    assert '--cpus-per-task=144' not in script
    assert 'config_event_v2_large_sync_ppo.py' in script
    assert 'checkpoint_update_001046.pth' in script
    assert ': "${SMOKE_SUMMARY:?' in script
    assert "summary['accepted'] is True" in script
    assert 'SEEDS=(5408 5409)' in script
    assert 'GPU_PAIR_A=' in script
    assert 'GPU_PAIR_B=' in script
    assert 'CUDA_VISIBLE_DEVICES="${GPU_PAIR_A}"' in script
    assert 'CUDA_VISIBLE_DEVICES="${GPU_PAIR_B}"' in script
    assert '--actors 12' in script
    assert '--active-environments 60' in script
    assert '--scene-start 205' in script
    assert '--scene-end 324' in script
    assert '--max-time-step 3600' in script
    assert '--max-updates 100000' in script
    assert '--checkpoint-every-updates 100' in script
    assert 'OMP_NUM_THREADS=4' in script
    assert 'MKL_NUM_THREADS=4' in script
    assert 'OPENBLAS_NUM_THREADS=4' in script
    assert 'wait "${pid}"' in script
    assert 'val_seen' not in script.lower()
    assert 'val_unseen' not in script.lower()
    assert '--split test' not in script.lower()
    assert os.access(FULL, os.X_OK)


def test_large_sync_resume_uses_each_seed_latest_without_restarting() -> None:
    script = RESUME.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:4' in script
    assert '#SBATCH --cpus-per-task=120' in script
    assert '#SBATCH --time=' not in script
    assert 'seed_5408/checkpoint_latest.pth' in script
    assert 'seed_5409/checkpoint_latest.pth' in script
    assert '--resume "${resume_checkpoint}"' in script
    assert '--max-updates 100000' in script
    assert '--checkpoint-every-updates 100' in script
    assert "summary.get('accepted') is True" in script
    assert 'CUDA_VISIBLE_DEVICES="${GPU_PAIR_A}"' in script
    assert 'CUDA_VISIBLE_DEVICES="${GPU_PAIR_B}"' in script
    assert os.access(RESUME, os.X_OK)


def test_large_sync_shell_wrappers_are_syntactically_valid() -> None:
    for script in (SMOKE, FULL, RESUME):
        result = subprocess.run(
            ['bash', '-n', str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
