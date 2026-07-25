from __future__ import annotations

import os
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'scripts/smoke_event_v2_large_sync_ppo_slurm.sh'
FULL = ROOT / 'scripts/train_event_v2_large_sync_ppo_full_slurm.sh'
RESUME = ROOT / 'scripts/resume_event_v2_large_sync_ppo_full_slurm.sh'
HELDOUT = ROOT / 'scripts/select_event_v2_large_sync_heldout_slurm.sh'
VAL_GATE = ROOT / 'scripts/eval_event_v2_large_sync_gate_slurm.sh'
FULL_VAL = ROOT / 'scripts/eval_event_v2_large_sync_full_val_slurm.sh'
TEST_ONCE = ROOT / 'scripts/eval_event_v2_large_sync_test_once_slurm.sh'


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
    for script in (
        SMOKE,
        FULL,
        RESUME,
        HELDOUT,
        VAL_GATE,
        FULL_VAL,
        TEST_ONCE,
    ):
        result = subprocess.run(
            ['bash', '-n', str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_large_sync_heldout_evaluates_every_permanent_checkpoint_only_on_train(
) -> None:
    script = HELDOUT.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:4' in script
    assert '#SBATCH --time=' not in script
    assert 'seed_5408' in script
    assert 'seed_5409' in script
    assert "checkpoint_update_*.pth" in script
    assert "checkpoint_final_update_*.pth" in script
    assert 'checkpoint_latest.pth' not in script
    assert 'seq 196 203' in script
    assert '--split train' in script
    assert '--max-time-step 3600' in script
    assert 'evaluate_event_v2_policy.py' in script
    assert 'select_event_v2_large_sync_heldout.py' in script
    assert '--expected-scene-ids "${SCENE_IDS[@]}"' in script
    assert '--best-link "${BEST_LINK}"' in script
    assert 'val_seen' not in script.lower()
    assert 'val_unseen' not in script.lower()
    assert '--split test' not in script.lower()
    assert os.access(HELDOUT, os.X_OK)


def test_large_sync_val_gate_uses_only_new_scenes_8_to_15_once() -> None:
    script = VAL_GATE.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:4' in script
    assert '#SBATCH --time=' not in script
    assert ': "${SELECTION_JSON:?' in script
    assert "['selected']['checkpoint']" in script
    assert 'checkpoint_update_001046.pth' in script
    assert 'SCENE_IDS=($(seq 8 15))' in script
    assert 'SPLITS=("val_seen" "val_seen" "val_unseen" "val_unseen")' in script
    assert 'evaluate_event_v2_policy.py' in script
    assert 'compare_event_v2_val_gate.py' in script
    assert '--minimum-q-improvement 0.005' in script
    assert '--baseline-stage V2-2' in script
    assert '--candidate-stage V2-2-Large' in script
    assert '--expected-scene-ids "${SCENE_IDS[@]}"' in script
    assert '--split test' not in script.lower()
    assert os.access(VAL_GATE, os.X_OK)


def test_large_sync_full_val_reports_three_preregistered_scene_groups() -> None:
    script = FULL_VAL.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:4' in script
    assert '#SBATCH --time=' not in script
    assert ': "${SELECTION_JSON:?' in script
    assert ': "${GATE_JSON:?' in script
    assert "gate['passed'] is True" in script
    assert 'GROUP_STARTS=(0 8 16)' in script
    assert 'GROUP_ENDS=(7 15 63)' in script
    assert 'GROUP_NAMES=("history_0_7" "gate_8_15" "rest_16_63")' in script
    assert 'val_seen' in script
    assert 'val_unseen' in script
    assert 'merge_event_v2_eval_summaries.py' in script
    assert 'compare_event_v2_full_val.py' in script
    assert '--minimum-q-improvement 0.005' in script
    assert '--expected-scene-ids "${ALL_SCENE_IDS[@]}"' in script
    assert 'TAT_s' in script
    assert 'PC_Wh' in script
    assert 'CS_paper' in script
    assert '--split test' not in script.lower()
    assert os.access(FULL_VAL, os.X_OK)


def test_large_sync_test_runs_locked_candidate_once_after_full_val_passes(
) -> None:
    script = TEST_ONCE.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:4' in script
    assert '#SBATCH --time=' not in script
    assert ': "${SELECTION_JSON:?' in script
    assert ': "${FULL_VAL_RESULT:?' in script
    assert "full_val['passed'] is True" in script
    assert "['selected']['checkpoint']" in script
    assert 'SHARD_STARTS=(0 16 32 48)' in script
    assert 'SHARD_ENDS=(15 31 47 63)' in script
    assert '--split test' in script.lower()
    assert 'merge_event_v2_eval_summaries.py' in script
    assert '--expected-scene-ids "${ALL_SCENE_IDS[@]}"' in script
    assert 'checkpoint_update_001046.pth' not in script
    assert os.access(TEST_ONCE, os.X_OK)
