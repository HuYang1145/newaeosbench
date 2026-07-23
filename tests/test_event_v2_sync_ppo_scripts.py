import os
from pathlib import Path
import runpy
import subprocess

import pytest
import torch

import tools.train_event_v2_sync_ppo as sync_ppo_tool

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / 'constellation/new_transformers/config_event_v2_sync_ppo.py'
FULL_CONFIG = (
    ROOT / 'constellation/new_transformers/config_event_v2_sync_ppo_full.py'
)
SCRIPT = ROOT / 'scripts/train_event_v2_sync_ppo_slurm.sh'
RESUME_SCRIPT = ROOT / 'scripts/resume_event_v2_sync_ppo_slurm.sh'
FULL_SMOKE_SCRIPT = ROOT / 'scripts/smoke_event_v2_2_sync_ppo_slurm.sh'
FULL_SCRIPT = ROOT / 'scripts/train_event_v2_2_full_slurm.sh'


def test_config_is_train_only_and_keeps_stage3_frozen() -> None:
    config = runpy.run_path(str(CONFIG))

    assert config['stage'] == 'V2-1'
    assert config['split'] == 'train'
    assert config['freeze_backbone'] is True
    assert config['max_hours'] == 4
    assert config['safety_review_seconds'] == 5
    assert config['gamma'] == 1.0
    assert config['amp_dtype'] == 'bfloat16'
    assert config['minibatch_events'] == 16
    assert config['optimizer']['lr'] == pytest.approx(2e-5)
    assert 0 < len(config['scene_ids']) <= 4


def test_sync_ppo_cli_help_exposes_bounded_preflight_and_resume() -> None:
    result = subprocess.run(
        [
            '/home/hy/miniconda3/envs/aeos/bin/python',
            'tools/train_event_v2_sync_ppo.py',
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
        '--bootstrap-checkpoint',
        '--scene-ids',
        '--seed',
        '--max-time-step',
        '--max-updates',
        '--resume',
        '--device',
        '--output',
    ):
        assert argument in result.stdout


def test_full_config_preregisters_four_disjoint_train_shards() -> None:
    config = runpy.run_path(str(FULL_CONFIG))
    shards = tuple(tuple(values) for values in config['scene_shards'])
    train_scene_ids = tuple(
        scene_id for shard in shards for scene_id in shard
    )
    heldout_scene_ids = tuple(config['heldout_train_scene_ids'])

    assert config['stage'] == 'V2-2'
    assert config['split'] == 'train'
    assert config['freeze_backbone'] is True
    assert config['max_hours'] == 16
    assert config['max_updates'] == 1400
    assert config['checkpoint_interval'] == 200
    assert config['amp_dtype'] == 'bfloat16'
    assert len(shards) == 4
    assert all(len(shard) == 48 for shard in shards)
    assert len(train_scene_ids) == len(set(train_scene_ids)) == 192
    assert train_scene_ids == tuple(range(4, 196))
    assert heldout_scene_ids == tuple(range(196, 204))
    assert not set(train_scene_ids).intersection(heldout_scene_ids)
    assert config['bootstrap_checkpoint'].endswith(
        'v2_1_sync_ppo/checkpoint_update_000101.pth',
    )


def test_slurm_wrapper_uses_local_10_aeos_and_real_3600_smoke() -> None:
    script = SCRIPT.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --time=04:00:00' in script
    assert '#SBATCH --gres=gpu:1' in script
    assert '#SBATCH --mem=96G' in script
    assert '/home/hy/miniconda3/envs/aeos/bin/python' in script
    assert '--synthetic-preflight' in script
    assert '--device cuda' in script
    assert '--max-time-step 60' in script
    assert '--max-time-step 3600' in script
    assert 'checkpoint_step_010000.pth' in script
    assert 'event_v2_sync_ppo_%j.log' in script
    assert 'v2_1_sync_ppo' in script
    assert 'val_seen' not in script.lower()
    assert 'val_unseen' not in script.lower()
    assert '/test/' not in script.lower()
    assert os.access(SCRIPT, os.X_OK)


def test_cosine_scheduler_holds_eta_min_after_registered_horizon() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=2,
        eta_min=0.1,
    )
    learning_rates = []

    for _ in range(4):
        optimizer.step()
        sync_ppo_tool._step_scheduler_without_restart(scheduler)
        learning_rates.append(optimizer.param_groups[0]['lr'])

    assert learning_rates == pytest.approx([0.55, 0.1, 0.1, 0.1])


def test_resume_wrapper_continues_checkpoint_64_without_restarting() -> None:
    assert RESUME_SCRIPT.is_file()
    script = RESUME_SCRIPT.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:1' in script
    assert 'checkpoint_update_000064.pth' in script
    assert '--resume "${RESUME}"' in script
    assert '--max-updates 104' in script
    assert '--max-time-step 3600' in script
    assert '--device cuda' in script
    assert '--synthetic-preflight' not in script


def test_v2_2_smoke_bootstraps_one_real_train_scene() -> None:
    script = FULL_SMOKE_SCRIPT.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:1' in script
    assert '#SBATCH --time=00:20:00' in script
    assert 'config_event_v2_sync_ppo_full.py' in script
    assert 'checkpoint_update_000101.pth' in script
    assert '--bootstrap-checkpoint "${BOOTSTRAP}"' in script
    assert '--scene-ids 4' in script
    assert '--max-time-step 120' in script
    assert '--max-updates 1' in script
    assert '--device cuda' in script
    assert 'val_seen' not in script.lower()
    assert 'val_unseen' not in script.lower()
    assert '/test/' not in script.lower()
    assert os.access(FULL_SMOKE_SCRIPT, os.X_OK)


def test_v2_2_full_wrapper_uses_four_gpu_disjoint_preregistered_shards() -> None:
    script = FULL_SCRIPT.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:4' in script
    assert '#SBATCH --cpus-per-task=96' in script
    assert '#SBATCH --mem=160G' in script
    assert '#SBATCH --time=16:00:00' in script
    assert 'config_event_v2_sync_ppo_full.py' in script
    assert 'checkpoint_update_000101.pth' in script
    assert 'seq 4 51' in script
    assert 'seq 52 99' in script
    assert 'seq 100 147' in script
    assert 'seq 148 195' in script
    for seed in ('4407', '4408', '4409', '4410'):
        assert seed in script
    assert 'srun --exclusive' in script
    assert '--gres=gpu:1' in script
    assert '--bootstrap-checkpoint "${BOOTSTRAP}"' in script
    assert '--max-time-step 3600' in script
    assert '--max-updates 1400' in script
    assert 'val_seen' not in script.lower()
    assert 'val_unseen' not in script.lower()
    assert '/test/' not in script.lower()
    assert os.access(FULL_SCRIPT, os.X_OK)


def test_exact_resume_keeps_finished_slots_for_future_checkpoints(
    monkeypatch,
) -> None:
    observation = sync_ppo_tool._synthetic_observation(0)

    class FakeRuntime:
        current_observation = observation

        @classmethod
        def from_state_dict(cls, state, *, statistics):
            del state, statistics
            return cls()

    monkeypatch.setattr(
        sync_ppo_tool,
        'BasiliskEventRuntime',
        FakeRuntime,
    )
    states = (
        {
            'environment_index': 0,
            'episode_id': 0,
            'event_index': 10,
            'finished': True,
            'runtime': {'scene_id': 4},
        },
        {
            'environment_index': 1,
            'episode_id': 0,
            'event_index': 8,
            'finished': False,
            'runtime': {'scene_id': 5},
        },
    )

    slots = sync_ppo_tool._restore_runtime_slots(
        states,
        statistics=object(),
    )

    assert len(slots) == 2
    assert [slot.finished for slot in slots] == [True, False]
