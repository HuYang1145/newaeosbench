from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_event_actor_smoke_uses_stage3_and_aeos_python() -> None:
    script = (
        ROOT / 'scripts' / 'run_event_actor_m1_smoke.sh'
    ).read_text(encoding='utf-8')

    assert '/home/hy/miniconda3/envs/aeos/bin/python' in script
    assert 'paper_joint_stage3_200k/checkpoints/iter_200000/model.pth' in script
    assert '--event-actor' in script
    assert '--event-commitment-seconds 5' in script
    assert '--event-idle-commitment-seconds 1' in script
    assert 'work_dirs/event_actor_m1_smoke' in script
    assert '--split test' not in script


def test_event_actor_val_wrapper_uses_slurm_and_never_test_split() -> None:
    script = (
        ROOT / 'scripts' / 'run_event_actor_m1_val_slurm.sh'
    ).read_text(encoding='utf-8')

    assert '#SBATCH --account=lab_team' in script
    assert '#SBATCH --partition=local-10' in script
    assert '/home/hy/miniconda3/envs/aeos/bin/python' in script
    assert 'val_seen val_unseen' in script
    assert '1 5 15 30 60' in script
    assert 'SLURM_PROCID' in script
    assert '--event-actor' in script
    assert '--event-idle-commitment-seconds 1' in script
    assert '--split test' not in script
