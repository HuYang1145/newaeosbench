from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m2_smoke_uses_real_stage3_checkpoint_and_aeos_python() -> None:
    script = (
        ROOT / 'scripts' / 'smoke_event_heads_m2.sh'
    ).read_text(encoding='utf-8')

    assert '/home/hy/miniconda3/envs/aeos/bin/python' in script
    assert 'tools/smoke_event_heads_m2.py' in script
    assert 'paper_joint_stage3_200k/checkpoints/iter_200000/model.pth' in script
    assert 'train_paper_stage3_tau_e_existing.json' in script
    assert '--split train' in script


def test_m2_training_wrapper_uses_slurm_and_local10() -> None:
    script = (
        ROOT / 'scripts' / 'train_event_heads_m2_slurm.sh'
    ).read_text(encoding='utf-8')

    assert '#SBATCH --account=lab_team' in script
    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --gres=gpu:1' in script
    assert '/home/hy/miniconda3/envs/aeos/bin' in script
    assert 'config_event_heads_m2.py' in script
    assert 'paper_joint_stage3_200k/checkpoints/iter_200000/model.pth' in script
    assert 'event_heads_m2_10k' in script
    assert '--split test' not in script


def test_m2_offline_evaluation_uses_val_only_and_multiple_checkpoints() -> None:
    script = (
        ROOT / 'scripts' / 'evaluate_event_heads_m2_offline_slurm.sh'
    ).read_text(encoding='utf-8')

    assert '#SBATCH --account=lab_team' in script
    assert '#SBATCH --partition=local-10' in script
    assert 'tools.evaluate_event_heads_m2' in script
    assert 'val_seen val_unseen' in script
    assert '1000 2000 5000 10000' in script
    assert '--device cpu' in script
    assert '--split test' not in script
    assert 'data/annotations/test.json' not in script
