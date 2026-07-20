from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_temporal_pilot_wrapper_pins_config_checkpoint_and_gpu_guard() -> None:
    path = ROOT / 'scripts/train_temporal_adapter_p0_10k.sh'
    source = path.read_text()

    assert 'config_temporal_adapter_p0.py' in source
    assert 'paper_joint_stage3_200k/checkpoints/iter_200000/model.pth' in source
    assert 'temporal_adapter_p0_10k' in source
    assert 'nvidia-smi --query-gpu=memory.used' in source
    assert 'GPU_MEMORY_LIMIT_MB' in source
    assert 'DRY_RUN' in source
    assert '--load-model-from' in source


def test_temporal_val8_wrapper_enables_adapter_for_both_val_splits() -> None:
    path = ROOT / 'scripts/eval_temporal_adapter_p0_8.sh'
    source = path.read_text()

    assert 'val_seen val_unseen' in source
    assert 'max_scenes=8' in source
    assert 'world_size=8' in source
    assert '--use-temporal-adapter' in source
    assert '--temporal-adapter-hidden-width 64' in source
    assert '--temporal-residual-scale 0.25' in source
    assert 'temporal_adapter_p0_10k_val8.json' in source
    assert 'DRY_RUN' in source


def test_temporal_val8_slurm_wrapper_requests_resources_and_reuses_eval() -> None:
    path = ROOT / 'scripts/eval_temporal_adapter_p0_8_slurm.sh'
    source = path.read_text()

    assert '#SBATCH --job-name=aeos_temporal_eval8' in source
    assert '#SBATCH --nodes=1' in source
    assert '#SBATCH --gres=gpu:1' in source
    assert '#SBATCH --cpus-per-task=24' in source
    assert '#SBATCH --mem=96G' in source
    assert '#SBATCH --time=02:00:00' in source
    assert '#SBATCH --account=lab_team' in source
    assert '#SBATCH --partition=local-10' in source
    assert 'temporal_adapter_p0_eval8_slurm_%j.log' in source
    assert 'SLURM_SUBMIT_DIR' in source
    assert 'scripts/eval_temporal_adapter_p0_8.sh' in source
