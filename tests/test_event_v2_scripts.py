import os
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).parents[1]


def test_warm_start_cli_help_is_runnable() -> None:
    result = subprocess.run(
        [
            '/home/hy/miniconda3/envs/aeos/bin/python',
            'tools/train_event_v2_warm_start.py',
            '--help',
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '--stage3-checkpoint' in result.stdout
    assert '--max-steps' in result.stdout
    assert '--device' in result.stdout


def test_warm_start_slurm_script_has_bounded_resources_and_preflight() -> None:
    path = ROOT / 'scripts/train_event_v2_warm_start_slurm.sh'
    script = path.read_text()

    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --time=04:00:00' in script
    assert '#SBATCH --gres=gpu:1' in script
    assert '/home/hy/miniconda3/envs/aeos/bin' in script
    assert 'iter_200000/model.pth' in script
    assert '--max-steps 1' in script
    assert 'train_event_v2_warm_start.py' in script
    assert os.access(path, os.X_OK)
