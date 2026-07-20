from pathlib import Path

from tools.generate_local_graph_q_dataset import (
    build_scene_command,
    discover_reference_trajectories,
)


def test_discover_reference_trajectories_is_sorted_and_limited(
    tmp_path: Path,
) -> None:
    for scene_id in (2, 0, 1):
        path = tmp_path / 'train' / '00' / f'{scene_id:05}.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    paths = discover_reference_trajectories(
        tmp_path,
        split='train',
        limit=2,
    )

    assert [path.stem for path in paths] == ['00000', '00001']


def test_build_scene_command_uses_one_output_directory_per_scene() -> None:
    command = build_scene_command(
        python=Path('/envs/aeos/bin/python'),
        checkpoint=Path('/models/stage3.pth'),
        reference=Path('/candidates/train/00/00017.pth'),
        output_root=Path('/outputs'),
        split='train',
        horizons=(180, 300, 600),
        primary_horizon=300,
        max_decisions=2,
        top_k=3,
        device='cpu',
        overwrite=False,
    )

    assert command[:2] == [
        '/envs/aeos/bin/python',
        'tools/generate_local_action_branches.py',
    ]
    assert '/outputs/scene_00017' in command
    assert command[-2:] == ['--top-k', '3']
