import random
from pathlib import Path

import pytest

from constellation import SATELLITES_ROOT
from constellation.data import Constellation, TaskSet
from constellation.data.constellations import SensorType
from tools.generate_curriculum_scenes import (
    CurriculumSceneSpec,
    generate_curriculum_split,
    sample_curriculum_task,
    validate_spec,
)


def _valid_spec(**overrides: object) -> CurriculumSceneSpec:
    values = dict(
        split='curriculum_600',
        horizon=600,
        num_scenes=128,
        satellite_min=1,
        satellite_max=5,
        task_min=10,
        task_max=50,
        seed=3407,
    )
    values.update(overrides)
    return CurriculumSceneSpec(**values)


def test_validate_spec_accepts_approved_600_second_configuration() -> None:
    validate_spec(_valid_spec())


@pytest.mark.parametrize(
    ('overrides', 'match'),
    [
        ({'horizon': 179}, 'horizon'),
        ({'num_scenes': 0}, 'num_scenes'),
        ({'satellite_min': 0}, 'satellite'),
        ({'satellite_min': 6, 'satellite_max': 5}, 'satellite'),
        ({'task_min': 0}, 'task'),
        ({'task_min': 51, 'task_max': 50}, 'task'),
        ({'split': 'train'}, 'split'),
    ],
)
def test_validate_spec_rejects_invalid_configuration(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_spec(_valid_spec(**overrides))


def test_sample_curriculum_task_stays_inside_approved_bounds() -> None:
    rng = random.Random(3407)

    for task_id in range(2_000):
        task = sample_curriculum_task(task_id, horizon=600, rng=rng)

        assert task.id_ == task_id
        assert 15 <= task.duration <= 60
        assert 0 <= task.release_time < task.due_time <= 600
        assert task.due_time - task.release_time >= 3 * task.duration
        assert -90 <= task.coordinate.x <= 90
        assert -180 <= task.coordinate.y <= 180
        assert task.sensor_type is SensorType.VISIBLE


def _small_spec() -> CurriculumSceneSpec:
    return CurriculumSceneSpec(
        split='curriculum_test',
        horizon=600,
        num_scenes=2,
        satellite_min=1,
        satellite_max=3,
        task_min=2,
        task_max=5,
        seed=17,
    )


def _small_satellite_pool(root: Path) -> Path:
    root.mkdir(parents=True)
    source_files = sorted((SATELLITES_ROOT / 'train').glob('*.json'))[:8]
    assert len(source_files) == 8
    for source in source_files:
        (root / source.name).symlink_to(source.resolve())
    return root


def _generate_in_root(root: Path, satellites_root: Path) -> dict[str, object]:
    return generate_curriculum_split(
        _small_spec(),
        satellites_root=satellites_root,
        constellations_root=root / 'constellations',
        tasksets_root=root / 'tasksets',
        metadata_root=root / 'metadata',
    )


def test_generate_curriculum_split_writes_loadable_expected_layout(
    tmp_path: Path,
) -> None:
    satellites_root = _small_satellite_pool(tmp_path / 'satellites')

    metadata = _generate_in_root(tmp_path / 'output', satellites_root)

    output = tmp_path / 'output'
    expected_paths = [
        output / 'constellations/curriculum_test/00/00000.json',
        output / 'constellations/curriculum_test/00/00001.json',
        output / 'tasksets/curriculum_test/00/00000.json',
        output / 'tasksets/curriculum_test/00/00001.json',
        output / 'metadata/curriculum_test/metadata.json',
    ]
    assert all(path.is_file() for path in expected_paths)
    assert metadata['audit']['scene_ids'] == [0, 1]
    for path in expected_paths[:2]:
        assert 1 <= len(Constellation.load(str(path))) <= 3
    for path in expected_paths[2:4]:
        assert 2 <= len(TaskSet.load(str(path))) <= 5


def test_generate_curriculum_split_is_deterministic(tmp_path: Path) -> None:
    satellites_root = _small_satellite_pool(tmp_path / 'satellites')

    _generate_in_root(tmp_path / 'first', satellites_root)
    _generate_in_root(tmp_path / 'second', satellites_root)

    for kind in ('constellations', 'tasksets'):
        first_files = sorted(
            (tmp_path / 'first' / kind / 'curriculum_test').rglob('*.json')
        )
        second_files = sorted(
            (tmp_path / 'second' / kind / 'curriculum_test').rglob('*.json')
        )
        assert [path.read_text() for path in first_files] == [
            path.read_text() for path in second_files
        ]


def test_generate_curriculum_split_refuses_nonempty_target(
    tmp_path: Path,
) -> None:
    satellites_root = _small_satellite_pool(tmp_path / 'satellites')
    target = tmp_path / 'output/constellations/curriculum_test'
    target.mkdir(parents=True)
    sentinel = target / 'keep.txt'
    sentinel.write_text('不要覆盖', encoding='utf-8')

    with pytest.raises(FileExistsError, match='curriculum_test'):
        _generate_in_root(tmp_path / 'output', satellites_root)

    assert sentinel.read_text(encoding='utf-8') == '不要覆盖'
    assert not (tmp_path / 'output/tasksets/curriculum_test').exists()
