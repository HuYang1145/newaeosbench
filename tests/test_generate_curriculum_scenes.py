import random

import pytest

from constellation.data.constellations import SensorType
from tools.generate_curriculum_scenes import (
    CurriculumSceneSpec,
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
