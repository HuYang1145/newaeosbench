"""生成短时长课程学习用的静态星座与任务场景。"""

import dataclasses
import random

from constellation.data import Coordinate, Task
from constellation.data.constellations import SensorType


@dataclasses.dataclass(frozen=True)
class CurriculumSceneSpec:
    """一个课程场景划分的可复现生成规格。"""

    split: str
    horizon: int
    num_scenes: int
    satellite_min: int
    satellite_max: int
    task_min: int
    task_max: int
    seed: int


def validate_spec(spec: CurriculumSceneSpec) -> None:
    """在写文件之前拒绝不合法或非课程用途的规格。"""

    if not spec.split.startswith('curriculum_'):
        raise ValueError("split 必须以 'curriculum_' 开头")
    if spec.horizon < 3 * 60:
        raise ValueError('horizon 必须至少为 180 秒')
    if spec.num_scenes <= 0:
        raise ValueError('num_scenes 必须为正整数')
    if spec.satellite_min <= 0 or spec.satellite_min > spec.satellite_max:
        raise ValueError('satellite 数量范围不合法')
    if spec.task_min <= 0 or spec.task_min > spec.task_max:
        raise ValueError('task 数量范围不合法')


def sample_curriculum_task(
    task_id: int,
    *,
    horizon: int,
    rng: random.Random,
) -> Task:
    """采样一个在给定时域内至少有三倍执行时长窗口的任务。"""

    duration = rng.randint(15, 60)
    release_time = rng.randint(0, horizon - 3 * duration)
    due_time = rng.randint(release_time + 3 * duration, horizon)
    return Task(
        id_=task_id,
        release_time=release_time,
        due_time=due_time,
        duration=duration,
        coordinate=Coordinate(
            rng.uniform(-90, 90),
            rng.uniform(-180, 180),
        ),
        sensor_type=SensorType.VISIBLE,
    )
