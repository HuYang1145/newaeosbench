"""生成短时长课程学习用的静态星座与任务场景。"""

import argparse
import dataclasses
import json
import pathlib
import random
import statistics
import tempfile

import todd

from constellation import CONSTELLATIONS_ROOT, SATELLITES_ROOT, TASKSETS_ROOT
from constellation.data import (
    Constellation,
    Coordinate,
    Satellite,
    Satellites,
    Task,
    TaskSet,
)
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


def load_satellite_pool(root: pathlib.Path) -> Satellites:
    """按文件名排序加载已通过筛选的单星卫星池。"""

    satellites: list[Satellite] = []
    files = sorted(root.glob('*.json'))
    if not files:
        raise FileNotFoundError(f'卫星池为空或不存在: {root}')
    for path in files:
        constellation = Constellation.load(str(path))
        if len(constellation) != 1:
            raise ValueError(f'卫星池文件必须只含一颗卫星: {path}')
        satellites.extend(constellation.values())
    return satellites


def _scene_id(path: pathlib.Path) -> int:
    try:
        return int(path.stem)
    except ValueError as error:
        raise ValueError(f'场景文件名不是数字 ID: {path}') from error


def _summary(values: list[int]) -> dict[str, float | int]:
    return {
        'min': min(values),
        'median': statistics.median(values),
        'max': max(values),
    }


def audit_generated_split(
    spec: CurriculumSceneSpec,
    *,
    constellation_dir: pathlib.Path,
    taskset_dir: pathlib.Path,
) -> dict[str, object]:
    """独立加载全部 JSON，并核对数量、ID 与课程范围。"""

    validate_spec(spec)
    constellation_files = sorted(constellation_dir.rglob('*.json'))
    taskset_files = sorted(taskset_dir.rglob('*.json'))
    expected_ids = list(range(spec.num_scenes))
    constellation_ids = [_scene_id(path) for path in constellation_files]
    taskset_ids = [_scene_id(path) for path in taskset_files]
    if constellation_ids != expected_ids:
        raise ValueError(
            f'星座场景 ID 不完整: expected={expected_ids}, actual={constellation_ids}'
        )
    if taskset_ids != expected_ids:
        raise ValueError(
            f'任务场景 ID 不完整: expected={expected_ids}, actual={taskset_ids}'
        )

    satellite_counts: list[int] = []
    task_counts: list[int] = []
    for scene_id, (constellation_path, taskset_path) in enumerate(
        zip(constellation_files, taskset_files, strict=True)
    ):
        constellation = Constellation.load(str(constellation_path))
        taskset = TaskSet.load(str(taskset_path))
        satellite_counts.append(len(constellation))
        task_counts.append(len(taskset))
        if not spec.satellite_min <= len(constellation) <= spec.satellite_max:
            raise ValueError(f'场景 {scene_id} 的卫星数量越界')
        if not spec.task_min <= len(taskset) <= spec.task_max:
            raise ValueError(f'场景 {scene_id} 的任务数量越界')
        if [task.id_ for task in taskset] != list(range(len(taskset))):
            raise ValueError(f'场景 {scene_id} 的任务 ID 不连续')
        for task in taskset:
            if not 15 <= task.duration <= 60:
                raise ValueError(f'场景 {scene_id} 的任务持续时间越界')
            if not 0 <= task.release_time < task.due_time <= spec.horizon:
                raise ValueError(f'场景 {scene_id} 的任务时间窗越界')
            if task.due_time - task.release_time < 3 * task.duration:
                raise ValueError(f'场景 {scene_id} 的任务时间窗过短')
            if not -90 <= task.coordinate.x <= 90:
                raise ValueError(f'场景 {scene_id} 的任务纬度越界')
            if not -180 <= task.coordinate.y <= 180:
                raise ValueError(f'场景 {scene_id} 的任务经度越界')
            if task.sensor_type is not SensorType.VISIBLE:
                raise ValueError(f'场景 {scene_id} 的任务传感器类型不匹配')

    return {
        'constellation_files': len(constellation_files),
        'taskset_files': len(taskset_files),
        'scene_ids': expected_ids,
        'satellite_counts': _summary(satellite_counts),
        'task_counts': _summary(task_counts),
        'errors': [],
    }


def _ensure_empty_or_absent(path: pathlib.Path) -> None:
    if not path.exists():
        return
    if not path.is_dir() or any(path.iterdir()):
        raise FileExistsError(f'目标已存在且非空，拒绝覆盖: {path}')


def generate_curriculum_split(
    spec: CurriculumSceneSpec,
    *,
    satellites_root: pathlib.Path,
    constellations_root: pathlib.Path,
    tasksets_root: pathlib.Path,
    metadata_root: pathlib.Path,
) -> dict[str, object]:
    """生成、审计并发布一个隔离的课程场景划分。"""

    validate_spec(spec)
    constellation_target = constellations_root / spec.split
    taskset_target = tasksets_root / spec.split
    metadata_dir = metadata_root / spec.split
    metadata_path = metadata_dir / 'metadata.json'
    _ensure_empty_or_absent(constellation_target)
    _ensure_empty_or_absent(taskset_target)
    if metadata_path.exists():
        raise FileExistsError(f'元数据已存在，拒绝覆盖: {metadata_path}')

    satellites = load_satellite_pool(satellites_root)
    if spec.satellite_max > len(satellites):
        raise ValueError(
            f'satellite_max={spec.satellite_max} 超过卫星池大小 {len(satellites)}'
        )

    todd.utils.init_seed(spec.seed)
    task_rng = random.Random(spec.seed)
    constellations_root.mkdir(parents=True, exist_ok=True)
    tasksets_root.mkdir(parents=True, exist_ok=True)
    constellation_temp = pathlib.Path(
        tempfile.mkdtemp(
            prefix=f'.{spec.split}.',
            dir=constellations_root,
        )
    )
    taskset_temp = pathlib.Path(
        tempfile.mkdtemp(prefix=f'.{spec.split}.', dir=tasksets_root)
    )

    for scene_id in range(spec.num_scenes):
        relative_path = pathlib.Path(
            f'{scene_id // 1000:02}/{scene_id:05}.json'
        )
        constellation_path = constellation_temp / relative_path
        taskset_path = taskset_temp / relative_path
        constellation_path.parent.mkdir(parents=True, exist_ok=True)
        taskset_path.parent.mkdir(parents=True, exist_ok=True)

        constellation = Constellation.sample(
            satellites,
            random.randint(spec.satellite_min, spec.satellite_max),
        )
        taskset = TaskSet([
            sample_curriculum_task(
                task_id,
                horizon=spec.horizon,
                rng=task_rng,
            )
            for task_id in range(random.randint(spec.task_min, spec.task_max))
        ])
        constellation.dump(str(constellation_path))
        taskset.dump(str(taskset_path))

    audit = audit_generated_split(
        spec,
        constellation_dir=constellation_temp,
        taskset_dir=taskset_temp,
    )

    if constellation_target.exists():
        constellation_target.rmdir()
    if taskset_target.exists():
        taskset_target.rmdir()
    constellation_temp.replace(constellation_target)
    taskset_temp.replace(taskset_target)

    train_ids = list(range(min(120, spec.num_scenes)))
    heldout_ids = (
        list(range(120, 128)) if spec.num_scenes == 128 else []
    )
    metadata: dict[str, object] = {
        'spec': dataclasses.asdict(spec),
        'satellite_pool': {
            'path': str(satellites_root),
            'count': len(satellites),
        },
        'scene_partition': {
            'train_ids': train_ids,
            'heldout_ids': heldout_ids,
        },
        'audit': audit,
    }
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', required=True)
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--num-scenes', type=int, required=True)
    parser.add_argument('--satellite-min', type=int, required=True)
    parser.add_argument('--satellite-max', type=int, required=True)
    parser.add_argument('--task-min', type=int, required=True)
    parser.add_argument('--task-max', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = CurriculumSceneSpec(
        split=args.split,
        horizon=args.horizon,
        num_scenes=args.num_scenes,
        satellite_min=args.satellite_min,
        satellite_max=args.satellite_max,
        task_min=args.task_min,
        task_max=args.task_max,
        seed=args.seed,
    )
    metadata = generate_curriculum_split(
        spec,
        satellites_root=SATELLITES_ROOT / 'train',
        constellations_root=CONSTELLATIONS_ROOT,
        tasksets_root=TASKSETS_ROOT,
        metadata_root=pathlib.Path('work_dirs/curriculum_scenes'),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
