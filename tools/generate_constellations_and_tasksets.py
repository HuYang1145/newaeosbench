"""为各数据划分生成星座和任务集 JSON 文件。

从预生成的卫星池中采样星座，并采样任务集，写入 ``data/constellations``
和 ``data/tasksets`` 目录。此为数据生成工具，非训练入口。
"""

import random
import math
from collections.abc import Sequence

import todd
import torch
from tqdm import tqdm, trange

from constellation import CONSTELLATIONS_ROOT, SATELLITES_ROOT, TASKSETS_ROOT
from constellation.constants import MAX_OFF_NADIR_ANGLE, MAX_TIME_STEP
from constellation.constants import MU_EARTH
from constellation.constants import RADIUS_EARTH
from constellation.data import Constellation, Satellite, Satellites, Task
from constellation.data import TaskSet

EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s


def load_satellites(split: str) -> Satellites:
    satellites_root = SATELLITES_ROOT / split
    satellites: list[Satellite] = []
    for f in tqdm(satellites_root.iterdir()):
        assert f.suffix == '.json'
        constellation = Constellation.load(str(f))
        assert len(constellation) == 1
        satellites.extend(constellation.values())

    todd.logger.info("Loaded %d satellites", len(satellites))
    return satellites


def has_contiguous_observation_window(
    task: Task,
    visible_by_time: Sequence[bool] | torch.Tensor,
) -> bool:
    """判断任务在自身时间窗内是否存在连续可观测片段。"""
    streak = 0
    start = max(task.release_time, 0)
    stop = min(task.due_time + 1, len(visible_by_time))

    for time_step in range(start, stop):
        visible = visible_by_time[time_step]
        if isinstance(visible, torch.Tensor):
            visible = bool(visible.item())

        if visible:
            streak += 1
            if streak >= task.duration:
                return True
        else:
            streak = 0

    return False


def renumber_taskset(taskset: TaskSet) -> TaskSet:
    """筛选后重新编号，避免 task id 与列表位置不一致。"""
    return TaskSet(
        Task(
            id_,
            task.release_time,
            task.due_time,
            task.duration,
            task.coordinate,
            task.sensor_type,
        ) for id_, task in enumerate(taskset)
    )


def scan_observable_task_flags(
    constellation: Constellation,
    taskset: TaskSet,
    *,
    max_time_step: int = MAX_TIME_STEP,
) -> list[bool]:
    """快速扫描每个任务是否存在可完成的连续观测窗口。"""
    visibility_trace = _fast_geometric_visibility_trace(
        constellation,
        taskset,
        max_time_step=max_time_step,
    )

    return [
        has_contiguous_observation_window(task, visibility_trace[:, i])
        for i, task in enumerate(taskset)
    ]


def _fast_geometric_visibility_trace(
    constellation: Constellation,
    taskset: TaskSet,
    *,
    max_time_step: int = MAX_TIME_STEP,
    chunk_size: int = 120,
) -> torch.Tensor:
    """不启动 Basilisk，仅用轨道传播和几何约束计算任务可见性。"""
    satellites = constellation.sort()
    satellite_sensor_type = torch.tensor(
        [satellite.sensor.type_ for satellite in satellites],
    )
    task_sensor_type = torch.tensor([task.sensor_type for task in taskset])
    mask_sensor = satellite_sensor_type.unsqueeze(1) == task_sensor_type

    taskset_ecef = torch.tensor(taskset.coordinates_ecef, dtype=torch.float32)
    visibility_trace = torch.zeros(
        max_time_step,
        len(taskset),
        dtype=torch.bool,
    )

    for start in range(0, max_time_step, chunk_size):
        stop = min(start + chunk_size, max_time_step)
        times = torch.arange(start, stop, dtype=torch.float32)

        constellation_eci = _satellite_positions_eci(satellites, times)
        taskset_eci = _task_positions_eci(taskset_ecef, times)
        delta = taskset_eci.unsqueeze(1) - constellation_eci.unsqueeze(2)
        distance = torch.norm(delta, dim=-1)
        orbital_radius = torch.norm(constellation_eci, dim=-1).unsqueeze(-1)

        mask_distance = distance < RADIUS_EARTH
        cosine = (
            (distance**2 + orbital_radius**2 - RADIUS_EARTH**2)
            / (2 * distance.clamp_min(1e-6) * orbital_radius)
        )
        mask_off_nadir = cosine > math.cos(MAX_OFF_NADIR_ANGLE)
        visibility_trace[start:stop] = (
            mask_distance & mask_off_nadir & mask_sensor
        ).any(1)

    return visibility_trace


def _satellite_positions_eci(
    satellites: Satellites,
    times: torch.Tensor,
) -> torch.Tensor:
    positions: list[torch.Tensor] = []
    for satellite in satellites:
        orbit = satellite.orbit
        eccentricity = orbit.eccentricity
        semi_major_axis = orbit.semi_major_axis
        inclination = math.radians(orbit.inclination)
        raan = math.radians(orbit.right_ascension_of_the_ascending_node)
        argument_of_perigee = math.radians(orbit.argument_of_perigee)
        true_anomaly = math.radians(satellite.true_anomaly)

        initial_eccentric_anomaly = 2 * math.atan2(
            math.sqrt(1 - eccentricity) * math.sin(true_anomaly / 2),
            math.sqrt(1 + eccentricity) * math.cos(true_anomaly / 2),
        )
        initial_mean_anomaly = (
            initial_eccentric_anomaly
            - eccentricity * math.sin(initial_eccentric_anomaly)
        )
        mean_motion = math.sqrt(MU_EARTH / semi_major_axis**3)
        mean_anomaly = (
            initial_mean_anomaly + mean_motion * times
        ).remainder(2 * math.pi)

        eccentric_anomaly = mean_anomaly.clone()
        for _ in range(5):
            eccentric_anomaly = eccentric_anomaly - (
                eccentric_anomaly
                - eccentricity * torch.sin(eccentric_anomaly)
                - mean_anomaly
            ) / (1 - eccentricity * torch.cos(eccentric_anomaly))

        x_perifocal = semi_major_axis * (
            torch.cos(eccentric_anomaly) - eccentricity
        )
        y_perifocal = (
            semi_major_axis
            * math.sqrt(1 - eccentricity**2)
            * torch.sin(eccentric_anomaly)
        )

        cos_raan = math.cos(raan)
        sin_raan = math.sin(raan)
        cos_arg = math.cos(argument_of_perigee)
        sin_arg = math.sin(argument_of_perigee)
        cos_inc = math.cos(inclination)
        sin_inc = math.sin(inclination)

        x_eci = (
            (cos_raan * cos_arg - sin_raan * sin_arg * cos_inc)
            * x_perifocal
            + (-cos_raan * sin_arg - sin_raan * cos_arg * cos_inc)
            * y_perifocal
        )
        y_eci = (
            (sin_raan * cos_arg + cos_raan * sin_arg * cos_inc)
            * x_perifocal
            + (-sin_raan * sin_arg + cos_raan * cos_arg * cos_inc)
            * y_perifocal
        )
        z_eci = (
            sin_arg * sin_inc * x_perifocal
            + cos_arg * sin_inc * y_perifocal
        )
        positions.append(torch.stack([x_eci, y_eci, z_eci], dim=-1))

    return torch.stack(positions, dim=1)


def _task_positions_eci(
    taskset_ecef: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    theta = EARTH_ROTATION_RATE * times
    cos_theta = torch.cos(theta).unsqueeze(1)
    sin_theta = torch.sin(theta).unsqueeze(1)

    x_ecef = taskset_ecef[:, 0].unsqueeze(0)
    y_ecef = taskset_ecef[:, 1].unsqueeze(0)
    z_ecef = taskset_ecef[:, 2].unsqueeze(0).expand(len(times), -1)

    x_eci = cos_theta * x_ecef - sin_theta * y_ecef
    y_eci = sin_theta * x_ecef + cos_theta * y_ecef
    return torch.stack([x_eci, y_eci, z_ecef], dim=-1)


def _geometric_accessibility(
    environment: object,
    taskset: TaskSet,
) -> torch.Tensor:
    """按 OptimalAlgorithm 的几何约束检查卫星-任务是否可达。"""
    earth_rotation = environment.get_earth_rotation()  # type: ignore[attr-defined]
    constellation = environment.get_constellation()  # type: ignore[attr-defined]

    taskset_eci = (
        earth_rotation.new_tensor(taskset.coordinates_ecef)
        @ earth_rotation
    )
    constellation_eci = constellation.coordinates_eci
    distance = torch.norm(
        taskset_eci.unsqueeze(0) - constellation_eci.unsqueeze(1),
        dim=2,
    )
    orbital_radius = constellation_eci.norm(dim=1).unsqueeze(1)

    mask_distance = distance < RADIUS_EARTH
    cosine = (
        (distance**2 + orbital_radius**2 - RADIUS_EARTH**2)
        / (2 * distance * orbital_radius)
    )
    mask_off_nadir = cosine > math.cos(MAX_OFF_NADIR_ANGLE)

    satellite_sensor_type = torch.tensor([
        satellite.sensor.type_ for satellite in constellation.sort()
    ])
    task_sensor_type = torch.tensor([task.sensor_type for task in taskset])
    mask_sensor = satellite_sensor_type.unsqueeze(1) == task_sensor_type

    return mask_distance & mask_off_nadir & mask_sensor


def sample_observable_taskset(
    constellation: Constellation,
    n: int,
    *,
    oversample_factor: int = 5,
    max_rounds: int = 10,
    max_time_step: int = MAX_TIME_STEP,
) -> TaskSet:
    """反复过采样任务，保留物理上存在观测机会的点位。"""
    kept: list[Task] = []

    for _ in range(max_rounds):
        remaining = n - len(kept)
        if remaining <= 0:
            break

        candidate_count = max(remaining * oversample_factor, remaining)
        candidates = TaskSet.sample(candidate_count)
        observable_flags = scan_observable_task_flags(
            constellation,
            candidates,
            max_time_step=max_time_step,
        )
        kept.extend(
            task for task, observable
            in zip(candidates, observable_flags)
            if observable
        )

    if len(kept) < n:
        raise RuntimeError(
            f'Only sampled {len(kept)} observable tasks, expected {n}. '
            'Increase oversample_factor/max_rounds or inspect the sampled '
            'constellation.'
        )

    return renumber_taskset(TaskSet(kept[:n]))


def generate_constellations_and_tasksets(
    split: str,
    n: int,
    *,
    filter_observable: bool = True,
) -> None:
    satellites = load_satellites(split)

    constellations_root = CONSTELLATIONS_ROOT / split
    tasks_root = TASKSETS_ROOT / split
    for i in trange(n):
        constellation_path = (
            constellations_root / f'{i // 1000:02}/{i:05}.json'
        )
        if not constellation_path.exists():
            constellation_path.parent.mkdir(parents=True, exist_ok=True)
            constellation = Constellation.sample(
                satellites,
                random.randint(1, 50),
            )
            constellation.dump(str(constellation_path))
        else:
            constellation = Constellation.load(str(constellation_path))

        taskset_path = tasks_root / f'{i // 1000:02}/{i:05}.json'
        if not taskset_path.exists():
            taskset_path.parent.mkdir(parents=True, exist_ok=True)
            num_tasks = random.randint(50, 300)
            if filter_observable:
                taskset = sample_observable_taskset(constellation, num_tasks)
            else:
                taskset = TaskSet.sample(num_tasks)
            taskset.dump(str(taskset_path))


def main() -> None:
    generate_constellations_and_tasksets('train', 100_000)
    generate_constellations_and_tasksets('val_seen', 500)
    generate_constellations_and_tasksets('val_unseen', 500)
    generate_constellations_and_tasksets('test', 1_000)


if __name__ == '__main__':
    main()
