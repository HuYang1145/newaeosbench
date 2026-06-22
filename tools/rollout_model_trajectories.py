"""运行已训练的 AEOS 模型并保存候选轨迹。

属于 Stage-2/Stage-3 工作流工具。加载 checkpoint，在指定划分上通过 Basilisk
场景运行贪心模型策略，输出轨迹 ``.pth`` 文件和指标 ``.json`` 文件供后续 tau_e 过滤使用。
"""

import argparse
import os
import pathlib

import torch
from todd.patches.py_ import json_dump, json_load

from constellation import (
    ANNOTATIONS_ROOT,
    CONSTELLATIONS_ROOT,
    MAX_TIME_STEP,
    STATISTICS_PATH,
    TASKSETS_ROOT,
)
from constellation.algorithms.base import BaseAlgorithm
from constellation.callbacks import ComposedCallback
from constellation.controller import Controller
from constellation.data import Action, Actions, Constellation, Task, TaskSet
from constellation.environments import BasiliskEnvironment, BaseEnvironment
from constellation.evaluators import (
    CompletionRateEvaluator,
    PowerUsageEvaluator,
    TurnAroundTimeEvaluator,
)
from constellation.loggers import TrajectoryLogger
from constellation.new_transformers import Model, Statistics
from constellation.task_managers import TaskManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Roll out a trained actor and save trajectories for stage 2',
    )
    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('output_root', type=pathlib.Path)
    parser.add_argument('--split', default='train')
    parser.add_argument('--annotation-file', type=pathlib.Path, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_state_dict(path: pathlib.Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f'Unsupported checkpoint format: {type(checkpoint)!r}')


class GreedyModelAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        *args,
        checkpoint: pathlib.Path,
        device: torch.device,
        statistics: Statistics,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._device = device
        self._statistics = statistics

        model = Model()
        missing = model.load_state_dict(
            load_state_dict(checkpoint),
            strict=False,
        )
        if missing.missing_keys:
            print(f'[rollout] missing keys: {len(missing.missing_keys)}')
        if missing.unexpected_keys:
            print(f'[rollout] unexpected keys: {len(missing.unexpected_keys)}')
        self._model = model.to(device)
        self._model.eval()

    def prepare(
        self,
        environment: BaseEnvironment,
        task_manager: TaskManager,
    ) -> None:
        self._task_manager = task_manager

    def _build_inputs(
        self,
        taskset: TaskSet[Task],
        constellation: Constellation,
    ) -> tuple[torch.Tensor, ...]:
        sensor_type, static_data = constellation.static_to_tensor()
        sensor_enabled, dynamic_data = constellation.dynamic_to_tensor()
        constellation_data = torch.cat([static_data, dynamic_data], -1)
        constellation_data = (
            (constellation_data - self._statistics.constellation_mean) /
            (self._statistics.constellation_std + 1e-6)
        )

        task_sensor_type, task_static_data = taskset.to_tensor()
        task_static_data = task_static_data.clone()
        task_static_data[..., 0] -= self._timer.time
        task_static_data[..., 1] -= self._timer.time
        task_progress = self._task_manager.progress[self._task_manager.ongoing_flags]
        task_dynamic_data = task_progress.unsqueeze(-1)
        task_data = torch.cat([task_static_data, task_dynamic_data], -1)
        task_data = (
            (task_data - self._statistics.taskset_mean) /
            (self._statistics.taskset_std + 1e-6)
        )

        return (
            torch.tensor([self._timer.time], dtype=torch.long, device=self._device),
            (sensor_type - 1).unsqueeze(0).to(self._device),
            sensor_enabled.unsqueeze(0).to(self._device),
            constellation_data.unsqueeze(0).float().to(self._device),
            torch.ones((1, len(constellation)), dtype=torch.bool, device=self._device),
            (task_sensor_type - 1).unsqueeze(0).to(self._device),
            task_data.unsqueeze(0).float().to(self._device),
            torch.ones((1, len(taskset)), dtype=torch.bool, device=self._device),
        )

    def step(
        self,
        taskset: TaskSet[Task],
        constellation: Constellation,
        earth_rotation: torch.Tensor,
    ) -> tuple[Actions, list[int]]:
        del earth_rotation

        if len(taskset) == 0:
            assignment = [-1] * len(constellation)
            actions = Actions(
                Action(
                    toggle=satellite.sensor.enabled,
                    target_location=None,
                ) for satellite in constellation.sort()
            )
            return actions, assignment

        with torch.inference_mode():
            logits = self._model.predict(*self._build_inputs(taskset, constellation))
        relative_task_ids = logits.argmax(-1).squeeze(0).cpu() - 1

        task_ids = taskset.ids
        assignment = torch.where(
            relative_task_ids >= 0,
            task_ids[relative_task_ids.clamp_min(0)],
            -1,
        ).tolist()

        actions = Actions(
            Action(
                toggle=((task_id == -1 and satellite.sensor.enabled) or
                        (task_id != -1 and not satellite.sensor.enabled)),
                target_location=(
                    None if relative_task_id == -1
                    else taskset[int(relative_task_id)].coordinate
                ),
            ) for satellite, task_id, relative_task_id in zip(
                constellation.sort(),
                assignment,
                relative_task_ids.tolist(),
            )
        )
        return actions, assignment


def load_annotation_ids(path: pathlib.Path) -> list[int]:
    payload = json_load(str(path))
    if isinstance(payload, dict):
        return list(payload['ids'])
    return list(payload)


def rollout_one(
    *,
    split: str,
    id_: int,
    checkpoint: pathlib.Path,
    device: torch.device,
    output_root: pathlib.Path,
    statistics: Statistics,
    overwrite: bool,
) -> None:
    trajectory_dir = output_root / split / f'{id_ // 1000:02}'
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = trajectory_dir / f'{id_:05}.json'
    trajectory_path = trajectory_dir / f'{id_:05}.pth'

    if not overwrite and metrics_path.exists() and trajectory_path.exists():
        print(f'[rollout] skip existing split={split} id={id_}')
        return

    constellation = Constellation.load(
        str(CONSTELLATIONS_ROOT / split / f'{id_ // 1000:02}' / f'{id_:05}.json'),
    )
    taskset: TaskSet[Task] = TaskSet.load(
        str(TASKSETS_ROOT / split / f'{id_ // 1000:02}' / f'{id_:05}.json'),
    )

    environment = BasiliskEnvironment(
        start_time=0,
        constellation=constellation,
        all_tasks=taskset,
    )
    task_manager = TaskManager(timer=environment.timer, taskset=taskset)
    callbacks = ComposedCallback(
        callbacks=[
            CompletionRateEvaluator(),
            TurnAroundTimeEvaluator(),
            PowerUsageEvaluator(),
            TrajectoryLogger(work_dir=trajectory_dir),
        ],
    )
    controller = Controller(
        f'{id_:05}',
        environment=environment,
        task_manager=task_manager,
        callbacks=callbacks,
    )
    algorithm = GreedyModelAlgorithm(
        timer=environment.timer,
        checkpoint=checkpoint,
        device=device,
        statistics=statistics,
    )
    algorithm.prepare(environment=environment, task_manager=task_manager)
    controller.run(algorithm, progress_bar=False)

    metrics = controller.memo['metrics']
    metrics['PC_Wh'] = metrics['PC'] / 3600.0
    json_dump(metrics, str(metrics_path))
    print(f'[rollout] split={split} id={id_} metrics={metrics}')


def main() -> None:
    args = parse_args()
    annotation_path = args.annotation_file
    if annotation_path is None:
        annotation_path = ANNOTATIONS_ROOT / f'{args.split}.json'

    ids = load_annotation_ids(annotation_path)
    if args.limit is not None:
        ids = ids[:args.limit]

    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    statistics: Statistics = torch.load(
        STATISTICS_PATH,
        map_location='cpu',
        weights_only=False,
    )
    device = torch.device(args.device)

    for index, id_ in enumerate(ids):
        if index % world_size != rank:
            continue
        rollout_one(
            split=args.split,
            id_=id_,
            checkpoint=args.checkpoint,
            device=device,
            output_root=args.output_root,
            statistics=statistics,
            overwrite=args.overwrite,
        )


if __name__ == '__main__':
    main()
