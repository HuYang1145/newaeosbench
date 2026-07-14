"""运行已训练的 AEOS 模型并保存候选轨迹。

属于 Stage-2/Stage-3 工作流工具。加载 checkpoint，在指定划分上通过 Basilisk
场景运行贪心或 seeded top-k 采样策略，输出轨迹 ``.pth`` 和指标
``.json``。默认仍为原始贪心行为。
"""

import argparse
import os
import pathlib
from typing import Literal

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
    parser.add_argument(
        '--strategy',
        choices=['greedy', 'top_k_sample'],
        default='greedy',
    )
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=3407)
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


def select_action_indices(
    logits: torch.Tensor,
    *,
    strategy: Literal['greedy', 'top_k_sample'],
    top_k: int = 3,
    temperature: float = 0.7,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """从含 null 动作的 logits 中选择索引。

    ``top_k_sample`` 只在每颗卫星的前 ``k`` 个候选中采样，避免把
    高不确定性变成接近随机的全任务搜索。
    """

    if strategy == 'greedy':
        return logits.argmax(-1)
    if strategy != 'top_k_sample':
        raise ValueError(f'unsupported candidate strategy: {strategy}')
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    if temperature <= 0:
        raise ValueError('temperature must be positive')

    k = min(top_k, logits.shape[-1])
    top_values, top_indices = logits.topk(k, dim=-1)
    probabilities = (top_values / temperature).softmax(-1)
    sampled_offsets = torch.multinomial(
        probabilities.reshape(-1, k),
        num_samples=1,
        generator=generator,
    ).reshape(*probabilities.shape[:-1], 1)
    return top_indices.gather(-1, sampled_offsets).squeeze(-1)


def write_rollout_metadata(
    output_root: pathlib.Path,
    *,
    metadata: dict[str, object],
    rank: int,
) -> None:
    """只由 rank 0 写共享候选目录的运行元数据。"""

    if rank != 0:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    json_dump(metadata, str(output_root / 'rollout_metadata.json'))


class GreedyModelAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        *args,
        checkpoint: pathlib.Path,
        device: torch.device,
        statistics: Statistics,
        strategy: Literal['greedy', 'top_k_sample'] = 'greedy',
        top_k: int = 3,
        temperature: float = 0.7,
        seed: int = 3407,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._device = device
        self._statistics = statistics
        self._strategy = strategy
        self._top_k = top_k
        self._temperature = temperature
        self._generator = torch.Generator(device=device).manual_seed(seed)

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
        relative_task_ids = select_action_indices(
            logits,
            strategy=self._strategy,
            top_k=self._top_k,
            temperature=self._temperature,
            generator=self._generator,
        ).squeeze(0).cpu() - 1

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
    strategy: Literal['greedy', 'top_k_sample'] = 'greedy',
    top_k: int = 3,
    temperature: float = 0.7,
    seed: int = 3407,
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
        strategy=strategy,
        top_k=top_k,
        temperature=temperature,
        seed=seed,
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

    write_rollout_metadata(
        args.output_root,
        metadata={
            'checkpoint': str(args.checkpoint),
            'split': args.split,
            'strategy': args.strategy,
            'top_k': args.top_k,
            'temperature': args.temperature,
            'seed': args.seed,
        },
        rank=rank,
    )

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
            strategy=args.strategy,
            top_k=args.top_k,
            temperature=args.temperature,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
