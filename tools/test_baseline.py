import argparse
import multiprocessing
import pathlib
from functools import partial
import numpy as np

import todd
from todd.patches.py_ import json_dump, json_load

from constellation import CONSTELLATIONS_ROOT, TASKSETS_ROOT, TRAJECTORIES_ROOT
from constellation.algorithms import OptimalAlgorithm
from constellation import ANNOTATIONS_ROOT
from constellation.algorithms.replay import ReplayAlgorithm
from constellation.controller import Controller
from constellation.data import Constellation, Task, TaskSet
from constellation.environments import BasiliskEnvironment
from constellation.evaluators import (
    CompletionRateEvaluator,
    PowerUsageEvaluator,
    TurnAroundTimeEvaluator,
)
from constellation import TaskManager
from constellation.callbacks import ComposedCallback

MAX_RETRY = 1


def test(work_dir: pathlib.Path, split: str, i: int) -> list[float] | None:
    todd.logger.info(f"{split=} {i=}")

    path = f'{split}/{i // 1000:02}/{i:05}.json'
    constellation_path = CONSTELLATIONS_ROOT / path
    taskset_path = TASKSETS_ROOT / path
    result_path = work_dir / path

    if result_path.exists():
        todd.logger.info(f'{split=} {i=} already exists')
        return None

    taskset: TaskSet[Task] = TaskSet.load(str(taskset_path))
    constellation = Constellation.load(str(constellation_path))

    result_path.parent.mkdir(parents=True, exist_ok=True)

    environment = BasiliskEnvironment(
        constellation=constellation,
        all_tasks=taskset,
    )
    algorithm = OptimalAlgorithm(timer=environment.timer)

    task_manager = TaskManager(timer=environment.timer, taskset=taskset)

    evaluators = [
        CompletionRateEvaluator(),
        TurnAroundTimeEvaluator(),
        PowerUsageEvaluator(),
    ]

    controller = Controller(
        name="baseline",
        environment=environment,
        task_manager=task_manager,
        callbacks=ComposedCallback(callbacks=evaluators),
    )

    algorithm.prepare(environment, task_manager)
    evaluators[0].before_run()
    evaluators[1].before_run()
    evaluators[2].before_run()

    controller.run(algorithm)

    evaluators[0].after_run()
    evaluators[1].after_run()
    evaluators[2].after_run()

    metrics = controller.memo

    # Convert tensors to native Python types and filter out non-serializable objects
    metrics_serializable = {}
    for k, v in metrics.items():
        # Skip non-serializable objects
        if k in ['algorithm', 'actions', 'assignment', 'is_visible']:
            continue
        if hasattr(v, 'tolist'):
            metrics_serializable[k] = v.tolist()
        elif hasattr(v, 'item'):
            metrics_serializable[k] = v.item()
        elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
            metrics_serializable[k] = v

    json_dump(metrics_serializable, str(result_path))

    todd.logger.info(f"{split=} {i=} {metrics=}")
    return metrics


def parallel_test(
    work_dir: pathlib.Path,
    num_workers: int,
    split: str,
) -> None:
    annotations_path = ANNOTATIONS_ROOT / f'{split}.tiny.json'
    annotations: list[int] = json_load(str(annotations_path))

    if num_workers == 0:
        for i in annotations:
            test(work_dir, split, i)
        return

    with multiprocessing.Pool(num_workers) as pool:
        list(
            pool.imap_unordered(
                partial(test, work_dir, split),
                annotations,
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate trajectories')
    parser.add_argument('name')
    parser.add_argument('num_workers', type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    work_dir: pathlib.Path = (
        pathlib.Path('work_dirs/test_baseline') / args.name
    )

    parallel_test(work_dir, args.num_workers, 'val_seen')
    parallel_test(work_dir, args.num_workers, 'val_unseen')
    parallel_test(work_dir, args.num_workers, 'test')


if __name__ == '__main__':
    main()
