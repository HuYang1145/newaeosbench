"""生成可观测性过滤后的 tasksets。

该脚本用于旧 checkpoint 重新评估：复用已有 constellations，只替换
评估 split 的 tasksets。每个场景沿用旧 taskset 的任务数量，从而尽量只
改变“任务点位是否物理可观测”这一变量。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from todd.patches.py_ import json_load
import torch
from tqdm import tqdm

from constellation import ANNOTATIONS_ROOT, CONSTELLATIONS_ROOT, TASKSETS_ROOT
from constellation.data import Constellation, TaskSet
from tools.generate_constellations_and_tasksets import (
    sample_observable_taskset,
)

FULL_SPLIT_SIZES = {
    'val_seen': 500,
    'val_unseen': 500,
    'test': 1_000,
}


def _load_annotation_ids(split: str) -> list[int]:
    annotation = json_load(str(ANNOTATIONS_ROOT / f'{split}.json'))
    if isinstance(annotation, dict):
        return list(annotation['ids'])
    return list(annotation)


def _load_ids(split: str, id_source: str) -> list[int]:
    if id_source == 'annotation':
        return _load_annotation_ids(split)
    if id_source == 'full':
        return list(range(FULL_SPLIT_SIZES[split]))
    raise ValueError(f'Unknown id_source: {id_source}')


def _taskset_path(root: Path, split: str, id_: int) -> Path:
    return root / split / f'{id_ // 1000:02}' / f'{id_:05}.json'


def generate_split(
    split: str,
    *,
    source_tasksets_root: Path,
    output_tasksets_root: Path,
    seed: int,
    oversample_factor: int,
    max_rounds: int,
    limit: int | None,
    overwrite: bool,
    id_source: str,
    rank: int,
    world_size: int,
) -> None:
    ids = _load_ids(split, id_source)
    if limit is not None:
        ids = ids[:limit]
    ids = ids[rank::world_size]

    split_seed_offset = sum(ord(ch) for ch in split) * 1_000_000
    for id_ in tqdm(ids, desc=f'generate {split} tasksets'):
        output_path = _taskset_path(output_tasksets_root, split, id_)
        if output_path.exists() and not overwrite:
            continue

        source_path = _taskset_path(source_tasksets_root, split, id_)
        old_taskset = TaskSet.load(str(source_path))
        num_tasks = len(old_taskset)

        constellation_path = (
            CONSTELLATIONS_ROOT / split / f'{id_ // 1000:02}' / f'{id_:05}.json'
        )
        constellation = Constellation.load(str(constellation_path))

        random.seed(seed + split_seed_offset + id_)
        taskset = sample_observable_taskset(
            constellation,
            num_tasks,
            oversample_factor=oversample_factor,
            max_rounds=max_rounds,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        taskset.dump(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['val_seen', 'val_unseen', 'test'],
    )
    parser.add_argument(
        '--source-tasksets-root',
        type=Path,
        required=True,
        help='旧 tasksets 根目录，用于读取每个场景原任务数量。',
    )
    parser.add_argument(
        '--output-tasksets-root',
        type=Path,
        default=TASKSETS_ROOT,
    )
    parser.add_argument('--seed', type=int, default=20260622)
    parser.add_argument('--oversample-factor', type=int, default=10)
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument(
        '--id-source',
        choices=['annotation', 'full'],
        default='annotation',
        help='annotation 只生成当前评估索引；full 生成完整评估 split。',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert 0 <= args.rank < args.world_size
    torch.set_num_threads(1)
    for split in args.splits:
        generate_split(
            split,
            source_tasksets_root=args.source_tasksets_root,
            output_tasksets_root=args.output_tasksets_root,
            seed=args.seed,
            oversample_factor=args.oversample_factor,
            max_rounds=args.max_rounds,
            limit=args.limit,
            overwrite=args.overwrite,
            id_source=args.id_source,
            rank=args.rank,
            world_size=args.world_size,
        )


if __name__ == '__main__':
    main()
