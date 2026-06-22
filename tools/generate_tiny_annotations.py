"""创建小型划分标注文件，用于快速基线测试/调试。

从每个完整标注划分中取前几个 ID，写入 ``*.tiny.json`` 文件，
并记录这些小子集上已有轨迹的平均指标。
"""

import argparse
import todd
import torch
from todd.patches.py_ import json_dump, json_load
from tqdm import trange

from constellation import ANNOTATIONS_ROOT, TRAJECTORIES_ROOT


def generate(split: str, n: int) -> None:
    annotations: list[int] = json_load(str(ANNOTATIONS_ROOT / f'{split}.json'))
    annotations = annotations[:n]

    if not todd.Store.DRY_RUN:
        json_dump(annotations, str(ANNOTATIONS_ROOT / f'{split}.tiny.json'))

    metrics: list[float] = []

    trajectories_root = TRAJECTORIES_ROOT / split
    for i in annotations:
        trajectory_path = trajectories_root / f'{i // 1000:02}/{i:05}.json'
        metrics.append(json_load(str(trajectory_path)))

    todd.logger.info(
        "%s average completion rate: %s",
        split,
        torch.tensor(metrics).mean(0),
    )


def main() -> None:
    ANNOTATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    generate('train', 256)
    generate('val_seen', 64)
    generate('val_unseen', 64)
    generate('test', 64)


if __name__ == '__main__':
    main()
