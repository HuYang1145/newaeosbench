#!/usr/bin/env python3
"""审计 Temporal Adapter 训练/验证 annotation 的 scene-level 隔离。"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class SceneFingerprint:
    scene_id: int
    split: str
    constellation_sha256: str
    taskset_sha256: str
    scene_sha256: str


def _canonical_json_bytes(path: Path) -> bytes:
    payload = json.loads(path.read_text())
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _annotation_records(
    annotation_file: Path,
    *,
    split: str,
    constellation_root: Path,
    taskset_root: Path,
) -> list[SceneFingerprint]:
    annotation = json.loads(annotation_file.read_text())
    if not isinstance(annotation, dict):
        raise TypeError('annotation must be an object')
    ids = annotation.get('ids')
    epochs = annotation.get('epochs')
    if not isinstance(ids, list) or not isinstance(epochs, list):
        raise TypeError('annotation ids and epochs must be lists')
    if len(ids) != len(epochs):
        raise ValueError('annotation ids and epochs must have equal length')
    normalized_ids = [int(value) for value in ids]
    duplicates = sorted(
        scene_id
        for scene_id, count in Counter(normalized_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f'annotation has duplicate scene ids: {duplicates}')

    records = []
    missing = []
    for scene_id in normalized_ids:
        relative = (
            Path(split) / f'{scene_id // 1000:02}' / f'{scene_id:05}.json'
        )
        constellation_path = constellation_root / relative
        taskset_path = taskset_root / relative
        scene_missing = False
        for path in (constellation_path, taskset_path):
            if not path.is_file():
                missing.append(path)
                scene_missing = True
        if scene_missing:
            continue
        constellation_bytes = _canonical_json_bytes(constellation_path)
        taskset_bytes = _canonical_json_bytes(taskset_path)
        records.append(SceneFingerprint(
            scene_id=scene_id,
            split=split,
            constellation_sha256=_sha256(constellation_bytes),
            taskset_sha256=_sha256(taskset_bytes),
            scene_sha256=_sha256(
                constellation_bytes + b'\0' + taskset_bytes
            ),
        ))
    if missing:
        preview = ', '.join(str(path) for path in missing[:5])
        suffix = '' if len(missing) <= 5 else f' (+{len(missing) - 5} more)'
        raise FileNotFoundError(f'missing static scene files: {preview}{suffix}')
    return records


def _by_fingerprint(
    records: list[SceneFingerprint],
    attribute: str,
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[getattr(record, attribute)].append(record.scene_id)
    return grouped


def _overlap(
    left: list[SceneFingerprint],
    right: list[SceneFingerprint],
) -> dict[str, Any]:
    left_scenes = _by_fingerprint(left, 'scene_sha256')
    right_scenes = _by_fingerprint(right, 'scene_sha256')
    shared_scenes = sorted(set(left_scenes) & set(right_scenes))
    exact_pairs = [
        {
            'left_scene_id': left_id,
            'right_scene_id': right_id,
        }
        for fingerprint in shared_scenes
        for left_id in sorted(left_scenes[fingerprint])
        for right_id in sorted(right_scenes[fingerprint])
    ]
    left_constellations = _by_fingerprint(left, 'constellation_sha256')
    right_constellations = _by_fingerprint(right, 'constellation_sha256')
    left_tasksets = _by_fingerprint(left, 'taskset_sha256')
    right_tasksets = _by_fingerprint(right, 'taskset_sha256')
    return {
        'exact_scene_count': len(shared_scenes),
        'exact_scene_pairs': exact_pairs,
        'shared_constellation_count': len(
            set(left_constellations) & set(right_constellations)
        ),
        'shared_taskset_count': len(
            set(left_tasksets) & set(right_tasksets)
        ),
    }


def audit_scene_splits(
    datasets: dict[str, tuple[Path, str]],
    *,
    constellation_root: Path,
    taskset_root: Path,
) -> dict[str, Any]:
    """比较完整静态场景指纹；共享卫星但任务集不同不算 scene 泄漏。"""
    records = {
        name: _annotation_records(
            annotation_file,
            split=split,
            constellation_root=constellation_root,
            taskset_root=taskset_root,
        )
        for name, (annotation_file, split) in datasets.items()
    }
    overlaps = {
        f'{left_name}__{right_name}': _overlap(
            records[left_name],
            records[right_name],
        )
        for left_name, right_name in itertools.combinations(records, 2)
    }
    return {
        'purpose': (
            'scene-level static input isolation audit; '
            'shared constellation alone is not treated as scene leakage'
        ),
        'datasets': {
            name: {
                'annotation_file': str(datasets[name][0]),
                'split': datasets[name][1],
                'scene_count': len(values),
                'unique_scene_count': len({
                    record.scene_id for record in values
                }),
                'scenes': [dataclasses.asdict(record) for record in values],
            }
            for name, values in records.items()
        },
        'overlaps': overlaps,
        'scene_level_disjoint': all(
            values['exact_scene_count'] == 0
            for values in overlaps.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--train-annotation',
        type=Path,
        default=Path(
            'data/annotations/train_paper_stage3_tau_e_existing.json'
        ),
    )
    parser.add_argument(
        '--val-seen-annotation',
        type=Path,
        default=Path('data/annotations/val_seen.json'),
    )
    parser.add_argument(
        '--val-unseen-annotation',
        type=Path,
        default=Path('data/annotations/val_unseen.json'),
    )
    parser.add_argument(
        '--constellation-root',
        type=Path,
        default=Path('data/constellations'),
    )
    parser.add_argument(
        '--taskset-root',
        type=Path,
        default=Path('data/tasksets'),
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_scene_splits(
        {
            'train': (args.train_annotation, 'train'),
            'val_seen': (args.val_seen_annotation, 'val_seen'),
            'val_unseen': (
                args.val_unseen_annotation,
                'val_unseen',
            ),
        },
        constellation_root=args.constellation_root,
        taskset_root=args.taskset_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(
        f'[done] scene_level_disjoint={result["scene_level_disjoint"]} '
        f'output={args.output}',
        flush=True,
    )


if __name__ == '__main__':
    main()
