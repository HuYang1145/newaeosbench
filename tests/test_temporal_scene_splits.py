import json
from pathlib import Path

import pytest

from tools.audit_temporal_scene_splits import audit_scene_splits


def _write_annotation(path: Path, ids: list[int]) -> None:
    path.write_text(json.dumps({
        'ids': ids,
        'epochs': [1] * len(ids),
    }))


def _write_scene(
    root: Path,
    *,
    split: str,
    scene_id: int,
    constellation: dict,
    taskset: list[dict],
) -> None:
    relative = Path(split) / f'{scene_id // 1000:02}'
    constellation_path = (
        root / 'constellations' / relative / f'{scene_id:05}.json'
    )
    taskset_path = root / 'tasksets' / relative / f'{scene_id:05}.json'
    constellation_path.parent.mkdir(parents=True, exist_ok=True)
    taskset_path.parent.mkdir(parents=True, exist_ok=True)
    constellation_path.write_text(json.dumps(constellation))
    taskset_path.write_text(json.dumps(taskset))


def test_scene_split_audit_detects_exact_static_scene_overlap(
    tmp_path: Path,
) -> None:
    train_annotation = tmp_path / 'train.json'
    val_annotation = tmp_path / 'val.json'
    _write_annotation(train_annotation, [7])
    _write_annotation(val_annotation, [19])
    constellation = {'satellites': [{'id': 0, 'mass': 10}]}
    taskset = [{'id': 0, 'duration': 3}]
    _write_scene(
        tmp_path,
        split='train',
        scene_id=7,
        constellation=constellation,
        taskset=taskset,
    )
    _write_scene(
        tmp_path,
        split='val_seen',
        scene_id=19,
        constellation=constellation,
        taskset=taskset,
    )

    result = audit_scene_splits(
        {
            'train': (train_annotation, 'train'),
            'val_seen': (val_annotation, 'val_seen'),
        },
        constellation_root=tmp_path / 'constellations',
        taskset_root=tmp_path / 'tasksets',
    )

    overlap = result['overlaps']['train__val_seen']
    assert result['scene_level_disjoint'] is False
    assert overlap['exact_scene_count'] == 1
    assert overlap['exact_scene_pairs'] == [{
        'left_scene_id': 7,
        'right_scene_id': 19,
    }]


def test_scene_split_audit_distinguishes_shared_constellation_from_scene(
    tmp_path: Path,
) -> None:
    train_annotation = tmp_path / 'train.json'
    val_annotation = tmp_path / 'val.json'
    _write_annotation(train_annotation, [7])
    _write_annotation(val_annotation, [19])
    constellation = {'satellites': [{'id': 0, 'mass': 10}]}
    _write_scene(
        tmp_path,
        split='train',
        scene_id=7,
        constellation=constellation,
        taskset=[{'id': 0, 'duration': 3}],
    )
    _write_scene(
        tmp_path,
        split='val_seen',
        scene_id=19,
        constellation=constellation,
        taskset=[{'id': 0, 'duration': 8}],
    )

    result = audit_scene_splits(
        {
            'train': (train_annotation, 'train'),
            'val_seen': (val_annotation, 'val_seen'),
        },
        constellation_root=tmp_path / 'constellations',
        taskset_root=tmp_path / 'tasksets',
    )

    overlap = result['overlaps']['train__val_seen']
    assert result['scene_level_disjoint'] is True
    assert overlap['exact_scene_count'] == 0
    assert overlap['shared_constellation_count'] == 1


def test_scene_split_audit_rejects_duplicate_annotation_ids(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / 'duplicates.json'
    _write_annotation(annotation, [7, 7])

    with pytest.raises(ValueError, match='duplicate scene ids'):
        audit_scene_splits(
            {'train': (annotation, 'train')},
            constellation_root=tmp_path / 'constellations',
            taskset_root=tmp_path / 'tasksets',
        )


def test_scene_split_audit_reports_missing_static_files(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / 'missing.json'
    _write_annotation(annotation, [7])

    with pytest.raises(FileNotFoundError, match='constellations'):
        audit_scene_splits(
            {'train': (annotation, 'train')},
            constellation_root=tmp_path / 'constellations',
            taskset_root=tmp_path / 'tasksets',
        )
