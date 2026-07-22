import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from constellation.new_transformers.event_v2.model import EventJointActorCritic
from constellation.new_transformers.event_v2.transition import (
    transition_schema_fingerprint,
)
from constellation.new_transformers.model import Model
from tools.evaluate_event_v2_unseen_offline import (
    aggregate_weighted_losses,
    audit_training_checkpoint,
    batch_supports,
    build_summary,
    build_paired_models,
    cuda_memory_snapshot,
    decide_acceptance,
    evaluate_dataset,
    write_json_atomic,
)


LOSS_WEIGHTS = {
    'task_distillation': 1.0,
    'termination': 2.0,
    'commitment': 3.0,
    'value': 4.0,
}


def _model_kwargs() -> dict[str, object]:
    return {
        'event_width': 8,
        'sensor_type_embedding_dim': 4,
        'tasks_data_embedding_dim': 4,
        'encoder_width': 8,
        'encoder_depth': 1,
        'encoder_num_heads': 2,
        'sensor_enabled_embedding_dim': 4,
        'constellation_data_embedding_dim': 4,
        'decoder_width': 8,
        'decoder_depth': 1,
        'decoder_num_heads': 2,
        'use_constraint_module': False,
        'use_sdpa': False,
        'freeze_backbone': True,
    }


def _pair_checkpoints(tmp_path, *, fingerprint: str = 'a' * 64):
    kwargs = _model_kwargs()
    stage3_kwargs = dict(kwargs)
    stage3_kwargs.pop('event_width')
    stage3_kwargs.pop('freeze_backbone')
    stage3 = Model(**stage3_kwargs)
    stage3_path = tmp_path / 'stage3.pth'
    torch.save({'model': stage3.state_dict()}, stage3_path)

    trained = EventJointActorCritic(**kwargs)
    trained.load_stage3_checkpoint(stage3_path)
    with torch.no_grad():
        trained.actor.idle_head.weight.add_(0.25)
    checkpoint = {
        'checkpoint_version': 1,
        'stage': 'V2-0',
        'steps': 10_000,
        'transition_schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': fingerprint,
        'model': trained.state_dict(),
    }
    trained_path = tmp_path / 'trained.pth'
    torch.save(checkpoint, trained_path)
    return stage3_path, trained_path, checkpoint


def _scene(
    *,
    random_losses: tuple[float, float, float, float],
    trained_losses: tuple[float, float, float, float],
    supports: tuple[int, int, int, int],
) -> dict:
    names = tuple(LOSS_WEIGHTS)
    return {
        'random': dict(zip(names, random_losses)),
        'trained': dict(zip(names, trained_losses)),
        'supports': dict(zip(names, supports)),
    }


def _passing_metrics() -> dict:
    return aggregate_weighted_losses(
        [
            _scene(
                random_losses=(2.0, 2.0, 2.0, 2.0),
                trained_losses=(1.0, 1.0, 1.0, 1.0),
                supports=(1, 1, 1, 1),
            ),
        ],
        LOSS_WEIGHTS,
    )


def test_aggregate_weights_each_component_by_its_own_support() -> None:
    metrics = aggregate_weighted_losses(
        [
            _scene(
                random_losses=(1.0, 10.0, 100.0, 1000.0),
                trained_losses=(0.5, 8.0, 80.0, 800.0),
                supports=(1, 1, 1, 1),
            ),
            _scene(
                random_losses=(3.0, 30.0, 300.0, 3000.0),
                trained_losses=(1.5, 24.0, 240.0, 2400.0),
                supports=(3, 2, 4, 5),
            ),
        ],
        LOSS_WEIGHTS,
    )

    assert metrics['supports'] == {
        'task_distillation': 4,
        'termination': 3,
        'commitment': 5,
        'value': 6,
    }
    assert metrics['random']['task_distillation'] == pytest.approx(2.5)
    assert metrics['random']['termination'] == pytest.approx(70 / 3)
    assert metrics['random']['commitment'] == pytest.approx(260.0)
    assert metrics['random']['value'] == pytest.approx(8000 / 3)
    expected_total = sum(
        metrics['random'][name] * LOSS_WEIGHTS[name]
        for name in LOSS_WEIGHTS
    )
    assert metrics['random']['total'] == pytest.approx(expected_total)
    assert metrics['delta']['total'] == pytest.approx(
        metrics['trained']['total'] - metrics['random']['total']
    )
    assert metrics['relative_reduction']['task_distillation'] == (
        pytest.approx(0.5)
    )


def test_aggregate_rejects_invalid_support() -> None:
    scene = _scene(
        random_losses=(1.0, 1.0, 1.0, 1.0),
        trained_losses=(0.5, 0.5, 0.5, 0.5),
        supports=(-1, 1, 1, 1),
    )

    with pytest.raises(ValueError, match='support'):
        aggregate_weighted_losses([scene], LOSS_WEIGHTS)


def test_acceptance_requires_all_losses_to_strictly_decrease() -> None:
    metrics = _passing_metrics()
    accepted = decide_acceptance(
        metrics=metrics,
        scene_count=64,
        expected_scene_count=64,
        audit_passed=True,
    )
    assert accepted == {'accepted': True, 'reasons': []}

    metrics['trained']['commitment'] = metrics['random']['commitment']
    metrics['delta']['commitment'] = 0.0
    rejected = decide_acceptance(
        metrics=metrics,
        scene_count=64,
        expected_scene_count=64,
        audit_passed=True,
    )
    assert rejected['accepted'] is False
    assert any('commitment' in reason for reason in rejected['reasons'])


@pytest.mark.parametrize(
    ('scene_count', 'audit_passed', 'mutation', 'expected_reason'),
    [
        (63, True, None, 'scene count'),
        (64, False, None, 'audit'),
        (64, True, ('supports', 'termination', 0), 'support'),
        (64, True, ('random', 'value', math.inf), 'finite'),
    ],
)
def test_acceptance_rejects_incomplete_or_invalid_evidence(
    scene_count: int,
    audit_passed: bool,
    mutation: tuple[str, str, float] | None,
    expected_reason: str,
) -> None:
    metrics = _passing_metrics()
    if mutation is not None:
        section, name, value = mutation
        metrics[section][name] = value

    result = decide_acceptance(
        metrics=metrics,
        scene_count=scene_count,
        expected_scene_count=64,
        audit_passed=audit_passed,
    )

    assert result['accepted'] is False
    assert any(expected_reason in reason for reason in result['reasons'])


def test_paired_models_are_reproducible_and_share_exact_backbone(
    tmp_path,
) -> None:
    stage3_path, trained_path, _ = _pair_checkpoints(tmp_path)

    first = build_paired_models(
        model_kwargs=_model_kwargs(),
        stage3_checkpoint=stage3_path,
        trained_checkpoint=trained_path,
        expected_config_fingerprint='a' * 64,
        seed=3407,
        device=torch.device('cpu'),
    )
    second = build_paired_models(
        model_kwargs=_model_kwargs(),
        stage3_checkpoint=stage3_path,
        trained_checkpoint=trained_path,
        expected_config_fingerprint='a' * 64,
        seed=3407,
        device=torch.device('cpu'),
    )

    for name, value in first.random_model.state_dict().items():
        torch.testing.assert_close(
            value,
            second.random_model.state_dict()[name],
            rtol=0,
            atol=0,
        )
    for name, value in first.random_model.backbone.transformer.state_dict().items():
        torch.testing.assert_close(
            value,
            first.trained_model.backbone.transformer.state_dict()[name],
            rtol=0,
            atol=0,
        )
    assert not torch.equal(
        first.random_model.actor.idle_head.weight,
        first.trained_model.actor.idle_head.weight,
    )
    assert first.audit['backbone_exact_match'] is True
    assert first.audit['strict_model_load'] is True


@pytest.mark.parametrize(
    ('field', 'bad_value', 'message'),
    [
        ('checkpoint_version', 2, 'version'),
        ('stage', 'M3', 'stage'),
        ('steps', 9_999, 'step'),
        ('transition_schema_fingerprint', 'bad', 'schema'),
        ('config_fingerprint', 'b' * 64, 'config'),
    ],
)
def test_checkpoint_audit_rejects_metadata_mismatch(
    tmp_path,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    _, _, checkpoint = _pair_checkpoints(tmp_path)
    checkpoint[field] = bad_value

    with pytest.raises(ValueError, match=message):
        audit_training_checkpoint(
            checkpoint,
            expected_config_fingerprint='a' * 64,
            expected_steps=10_000,
        )


def test_paired_model_loader_rejects_non_strict_state_dict(tmp_path) -> None:
    stage3_path, trained_path, checkpoint = _pair_checkpoints(tmp_path)
    checkpoint['model'].pop('actor.idle_head.bias')
    torch.save(checkpoint, trained_path)

    with pytest.raises(RuntimeError, match='Missing key'):
        build_paired_models(
            model_kwargs=_model_kwargs(),
            stage3_checkpoint=stage3_path,
            trained_checkpoint=trained_path,
            expected_config_fingerprint='a' * 64,
            seed=3407,
            device=torch.device('cpu'),
        )


def _fake_batch(scene_index: int = 0):
    return SimpleNamespace(
        stage3_batch=SimpleNamespace(
            annotation_id=100 + scene_index,
            time_steps=[2, 7],
            constellation_mask=torch.tensor([
                [True, True],
                [True, False],
            ]),
        ),
        targets=SimpleNamespace(
            task_observed=torch.tensor([
                [True, False],
                [True, True],
            ]),
            termination_observed=torch.tensor([
                [False, True],
                [True, True],
            ]),
            commitment_observed=torch.tensor([
                [True, True],
                [False, True],
            ]),
            value_returns=torch.tensor([0.5, 0.25]),
        ),
    )


def test_batch_supports_respect_observed_and_satellite_masks() -> None:
    assert batch_supports(_fake_batch()) == {
        'task_distillation': 2,
        'termination': 2,
        'commitment': 2,
        'value': 2,
    }


def test_evaluate_dataset_builds_once_and_reuses_identical_batch() -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            self.calls.append(index)
            return _fake_batch(index)

    dataset = FakeDataset()
    seen: dict[int, list[int]] = {0: [], 1: []}

    def fake_loss(model, batch, **_):
        scene_index = batch.stage3_batch.annotation_id - 100
        seen[scene_index].append(id(batch))
        base = 2.0 if model == 'random' else 1.0
        value = torch.tensor(base)
        return SimpleNamespace(
            total=value * 4,
            task_distillation=value,
            termination=value,
            commitment=value,
            value=value,
        )

    records = evaluate_dataset(
        dataset=dataset,
        random_model='random',
        trained_model='trained',
        loss_weights={name: 1.0 for name in LOSS_WEIGHTS},
        device=torch.device('cpu'),
        seed=3407,
        loss_fn=fake_loss,
    )

    assert dataset.calls == [0, 1]
    assert all(len(ids) == 2 and ids[0] == ids[1] for ids in seen.values())
    assert [record['scene_id'] for record in records] == [100, 101]
    assert all(record['event_times'] == [2, 7] for record in records)
    assert records[0]['random']['task_distillation'] == pytest.approx(2.0)
    assert records[0]['trained']['task_distillation'] == pytest.approx(1.0)


def test_json_writer_refuses_to_overwrite_without_explicit_permission(
    tmp_path,
) -> None:
    path = tmp_path / 'summary.json'
    write_json_atomic(path, {'first': True}, overwrite=False)

    with pytest.raises(FileExistsError, match='already exists'):
        write_json_atomic(path, {'second': True}, overwrite=False)

    write_json_atomic(path, {'second': True}, overwrite=True)
    assert 'second' in path.read_text()


def test_memory_snapshot_has_stable_audit_fields_on_cpu() -> None:
    snapshot = cuda_memory_snapshot(
        torch.device('cpu'),
        requested_event_batch_size=64,
        maximum_actual_events=11,
    )

    assert snapshot == {
        'device': 'cpu',
        'device_name': None,
        'requested_event_batch_size': 64,
        'maximum_actual_events': 11,
        'max_memory_allocated_bytes': None,
        'max_memory_reserved_bytes': None,
        'total_memory_bytes': None,
        'max_reserved_fraction': None,
    }


def test_summary_records_fixed_scope_and_strict_formal_decision() -> None:
    passing_scene = _scene(
        random_losses=(2.0, 2.0, 2.0, 2.0),
        trained_losses=(1.0, 1.0, 1.0, 1.0),
        supports=(1, 1, 1, 1),
    )
    records = [
        {
            **passing_scene,
            'scene_index': index,
            'scene_id': 1000 + index,
            'event_times': [1],
            'actual_events': 1,
        }
        for index in range(64)
    ]
    summary = build_summary(
        records=records,
        loss_weights=LOSS_WEIGHTS,
        checkpoint_audit={
            'strict_model_load': True,
            'backbone_exact_match': True,
        },
        resources={'device': 'cpu'},
        formal=True,
        event_batch_size=8,
        annotation_file='val_unseen.json',
    )

    assert summary['decision'] == {'accepted': True, 'reasons': []}
    assert summary['scope'] == {
        'split': 'val_unseen',
        'annotation_file': 'val_unseen.json',
        'expected_scenes': 64,
        'processed_scenes': 64,
        'event_batch_size': 8,
        'formal': True,
    }
    assert summary['audit']['called_basilisk'] is False
    assert summary['audit']['read_test'] is False
    assert len(summary['scenes']) == 64


def test_cli_help_exposes_probe_and_formal_controls() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            '/home/hy/miniconda3/envs/aeos/bin/python',
            'tools/evaluate_event_v2_unseen_offline.py',
            '--help',
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '--event-batch-size' in result.stdout
    assert '--limit' in result.stdout
    assert '--formal' in result.stdout
    assert '--overwrite' in result.stdout
