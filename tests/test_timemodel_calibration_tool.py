import sys

import torch

from constellation.new_transformers.dataset import TimeSpan, TimeSpans
from tools import calibrate_timemodel_feasibility as calibration


def test_build_span_rows_combines_real_positive_and_negative_pairs() -> None:
    build_span_rows = getattr(calibration, 'build_span_rows', None)
    assert callable(build_span_rows)

    positives = TimeSpans()
    positives.append(TimeSpan(0, 3, 0, 1))
    negatives = TimeSpans()
    negatives.append(TimeSpan(3, 5, 0, 2))
    negatives.append(TimeSpan(5, 7, 0, -1))

    rows = build_span_rows(positives, negatives)

    assert rows.tolist() == [
        [0, 3, 0, 1],
        [1, 2, 0, 1],
        [2, 1, 0, 1],
        [3, -50, 0, 2],
        [4, -50, 0, 2],
    ]


def test_extract_time_model_state_dict_removes_model_prefix() -> None:
    extract = getattr(calibration, 'extract_time_model_state_dict', None)
    assert callable(extract)
    state_dict = {
        '_transformer._time_model._mlp.0.weight': torch.ones(1),
        '_transformer._encoder._norm.weight': torch.zeros(1),
    }

    extracted = extract(state_dict)

    assert list(extracted) == ['_mlp.0.weight']
    assert extracted['_mlp.0.weight'].item() == 1


def test_calibration_cli_parses_explicit_scope(monkeypatch) -> None:
    parse_args = getattr(calibration, 'parse_args', None)
    assert callable(parse_args)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'calibrate_timemodel_feasibility.py',
            'model.pth',
            '--split',
            'val_seen',
            '--max-scenes',
            '2',
            '--thresholds',
            '0.01',
            '0.05',
            '--output',
            'summary.json',
        ],
    )

    args = parse_args()

    assert args.checkpoint.name == 'model.pth'
    assert args.split == 'val_seen'
    assert args.max_scenes == 2
    assert args.thresholds == [0.01, 0.05]
    assert args.output.name == 'summary.json'
