import sys

import pytest

from tools import build_tau_e_annotation


def test_paper_full_defaults_to_tat_seconds_over_700(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'build_tau_e_annotation',
            'base.json',
            'trajectories.2',
            'output.json',
            '--candidate-epoch',
            '2',
        ],
    )
    args = build_tau_e_annotation.parse_args()

    score, formula = build_tau_e_annotation.score_from_metrics(
        {
            'CR': 0.5,
            'PCR': 0.5,
            'WCR': 0.5,
            'TAT': 700.0,
            'PC_Wh': 0.0,
        },
        score_key='CS',
        formula=args.formula,
        tat_scale=args.tat_scale,
        pc_scale=args.pc_scale,
        tat_weight_scale=args.tat_weight_scale,
    )

    assert args.tat_scale == 100.0
    assert score == pytest.approx(3.0)
    assert '(TAT/100)/7' in formula
