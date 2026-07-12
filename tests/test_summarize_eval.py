import pytest

from tools import summarize_eval


@pytest.mark.parametrize(
    ('cr', 'pcr', 'wcr', 'tat_100s', 'pc_wh', 'expected_cs'),
    [
        (0.3047, 0.3368, 0.3005, 7.50, 71.27, 5.00),
        (0.3542, 0.3893, 0.3514, 6.78, 68.99, 4.43),
        (0.1925, 0.2231, 0.1873, 5.67, 40.91, 6.28),
    ],
)
def test_compute_scores_reproduces_paper_table_2(
    cr: float,
    pcr: float,
    wcr: float,
    tat_100s: float,
    pc_wh: float,
    expected_cs: float,
) -> None:
    scores = summarize_eval.compute_scores(
        cr=cr,
        pcr=pcr,
        wcr=wcr,
        tat_s=tat_100s * 100,
        pc_wh=pc_wh,
    )

    assert scores['TAT_100s'] == pytest.approx(tat_100s)
    assert scores['CS_paper'] == pytest.approx(expected_cs, abs=0.01)


def test_compute_scores_rejects_non_positive_quality() -> None:
    with pytest.raises(ValueError, match='quality'):
        summarize_eval.compute_scores(
            cr=0.0,
            pcr=0.0,
            wcr=0.0,
            tat_s=0.0,
            pc_wh=0.0,
        )
