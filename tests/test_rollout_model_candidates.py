import json
from pathlib import Path

import pytest
import torch

from tools.rollout_model_trajectories import (
    select_action_indices,
    write_rollout_metadata,
)


def test_greedy_candidate_matches_existing_argmax() -> None:
    logits = torch.tensor([
        [[1.0, 4.0, 2.0], [3.0, 2.0, 5.0]],
    ])

    selected = select_action_indices(logits, strategy='greedy')

    assert torch.equal(selected, logits.argmax(-1))


def test_top_k_candidate_is_seeded_and_stays_inside_top_k() -> None:
    logits = torch.tensor([
        [[5.0, 4.0, 3.0, -100.0], [1.0, 3.0, 2.0, -100.0]],
    ])

    first = select_action_indices(
        logits,
        strategy='top_k_sample',
        top_k=2,
        temperature=0.7,
        generator=torch.Generator().manual_seed(17),
    )
    second = select_action_indices(
        logits,
        strategy='top_k_sample',
        top_k=2,
        temperature=0.7,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.equal(first, second)
    assert first[0, 0].item() in {0, 1}
    assert first[0, 1].item() in {1, 2}


@pytest.mark.parametrize(
    ('top_k', 'temperature'),
    [(0, 0.7), (3, 0.0), (3, -1.0)],
)
def test_top_k_candidate_rejects_invalid_sampling_parameters(
    top_k: int,
    temperature: float,
) -> None:
    with pytest.raises(ValueError):
        select_action_indices(
            torch.zeros(1, 2, 4),
            strategy='top_k_sample',
            top_k=top_k,
            temperature=temperature,
        )


def test_only_rank_zero_writes_shared_rollout_metadata(tmp_path: Path) -> None:
    metadata = {'strategy': 'top_k_sample', 'seed': 17}

    write_rollout_metadata(tmp_path, metadata=metadata, rank=1)
    assert not (tmp_path / 'rollout_metadata.json').exists()

    write_rollout_metadata(tmp_path, metadata=metadata, rank=0)
    assert json.loads(
        (tmp_path / 'rollout_metadata.json').read_text(encoding='utf-8'),
    ) == metadata
