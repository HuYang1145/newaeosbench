from pathlib import Path

import torch

from tools.audit_same_scene_preference_critic import load_saved_bundle


def test_load_saved_bundle_restores_models_and_statistics(tmp_path: Path) -> None:
    checkpoint = tmp_path / 'critic.pth'
    torch.save({
        'baseline': {
            '_layers.0.weight': torch.zeros(4, 2),
            '_layers.0.bias': torch.zeros(4),
            '_layers.1.weight': torch.ones(4),
            '_layers.1.bias': torch.zeros(4),
            '_layers.3.weight': torch.zeros(4, 4),
            '_layers.3.bias': torch.zeros(4),
            '_layers.5.weight': torch.zeros(1, 4),
            '_layers.5.bias': torch.zeros(1),
        },
        'critic': {
            '_layers.0.weight': torch.zeros(4, 3),
            '_layers.0.bias': torch.zeros(4),
            '_layers.1.weight': torch.ones(4),
            '_layers.1.bias': torch.zeros(4),
            '_layers.3.weight': torch.zeros(4, 4),
            '_layers.3.bias': torch.zeros(4),
            '_layers.5.weight': torch.zeros(1, 4),
            '_layers.5.bias': torch.zeros(1),
        },
        'state_mean': torch.tensor([1.0, 2.0]),
        'state_std': torch.tensor([3.0, 4.0]),
        'action_mean': torch.tensor([5.0]),
        'action_std': torch.tensor([6.0]),
    }, checkpoint)

    bundle = load_saved_bundle(
        checkpoint,
        state_dim=2,
        action_dim=1,
        hidden_dim=4,
    )

    assert bundle.state_mean.tolist() == [1.0, 2.0]
    assert bundle.action_std.tolist() == [6.0]
