import torch

from constellation.new_transformers.dataset import JointDataset


class _FakeSpans:

    def __init__(self, rows: torch.Tensor) -> None:
        self._rows = rows

    def sample_data(self, n: int, **kwargs) -> torch.Tensor:
        return self._rows[:n]


def test_joint_dataset_loads_trajectory_once(monkeypatch) -> None:
    dataset = object.__new__(JointDataset)
    dataset._annotations = {'ids': [1234], 'epochs': [7]}
    dataset._split = 'train'
    dataset._batch_size = 3
    dataset._constraint_batch_size = 2

    trajectory = {
        'constellation': {
            'sensor_enabled': torch.ones(4, 1, dtype=torch.bool),
            'data': torch.zeros(4, 1, 56),
        },
        'taskset': {
            'progress': torch.zeros(4, 2),
        },
        'actions': {
            'task_id': torch.zeros(4, 1, dtype=torch.long),
        },
        'is_visible': torch.ones(4, 1, 2, dtype=torch.bool),
    }
    load_calls = []
    load_tasks_calls = []
    load_constellation_calls = []

    def fake_torch_load(*args, **kwargs):
        load_calls.append(args[0])
        return trajectory

    def fake_load_tasks(self, taskset, id_):
        load_tasks_calls.append(id_)
        t = taskset['progress'].shape[0]
        nt = taskset['progress'].shape[1]
        return (
            torch.ones(t, nt, dtype=torch.long),
            torch.zeros(t, nt, 6),
            torch.ones(t, nt, dtype=torch.bool),
        )

    def fake_load_constellation(self, constellation, id_, indices):
        load_constellation_calls.append((id_, tuple(indices)))
        n = len(indices)
        ns = constellation['data'].shape[1]
        return (
            torch.ones(n, ns, dtype=torch.long),
            torch.ones(n, ns, dtype=torch.long),
            torch.zeros(n, ns, 56),
            torch.ones(n, ns, dtype=torch.bool),
        )

    def fake_parse_time_spans(self, actions, is_visible):
        positives = _FakeSpans(torch.tensor([[0, 2, 0, 0]], dtype=torch.int))
        negatives = _FakeSpans(torch.tensor([[1, -50, 0, 1]], dtype=torch.int))
        return positives, negatives

    monkeypatch.setattr(torch, 'load', fake_torch_load)
    monkeypatch.setattr(JointDataset, '_load_tasks', fake_load_tasks)
    monkeypatch.setattr(JointDataset, '_load_constellation', fake_load_constellation)
    monkeypatch.setattr(JointDataset, '_parse_time_spans', fake_parse_time_spans)

    _ = dataset[0]

    assert len(load_calls) == 1
    assert len(load_tasks_calls) == 1
    assert len(load_constellation_calls) == 1
