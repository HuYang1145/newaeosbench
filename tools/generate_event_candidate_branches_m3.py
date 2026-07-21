"""生成 M3 事件级 stay/switch-duration 受控局部分支。"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import torch
from todd.patches.py_ import json_dump, json_load

from constellation import (
    CONSTELLATIONS_ROOT,
    STATISTICS_PATH,
    TASKSETS_ROOT,
)
from constellation.callbacks import ComposedCallback
from constellation.controller import Controller
from constellation.data import Constellation, Task, TaskSet
from constellation.environments import BasiliskEnvironment
from constellation.new_transformers import Statistics
from constellation.new_transformers.event_candidate import (
    EventDecisionPoint,
    audit_preference_pair,
    build_event_candidate_specs,
    find_event_decisions,
)
from constellation.new_transformers.local_action_branch import (
    ControlledCommitmentAlgorithm,
    LocalWindowCallback,
    is_decision_replayable,
)
from constellation.task_managers import TaskManager
from tools.rollout_model_trajectories import GreedyModelAlgorithm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate controlled M3 event candidate branches',
    )
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reference_trajectory', type=Path)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--m2-checkpoint', type=Path, default=None)
    parser.add_argument('--split', default='train')
    parser.add_argument('--scene-id', type=int, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=[60, 180, 300],
    )
    parser.add_argument(
        '--commitments',
        type=int,
        nargs='+',
        default=[1, 5, 15, 30, 60],
    )
    parser.add_argument('--max-decisions', type=int, default=2)
    parser.add_argument('--min-margin', type=float, default=0.01)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def resolve_switch_task_id(
    *,
    stay_task_id: int,
    actor_logits: list[float],
    ongoing_task_ids: list[int],
) -> int:
    """选择 Stage3 中排名最高且不同于 stay 的动作。"""
    logits = torch.tensor(actor_logits, dtype=torch.float)
    if logits.ndim != 1 or logits.numel() != len(ongoing_task_ids) + 1:
        raise ValueError('actor logits do not match ongoing task ids')
    for relative_index in logits.argsort(descending=True).tolist():
        task_id = (
            -1
            if relative_index == 0
            else int(ongoing_task_ids[relative_index - 1])
        )
        if task_id != stay_task_id:
            return task_id
    raise ValueError('no actor candidate differs from stay')


def validate_common_decision_state(
    branches: dict[str, dict[str, Any]],
) -> None:
    if not branches:
        raise ValueError('at least one branch is required')
    signatures = {
        branch.get('decision_state_signature')
        for branch in branches.values()
    }
    if None in signatures or len(signatures) != 1:
        raise ValueError('branches must share the same decision state')
    contexts = [
        branch.get('decision_context') for branch in branches.values()
    ]
    if any(context is None for context in contexts):
        raise ValueError('branch is missing decision context')
    if any(context != contexts[0] for context in contexts[1:]):
        raise ValueError('branches must share the same decision context')


def build_group_preference_audits(
    branches: dict[str, dict[str, Any]],
    *,
    min_margin: float,
) -> list[dict[str, Any]]:
    """保留所有 pair 的接受或拒绝原因。"""
    validate_common_decision_state(branches)
    return [
        audit_preference_pair(
            first,
            second,
            branches,
            min_margin=min_margin,
        ).to_dict()
        for first, second in itertools.combinations(branches, 2)
    ]


def _spread_decisions(
    decisions: list[EventDecisionPoint],
    max_decisions: int,
) -> list[EventDecisionPoint]:
    first_by_bin: dict[int, EventDecisionPoint] = {}
    for decision in decisions:
        first_by_bin.setdefault(decision.decision_time // 300, decision)
    representatives = list(first_by_bin.values())
    if len(representatives) <= max_decisions:
        return representatives
    if max_decisions == 1:
        return [representatives[len(representatives) // 2]]
    indices = [
        round(index * (len(representatives) - 1) / (max_decisions - 1))
        for index in range(max_decisions)
    ]
    return [representatives[index] for index in indices]


def select_replayable_event_decisions(
    *,
    actions: torch.Tensor,
    reference_progress: torch.Tensor,
    taskset: TaskSet,
    horizons: list[int],
    max_decisions: int,
) -> list[EventDecisionPoint]:
    if not horizons or min(horizons) <= 0:
        raise ValueError('horizons must be positive')
    if max_decisions <= 0:
        raise ValueError('max_decisions must be positive')
    latest = actions.shape[0] - max(horizons) - 1
    if latest < 1:
        return []
    candidates = find_event_decisions(
        actions,
        max_decisions=max(actions.numel(), 1),
        latest_decision_time=latest,
        bin_seconds=1,
    )
    replayable = [
        decision
        for decision in candidates
        if is_decision_replayable(
            decision,
            taskset=taskset,
            reference_progress=reference_progress,
        )
    ]
    return _spread_decisions(replayable, max_decisions)


def run_event_branch(
    *,
    split: str,
    scene_id: int,
    checkpoint: Path,
    device: torch.device,
    statistics: Statistics,
    decision: EventDecisionPoint,
    horizons: list[int],
    branch_name: str,
    forced_task_id: int,
    commitment_seconds: int,
) -> dict[str, Any]:
    relative_path = Path(f'{scene_id // 1000:02}') / f'{scene_id:05}.json'
    constellation = Constellation.load(
        str(CONSTELLATIONS_ROOT / split / relative_path),
    )
    taskset: TaskSet[Task] = TaskSet.load(
        str(TASKSETS_ROOT / split / relative_path),
    )
    environment = BasiliskEnvironment(
        start_time=0,
        constellation=constellation,
        all_tasks=taskset,
    )
    task_manager = TaskManager(timer=environment.timer, taskset=taskset)
    collector = LocalWindowCallback(
        decision_time=decision.decision_time,
        horizons=horizons,
        target_satellite_index=decision.satellite_index,
    )
    controller = Controller(
        f'{scene_id:05}_{decision.decision_time}_{branch_name}',
        environment=environment,
        task_manager=task_manager,
        callbacks=ComposedCallback(callbacks=[collector]),
    )
    base_algorithm = GreedyModelAlgorithm(
        timer=environment.timer,
        checkpoint=checkpoint,
        device=device,
        statistics=statistics,
        strategy='greedy',
    )
    algorithm = ControlledCommitmentAlgorithm(
        timer=environment.timer,
        base_algorithm=base_algorithm,
        decision_time=decision.decision_time,
        satellite_index=decision.satellite_index,
        forced_task_id=forced_task_id,
        commitment_seconds=commitment_seconds,
    )
    algorithm.prepare(environment=environment, task_manager=task_manager)
    controller.run(
        algorithm,
        max_time_step=decision.decision_time + max(horizons) + 1,
        progress_bar=False,
    )
    if not algorithm.override_applied:
        raise RuntimeError('event commitment override was not applied')
    if algorithm.original_task_id != decision.switch_task_id:
        raise RuntimeError(
            'deterministic replay diverged before the event decision: '
            f'expected {decision.switch_task_id}, '
            f'got {algorithm.original_task_id}'
        )
    return {
        'branch': branch_name,
        'applied_task_id': algorithm.applied_task_id,
        'original_task_id': algorithm.original_task_id,
        'original_assignment': algorithm.original_assignment,
        'requested_commitment_seconds': (
            algorithm.requested_commitment_seconds
        ),
        'actual_commitment_seconds': algorithm.actual_commitment_seconds,
        'interruption_reason': algorithm.interruption_reason,
        'decision_state_signature': algorithm.decision_state_signature,
        'decision_context': algorithm.decision_context,
        'decision_time': decision.decision_time,
        'satellite_index': decision.satellite_index,
        'horizons': {
            str(horizon): summary
            for horizon, summary in collector.summaries.items()
        },
    }


def main() -> None:
    args = parse_args()
    horizons = sorted(set(int(value) for value in args.horizons))
    commitments = tuple(sorted(set(int(value) for value in args.commitments)))
    if not horizons or min(horizons) <= 0:
        raise ValueError('horizons must be positive')
    if args.max_decisions <= 0:
        raise ValueError('max-decisions must be positive')
    if args.min_margin < 0:
        raise ValueError('min-margin must be non-negative')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.reference_trajectory.is_file():
        raise FileNotFoundError(args.reference_trajectory)
    if args.m2_checkpoint is not None and not args.m2_checkpoint.is_file():
        raise FileNotFoundError(args.m2_checkpoint)

    scene_id = (
        int(args.reference_trajectory.stem)
        if args.scene_id is None
        else args.scene_id
    )
    reference = torch.load(
        args.reference_trajectory,
        map_location='cpu',
        weights_only=False,
    )
    relative_path = Path(f'{scene_id // 1000:02}') / f'{scene_id:05}.json'
    taskset: TaskSet[Task] = TaskSet.load(
        str(TASKSETS_ROOT / args.split / relative_path),
    )
    decisions = select_replayable_event_decisions(
        actions=reference['actions']['task_id'],
        reference_progress=reference['taskset']['progress'],
        taskset=taskset,
        horizons=horizons,
        max_decisions=args.max_decisions,
    )
    if not decisions:
        raise RuntimeError('no replayable event decisions found')

    statistics: Statistics = torch.load(
        STATISTICS_PATH,
        map_location='cpu',
        weights_only=False,
    )
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for decision_index, decision in enumerate(decisions):
        branch_dir = args.output_root / (
            f'decision_{decision_index:03}_{decision.decision_time:04}'
        )
        branch_dir.mkdir(parents=True, exist_ok=True)

        def load_or_run(spec) -> dict[str, Any]:
            path = branch_dir / f'{spec.name}.json'
            if path.exists() and not args.overwrite:
                return json_load(str(path))
            result = run_event_branch(
                split=args.split,
                scene_id=scene_id,
                checkpoint=args.checkpoint,
                device=device,
                statistics=statistics,
                decision=decision,
                horizons=horizons,
                branch_name=spec.name,
                forced_task_id=spec.task_id,
                commitment_seconds=spec.commitment_seconds,
            )
            json_dump(result, str(path))
            return result

        seed_specs = build_event_candidate_specs(
            stay_task_id=decision.stay_task_id,
            switch_task_id=decision.switch_task_id,
            commitments=commitments,
        )
        seed_stay = next(
            item
            for item in seed_specs
            if item.action_kind == 'stay' and item.commitment_seconds == 1
        )
        seed_result = load_or_run(seed_stay)
        context = seed_result['decision_context']
        switch_task_id = resolve_switch_task_id(
            stay_task_id=decision.stay_task_id,
            actor_logits=context['actor_logits'][decision.satellite_index],
            ongoing_task_ids=context['ongoing_task_ids'],
        )
        specs = build_event_candidate_specs(
            stay_task_id=decision.stay_task_id,
            switch_task_id=switch_task_id,
            commitments=commitments,
        )
        branches = {seed_stay.name: seed_result}
        for spec in specs:
            if spec.name in branches:
                continue
            branches[spec.name] = load_or_run(spec)
        validate_common_decision_state(branches)
        pair_audits = build_group_preference_audits(
            branches,
            min_margin=args.min_margin,
        )
        records.append({
            'scene_id': scene_id,
            'decision_index': decision_index,
            'decision': decision.to_dict(),
            'resolved_switch_task_id': switch_task_id,
            'branches': branches,
            'pair_audits': pair_audits,
            'accepted_preferences': [
                item for item in pair_audits if item['accepted']
            ],
        })

    summary = {
        'protocol': (
            'same deterministic Stage3 state; one satellite task and '
            'commitment differ; all branches share the frozen actor'
        ),
        'checkpoint': str(args.checkpoint),
        'm2_checkpoint': (
            None if args.m2_checkpoint is None else str(args.m2_checkpoint)
        ),
        'm2_shadow_available': False,
        'm2_shadow_reason': 'shadow predictions not captured in M3-A',
        'reference_trajectory': str(args.reference_trajectory),
        'split': args.split,
        'scene_id': scene_id,
        'horizons': horizons,
        'commitments': commitments,
        'min_margin': args.min_margin,
        'records': records,
    }
    json_dump(summary, str(args.output_root / 'summary.json'))
    print(f'[m3-event] wrote {args.output_root / "summary.json"}')


if __name__ == '__main__':
    main()
