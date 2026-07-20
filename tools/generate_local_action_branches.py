"""生成 stay/switch 受控局部 rollout。

两个分支从同一场景重新运行到同一决策时刻，只覆盖一颗卫星当前一步的动作，之后
继续使用同一个冻结的 deterministic greedy Actor。输出只包含局部原始指标，暂不
把完成、进度、功耗和切换压成单一奖励。
"""

from __future__ import annotations

import argparse
import itertools
from numbers import Real
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
from constellation.new_transformers.local_action_branch import (
    BranchDecision,
    ControlledActionAlgorithm,
    LocalWindowCallback,
    find_stay_switch_decisions,
    is_decision_replayable,
)
from constellation.task_managers import TaskManager
from tools.rollout_model_trajectories import GreedyModelAlgorithm

LOCAL_METRIC_KEYS = (
    'completed_tasks',
    'completed_duration',
    'partial_progress_gain',
    'working_satellite_seconds',
    'pc_wh',
    'switches',
    'target_satellite_switches',
    'one_second_runs',
    'target_satellite_one_second_runs',
    'direct_visible_satellite_seconds',
    'target_satellite_direct_visible_seconds',
    'redundant_satellite_seconds',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate controlled local stay/switch branches',
    )
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reference_trajectory', type=Path)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--split', default='train')
    parser.add_argument('--scene-id', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=[180, 300, 600],
    )
    parser.add_argument('--primary-horizon', type=int, default=300)
    parser.add_argument('--max-decisions', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def candidate_branch_specs(
    *,
    stay_task_id: int,
    top_k: int,
    num_available_candidates: int | None = None,
) -> list[dict[str, int | str]]:
    """构造 stay 与 Actor 有序 top-k 候选规格。"""

    if top_k <= 0:
        raise ValueError('top_k must be positive')
    if num_available_candidates is not None and num_available_candidates <= 0:
        raise ValueError('num_available_candidates must be positive')
    ranked_count = (
        top_k if num_available_candidates is None else
        min(top_k, num_available_candidates)
    )
    return [
        {
            'name': 'stay',
            'forced_task_id': int(stay_task_id)
        },
        *[{
            'name': f'actor_rank_{rank}',
            'forced_candidate_rank': rank,
        } for rank in range(ranked_count)],
    ]


def unique_candidate_branch_specs(
    *,
    stay_task_id: int,
    actor_logits: list[float],
    ongoing_task_ids: list[int],
    top_k: int,
) -> list[dict[str, int | str]]:
    """解析实际 top-k，并跳过与 stay 或更高排名相同的动作。"""

    logits = torch.tensor(actor_logits)
    if logits.ndim != 1 or logits.numel() != len(ongoing_task_ids) + 1:
        raise ValueError('actor logits do not match ongoing task ids')
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    order = logits.argsort(descending=True)[:top_k].tolist()
    specs: list[dict[str, int | str]] = [{
        'name': 'stay',
        'forced_task_id': int(stay_task_id),
        'resolved_task_id': int(stay_task_id),
    }]
    seen = {int(stay_task_id)}
    for rank, relative_index in enumerate(order):
        task_id = (
            -1 if relative_index == 0 else
            int(ongoing_task_ids[relative_index - 1])
        )
        if task_id in seen:
            continue
        seen.add(task_id)
        specs.append({
            'name': f'actor_rank_{rank}',
            'forced_candidate_rank': rank,
            'resolved_task_id': task_id,
        })
    return specs


def select_replayable_decisions(
    *,
    actions: torch.Tensor,
    reference_progress: torch.Tensor,
    taskset: TaskSet,
    horizons: list[int],
    max_decisions: int,
) -> list[BranchDecision]:
    """选择能完整覆盖最大窗口、且两个动作均可重放的决策点。"""

    if not horizons or min(horizons) <= 0:
        raise ValueError('horizons must be positive')
    latest = actions.shape[0] - max(horizons) - 1
    if latest < 1:
        return []
    candidates = find_stay_switch_decisions(
        actions,
        max_decisions=actions.numel(),
        latest_decision_time=latest,
    )
    candidates = [
        decision for decision in candidates if is_decision_replayable(
            decision,
            taskset=taskset,
            reference_progress=reference_progress,
        )
    ]

    # 先取每个五分钟区间的首个候选，再在整段时间上均匀抽取，避免小样本
    # 永远只覆盖场景开头。
    first_by_bin: dict[int, BranchDecision] = {}
    for decision in candidates:
        time_bin = decision.decision_time // 300
        first_by_bin.setdefault(time_bin, decision)
    representatives = list(first_by_bin.values())
    if len(representatives) <= max_decisions:
        selected = representatives.copy()
    elif max_decisions == 1:
        selected = [representatives[len(representatives) // 2]]
    else:
        positions = [
            round(index * (len(representatives) - 1) / (max_decisions - 1))
            for index in range(max_decisions)
        ]
        selected = [representatives[position] for position in positions]
    if len(selected) >= max_decisions:
        return selected[:max_decisions]
    for decision in candidates:
        if decision in selected:
            continue
        selected.append(decision)
        if len(selected) >= max_decisions:
            break
    return selected


def build_pair_summary(
    *,
    stay: dict[str, Any],
    switch: dict[str, Any],
) -> dict[str, Any]:
    """保留两个分支及 switch-stay 原始数值差，不提前规定奖励权重。"""

    stay_signature = stay.get('decision_state_signature')
    switch_signature = switch.get('decision_state_signature')
    if stay_signature is None or switch_signature is None:
        raise ValueError('controlled pair is missing a decision state')
    if stay_signature != switch_signature:
        raise ValueError(
            'controlled branches started from different decision states'
        )

    delta = {
        key: float(switch[key] - stay[key])
        for key in LOCAL_METRIC_KEYS
        if (
            key in stay and key in switch and isinstance(stay[key], Real) and
            not isinstance(stay[key], bool) and isinstance(switch[key], Real)
            and not isinstance(switch[key], bool)
        )
    }
    return {
        'stay': stay,
        'switch': switch,
        'switch_minus_stay': delta,
    }


def build_candidate_pair_records(
    branches: dict[str, dict[str, Any]],
    *,
    primary_horizon: int,
) -> list[dict[str, Any]]:
    """按主窗口前缀 cost 构造同状态候选偏好对。"""

    if primary_horizon <= 0:
        raise ValueError('primary_horizon must be positive')
    names = list(branches)
    signatures = {
        branches[name].get('decision_state_signature')
        for name in names
    }
    if None in signatures or len(signatures) != 1:
        raise ValueError('candidate branches do not share one decision state')
    contexts = [branches[name].get('decision_context') for name in names]
    if any(context is None for context in contexts):
        raise ValueError('candidate branch is missing decision context')
    if any(context != contexts[0] for context in contexts[1:]):
        raise ValueError('candidate branches have different decision contexts')

    unique_names = []
    seen_actions: set[tuple[int, ...]] = set()
    for name in names:
        branch = branches[name]
        assignment = list(branch.get('original_assignment', []))
        satellite_index = branch.get('satellite_index')
        if assignment and satellite_index is not None:
            assignment[int(satellite_index)] = int(branch['applied_task_id'])
            signature = tuple(int(value) for value in assignment)
        else:
            signature = (int(branch['applied_task_id']), )
        if signature in seen_actions:
            continue
        seen_actions.add(signature)
        unique_names.append(name)

    records = []
    horizon_key = str(primary_horizon)
    for first_name, second_name in itertools.combinations(unique_names, 2):
        first = branches[first_name]
        second = branches[second_name]
        if first.get('applied_task_id') == second.get('applied_task_id'):
            continue
        first_cost = first['horizons'][horizon_key]['prefix_metrics'][
            'prefix_cost']
        second_cost = second['horizons'][horizon_key]['prefix_metrics'][
            'prefix_cost']
        if first_cost is None or second_cost is None:
            continue
        first_cost = float(first_cost)
        second_cost = float(second_cost)
        if first_cost == second_cost:
            continue
        if first_cost < second_cost:
            better_name, better, better_cost = first_name, first, first_cost
            worse_name, worse, worse_cost = second_name, second, second_cost
        else:
            better_name, better, better_cost = second_name, second, second_cost
            worse_name, worse, worse_cost = first_name, first, first_cost
        records.append({
            'better_branch': better_name,
            'worse_branch': worse_name,
            'better_task_id': int(better['applied_task_id']),
            'worse_task_id': int(worse['applied_task_id']),
            'better_cost': better_cost,
            'worse_cost': worse_cost,
            'cost_margin': worse_cost - better_cost,
            'primary_horizon': primary_horizon,
        })
    return records


def run_local_branch(
    *,
    split: str,
    scene_id: int,
    checkpoint: Path,
    device: torch.device,
    statistics: Statistics,
    decision: BranchDecision,
    horizons: list[int],
    branch: str,
    forced_task_id: int | None = None,
    forced_candidate_rank: int | None = None,
) -> dict[str, Any]:
    """从零重放场景，一次提取同一分支的全部局部窗口。"""

    if not horizons or min(horizons) <= 0:
        raise ValueError('horizons must be positive')

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
        horizons=tuple(horizons),
        target_satellite_index=decision.satellite_index,
    )
    callbacks = ComposedCallback(callbacks=[collector])
    controller = Controller(
        f'{scene_id:05}_{decision.decision_time}_{branch}_{max(horizons)}',
        environment=environment,
        task_manager=task_manager,
        callbacks=callbacks,
    )
    base_algorithm = GreedyModelAlgorithm(
        timer=environment.timer,
        checkpoint=checkpoint,
        device=device,
        statistics=statistics,
        strategy='greedy',
    )
    algorithm = ControlledActionAlgorithm(
        timer=environment.timer,
        base_algorithm=base_algorithm,
        decision_time=decision.decision_time,
        satellite_index=decision.satellite_index,
        forced_task_id=forced_task_id,
        forced_candidate_rank=forced_candidate_rank,
    )
    algorithm.prepare(environment=environment, task_manager=task_manager)
    controller.run(
        algorithm,
        max_time_step=decision.decision_time + max(horizons) + 1,
        progress_bar=False,
    )
    if not algorithm.override_applied:
        raise RuntimeError('controlled action override was not applied')
    if algorithm.original_task_id != decision.switch_task_id:
        raise RuntimeError(
            'deterministic replay diverged before the decision: '
            f'expected {decision.switch_task_id}, '
            f'got {algorithm.original_task_id}',
        )

    return {
        'horizons': {
            str(horizon): summary
            for horizon, summary in collector.summaries.items()
        },
        'branch': branch,
        'forced_task_id': forced_task_id,
        'forced_candidate_rank': forced_candidate_rank,
        'applied_task_id': algorithm.applied_task_id,
        'original_task_id': algorithm.original_task_id,
        'original_assignment': algorithm.original_assignment,
        'decision_state_signature': algorithm.decision_state_signature,
        'decision_context': algorithm.decision_context,
        'decision_time': decision.decision_time,
        'satellite_index': decision.satellite_index,
    }


def main() -> None:
    args = parse_args()
    if args.max_decisions <= 0:
        raise ValueError('max-decisions must be positive')
    horizons = sorted(set(args.horizons))
    if not horizons or min(horizons) <= 0:
        raise ValueError('horizons must be positive')
    if args.primary_horizon not in horizons:
        raise ValueError('primary-horizon must be included in horizons')
    if args.top_k <= 0:
        raise ValueError('top-k must be positive')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.reference_trajectory.is_file():
        raise FileNotFoundError(args.reference_trajectory)

    scene_id = (
        int(args.reference_trajectory.stem)
        if args.scene_id is None else args.scene_id
    )
    reference = torch.load(
        args.reference_trajectory,
        map_location='cpu',
        weights_only=False,
    )
    relative_path = Path(f'{scene_id // 1000:02}') / f'{scene_id:05}.json'
    taskset = TaskSet.load(str(TASKSETS_ROOT / args.split / relative_path))
    decisions = select_replayable_decisions(
        actions=reference['actions']['task_id'],
        reference_progress=reference['taskset']['progress'],
        taskset=taskset,
        horizons=horizons,
        max_decisions=args.max_decisions,
    )
    if not decisions:
        raise RuntimeError('no replayable one-second decisions found')

    statistics: Statistics = torch.load(
        STATISTICS_PATH,
        map_location='cpu',
        weights_only=False,
    )
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for decision_index, decision in enumerate(decisions):
        decision_record: dict[str, Any] = {
            'decision_index': decision_index,
            'decision': decision.to_dict(),
            'branches': {},
        }
        branch_dir = (
            args.output_root
            / f'decision_{decision_index:03}_{decision.decision_time:04}'
        )
        branch_dir.mkdir(parents=True, exist_ok=True)
        branch_results: dict[str, dict[str, Any]] = {}

        def load_or_run(spec: dict[str, int | str]) -> dict[str, Any]:
            branch = str(spec['name'])
            output_path = branch_dir / f'{branch}.json'
            if output_path.exists() and not args.overwrite:
                return json_load(str(output_path))
            result = run_local_branch(
                split=args.split,
                scene_id=scene_id,
                checkpoint=args.checkpoint,
                device=device,
                statistics=statistics,
                decision=decision,
                horizons=horizons,
                branch=branch,
                forced_task_id=spec.get('forced_task_id'),
                forced_candidate_rank=spec.get('forced_candidate_rank'),
            )
            json_dump(result, str(output_path))
            return result

        stay_spec = candidate_branch_specs(
            stay_task_id=decision.stay_task_id,
            top_k=args.top_k,
        )[0]
        stay_result = load_or_run(stay_spec)
        branch_results['stay'] = stay_result
        context = stay_result['decision_context']
        specs = unique_candidate_branch_specs(
            stay_task_id=decision.stay_task_id,
            actor_logits=context['actor_logits'][decision.satellite_index],
            ongoing_task_ids=context['ongoing_task_ids'],
            top_k=args.top_k,
        )
        for spec in specs[1:]:
            branch = str(spec['name'])
            result = load_or_run(spec)
            branch_results[branch] = result
        decision_record['branches'] = branch_results
        decision_record['preference_pairs'] = build_candidate_pair_records(
            branch_results,
            primary_horizon=args.primary_horizon,
        )
        records.append(decision_record)

    summary = {
        'protocol': (
            'same deterministic actor; one satellite differs for one action; '
            'action[t] is scored from outcomes[t+1:t+H+1]'
        ),
        'checkpoint': str(args.checkpoint),
        'reference_trajectory': str(args.reference_trajectory),
        'split': args.split,
        'scene_id': scene_id,
        'horizons': horizons,
        'primary_horizon': args.primary_horizon,
        'top_k': args.top_k,
        'records': records,
    }
    json_dump(summary, str(args.output_root / 'summary.json'))
    print(f'[local-branch] wrote {args.output_root / "summary.json"}')


if __name__ == '__main__':
    main()
