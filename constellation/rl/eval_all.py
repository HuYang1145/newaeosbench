import argparse
import atexit
import signal
import importlib
import os
import pathlib
import pickle
from functools import partial
from itertools import count
from typing import Any

import numpy as np
import pandas as pd
import numpy.typing as npt
import todd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from todd.configs import PyConfig
from todd.patches.py_ import DictAction, json_dump
from todd.utils import init_seed

from .environment import Environment, Observation, null_observation
from .controller_environment import ControllerEnvironment
from .coordination_diagnostics import (
    SceneRecorder,
    build_step_diagnostics,
    map_topk_task_ids,
    summarize_scene_results,
)
from .policy import Policy
from .owner_assignment import resolve_owner_assignments
from constellation.new_transformers.model import GLOBALS

COMPLETION_RATE_THRESHOLD = 0.01


def limit_annotations(
    annotations: list[int],
    max_scenes: int | None,
) -> list[int]:
    """限制评估场景数，用于不改变 annotation 的小规模消融。"""
    if max_scenes is None:
        return annotations
    if max_scenes <= 0:
        raise ValueError('max_scenes must be positive')
    return annotations[:max_scenes]


class EvalEnvironment(ControllerEnvironment):

    @classmethod
    def build(
        cls,
        world_size: int,
        gen_trajectory_dir: pathlib.Path,
        *args,
        **kwargs,
    ) -> SubprocVecEnv:
        assert world_size > 0, "world_size must be greater than 0"
        return SubprocVecEnv([
            partial(
                cls,
                *args,
                world_size=int(os.environ['WORLD_SIZE']) * world_size,
                rank=int(os.environ['RANK']) * world_size + i,
                gen_trajectory_dir=gen_trajectory_dir,
                **kwargs,
            ) for i in range(world_size)
        ])

    def __init__(
        self,
        *args,
        world_size: int,
        rank: int,
        retry_from: pathlib.Path | None = None,
        max_scenes: int | None = None,
        gen_trajectory_dir: pathlib.Path | None = None,
        enable_coordination_diagnostics: bool = False,
        enable_owner_assignment: bool = False,
        owner_continuation_bonus: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._world_size = world_size
        self._rank = rank
        self._counter = -1
        self._gen_trajectory_dir = gen_trajectory_dir
        self._enable_coordination_diagnostics = (
            enable_coordination_diagnostics
        )
        self._coordination_before: dict[str, Any] | None = None
        self._coordination_step: dict[str, Any] | None = None
        self._enable_owner_assignment = enable_owner_assignment
        self._owner_continuation_bonus = owner_continuation_bonus
        self._previous_owner_task_ids: npt.NDArray[np.int64] | None = None

        if retry_from is not None:
            df = pd.read_csv(
                retry_from,
                names=['id', 'completion_rate'],
                index_col='id',
            )
            completion_rates: dict[int, float] = \
                df['completion_rate'].to_dict()
            self._annotations = [
                annotation for annotation in self._annotations if
                completion_rates.get(annotation, 0) < COMPLETION_RATE_THRESHOLD
            ]
        self._annotations = limit_annotations(
            self._annotations,
            max_scenes,
        )

    @property
    def _index(self) -> int:
        return self._counter * self._world_size + self._rank

    @property
    def all_done(self) -> bool:
        return self._index >= len(self._annotations)

    def _get_annotation(self) -> int:
        return self._annotations[self._index]

    def reset(self, *args, **kwargs) -> tuple[Observation, dict[str, Any]]:
        self._previous_owner_task_ids = None
        if self._counter != -1 and not self.all_done:
            id_ = self._get_annotation()
            save_dir = self._gen_trajectory_dir / f'{id_ // 1000:02d}'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'{id_:05d}.pth'
            # self._logger.pth_dump(save_path)

        self._counter += 1

        if self.all_done:
            return null_observation, dict(all_done=True)

        obs, info = super().reset(*args, **kwargs)
        print(info)
        # with open("./test.pkl", "wb") as f:
        #     pickle.dump((obs, info), f)
        # self._logger = Logger(
        #     task_manager=self._task_manager,
        #     constellation=None,
        #     work_dir=None,
        # )
        return obs, info

    def step(
        self,
        action: npt.NDArray[np.uint16],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self.all_done:
            return null_observation, 0.0, False, False, dict(all_done=True)

        if isinstance(action, tuple):  # TODO: one action
            action, auxiliary = action
            if isinstance(auxiliary, dict):
                pred_mask = auxiliary.get('pred_mask')
                if pred_mask is not None:
                    self._pred_mask = pred_mask
                owner_logits = auxiliary.get('owner_logits')
                if owner_logits is not None:
                    if not self._enable_owner_assignment:
                        raise RuntimeError('owner assignment is not enabled')
                    controller = self._require_controller()
                    action, self._previous_owner_task_ids = (
                        resolve_owner_assignments(
                            owner_logits,
                            task_ids=(
                                controller.task_manager.ongoing_tasks.ids
                                .tolist()
                            ),
                            num_satellites=(
                                controller.environment.num_satellites
                            ),
                            previous_owner_task_ids=(
                                self._previous_owner_task_ids
                            ),
                            continuation_bonus=(
                                self._owner_continuation_bonus
                            ),
                        )
                    )
            else:
                self._pred_mask = auxiliary

        if self._enable_coordination_diagnostics:
            controller = self._require_controller()
            self._coordination_before = dict(
                time_step=controller.environment.timer.time,
                ongoing_task_ids=(
                    controller.task_manager.ongoing_tasks.ids.tolist()
                ),
                all_task_ids=controller.task_manager.taskset.ids.tolist(),
                progress_before=controller.task_manager.progress.clone(),
            )

        if self._controller.task_manager.progress.any(
        ) and self._controller.environment.timer.time % 50 == 0 and self._controller.environment.timer.time <= 1800:
            todd.logger.info(
                "env_rank %s sim_step %d progress_sum %d finished_num %d",
                self._rank,
                self._controller.environment.timer.time,
                self._controller.task_manager.progress.sum(),
                self._controller.task_manager.num_succeeded_tasks,
            )

        # self._logger.add_time_csv(
        #     constellation=self._simulator.get_constellation(),
        #     task_id_list=(action[:self._simulator.num_satellites]
        #                   - 1).tolist(),
        #     is_visible=self._simulator.is_visible(self._task_manager.tasks),
        # )

        observation, reward, terminated, truncated, info = (
            super().step(action)
        )

        id_ = self._get_annotation()
        info.update(id=id_)
        if self._enable_coordination_diagnostics:
            assert self._coordination_step is not None
            step_diagnostics = self._coordination_step
            if terminated or truncated:
                controller = self._require_controller()
                all_task_ids = set(
                    controller.task_manager.taskset.ids.tolist(),
                )
                succeeded_task_ids = set(
                    controller.task_manager.succeeded_tasks.ids.tolist(),
                )
                failed_task_ids = set(
                    controller.task_manager.failed_tasks.ids.tolist(),
                )
                step_diagnostics.update(
                    succeeded_task_ids=sorted(succeeded_task_ids),
                    failed_task_ids=sorted(failed_task_ids),
                    open_task_ids=sorted(
                        all_task_ids - succeeded_task_ids - failed_task_ids,
                    ),
                )
            info['coordination_diagnostics'] = step_diagnostics
            self._coordination_before = None
            self._coordination_step = None

        return observation, reward, terminated, truncated, info

    def _after_policy_action(
        self,
        action: npt.NDArray[np.int32],
    ) -> None:
        if not self._enable_coordination_diagnostics:
            return
        assert self._coordination_before is not None
        controller = self._require_controller()
        self._coordination_step = build_step_diagnostics(
            time_step=self._coordination_before['time_step'],
            action=action[:controller.environment.num_satellites].tolist(),
            ongoing_task_ids=self._coordination_before['ongoing_task_ids'],
            all_task_ids=self._coordination_before['all_task_ids'],
            progress_before=self._coordination_before['progress_before'],
            progress_after=controller.task_manager.progress,
            is_visible=controller.memo['is_visible'],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--load-model-from', nargs='+', default=[])
    parser.add_argument('--load-ppo-from', type=pathlib.Path, default=None)
    parser.add_argument('--load-from')
    parser.add_argument('--auto-resume', action='store_true')
    parser.add_argument('--retry-from', type=pathlib.Path, default=None)
    parser.add_argument('--max-scenes', type=int, default=None)
    parser.add_argument('--feasibility-threshold', type=float, default=None)
    parser.add_argument(
        '--feasibility-penalty-threshold',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--feasibility-penalty-strength',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--coordination-diagnostics-top-k',
        type=int,
        default=None,
    )
    parser.add_argument('--use-assignment-head', action='store_true')
    parser.add_argument(
        '--assignment-head-hidden-width',
        type=int,
        default=32,
    )
    parser.add_argument('--use-temporal-adapter', action='store_true')
    parser.add_argument(
        '--temporal-adapter-hidden-width',
        type=int,
        default=64,
    )
    parser.add_argument(
        '--temporal-residual-scale',
        type=float,
        default=0.25,
    )
    parser.add_argument('--owner-assignment', action='store_true')
    parser.add_argument(
        '--owner-continuation-bonus',
        type=float,
        default=0.25,
    )
    args = parser.parse_args()
    return args


def build_policy_kwargs(
    load_model_from: list[str],
    feasibility_threshold: float | None,
    *,
    feasibility_penalty_threshold: float | None,
    feasibility_penalty_strength: float | None,
    use_assignment_head: bool = False,
    assignment_head_hidden_width: int = 32,
    use_temporal_adapter: bool = False,
    temporal_adapter_hidden_width: int = 64,
    temporal_residual_scale: float = 0.25,
) -> dict[str, Any]:
    kwargs = dict(
        load_model_from=load_model_from,
        actor_model_kwargs=dict(
            use_constraint_module=True,
            use_sdpa=True,
            feasibility_threshold=feasibility_threshold,
            feasibility_penalty_threshold=feasibility_penalty_threshold,
            feasibility_penalty_strength=feasibility_penalty_strength,
        ),
    )
    if use_assignment_head:
        kwargs['actor_model_kwargs'].update(
            use_assignment_head=True,
            assignment_head_hidden_width=assignment_head_hidden_width,
        )
    if use_temporal_adapter:
        kwargs['actor_model_kwargs'].update(
            use_temporal_adapter=True,
            temporal_adapter_hidden_width=temporal_adapter_hidden_width,
            temporal_horizons=(5, 15, 30, 300),
            temporal_residual_scale=temporal_residual_scale,
        )
    return kwargs


def build_eval_metadata(
    *,
    split: str,
    world_size: int,
    max_scenes: int | None,
    load_model_from: list[str],
    feasibility_threshold: float | None,
    feasibility_penalty_threshold: float | None,
    feasibility_penalty_strength: float | None,
    coordination_diagnostics_top_k: int | None,
    use_assignment_head: bool = False,
    assignment_head_hidden_width: int = 32,
    use_temporal_adapter: bool = False,
    temporal_adapter_hidden_width: int = 64,
    temporal_residual_scale: float = 0.25,
    owner_assignment: bool = False,
    owner_continuation_bonus: float = 0.25,
) -> dict[str, Any]:
    """记录影响评估可复现性的关键参数。"""
    metadata = dict(
        split=split,
        world_size=world_size,
        max_scenes=max_scenes,
        load_model_from=load_model_from,
        feasibility_threshold=feasibility_threshold,
        feasibility_penalty_threshold=feasibility_penalty_threshold,
        feasibility_penalty_strength=feasibility_penalty_strength,
        coordination_diagnostics_top_k=coordination_diagnostics_top_k,
        use_assignment_head=use_assignment_head,
        assignment_head_hidden_width=assignment_head_hidden_width,
        owner_assignment=owner_assignment,
        owner_continuation_bonus=owner_continuation_bonus,
    )
    if use_temporal_adapter:
        metadata.update(
            use_temporal_adapter=True,
            temporal_adapter_hidden_width=temporal_adapter_hidden_width,
            temporal_residual_scale=temporal_residual_scale,
        )
    return metadata


'''
CUDA_VISIBLE_DEVICES=0,3,4,5,6,7 auto_torchrun -m rl.eval_all \
    rl_loaded_eval \
    rl/config_eval.py \
    --load-model-from './work_dirs/model610000.pth'
'''


def main() -> None:
    args = parse_args()
    if (
        args.coordination_diagnostics_top_k is not None
        and args.coordination_diagnostics_top_k <= 0
    ):
        raise ValueError('coordination diagnostics top-k must be positive')
    if args.assignment_head_hidden_width <= 0:
        raise ValueError('assignment head hidden width must be positive')
    if args.temporal_adapter_hidden_width <= 0:
        raise ValueError('temporal adapter hidden width must be positive')
    if args.temporal_residual_scale < 0:
        raise ValueError('temporal residual scale must be non-negative')
    if args.owner_continuation_bonus < 0:
        raise ValueError('owner continuation bonus must be non-negative')
    config = PyConfig.load(args.config, **args.config_options)
    config.override(args.override)
    init_seed(args.seed)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    work_dir = pathlib.Path('work_dirs') / f'rl_eval_{args.name}'
    work_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_eval_metadata(
        split=config.environment.split,
        world_size=config.environment.world_size,
        max_scenes=args.max_scenes,
        load_model_from=args.load_model_from,
        feasibility_threshold=args.feasibility_threshold,
        feasibility_penalty_threshold=args.feasibility_penalty_threshold,
        feasibility_penalty_strength=args.feasibility_penalty_strength,
        coordination_diagnostics_top_k=(
            args.coordination_diagnostics_top_k
        ),
        use_assignment_head=args.use_assignment_head,
        assignment_head_hidden_width=args.assignment_head_hidden_width,
        use_temporal_adapter=args.use_temporal_adapter,
        temporal_adapter_hidden_width=args.temporal_adapter_hidden_width,
        temporal_residual_scale=args.temporal_residual_scale,
        owner_assignment=args.owner_assignment,
        owner_continuation_bonus=args.owner_continuation_bonus,
    )
    json_dump(metadata, str(work_dir / 'eval_metadata.json'))

    gen_trajectory_dir = work_dir / config.environment.split
    gen_trajectory_dir.mkdir(parents=True, exist_ok=True)

    environment = EvalEnvironment.build(
        retry_from=args.retry_from,
        max_scenes=args.max_scenes,
        gen_trajectory_dir=gen_trajectory_dir,
        enable_coordination_diagnostics=(
            args.coordination_diagnostics_top_k is not None
        ),
        enable_owner_assignment=args.owner_assignment,
        owner_continuation_bonus=args.owner_continuation_bonus,
        **config.environment,
    )
    atexit.register(environment.close)
    signal.signal(signal.SIGINT, lambda s, f: environment.close())
    signal.signal(signal.SIGTERM, lambda s, f: environment.close())

    device = torch.device(int(os.environ['RANK']) % torch.cuda.device_count())
    torch.cuda.set_device(device)

    if args.load_model_from != []:
        algorithm = PPO(
            Policy,
            environment,
            policy_kwargs=build_policy_kwargs(
                args.load_model_from,
                args.feasibility_threshold,
                feasibility_penalty_threshold=(
                    args.feasibility_penalty_threshold
                ),
                feasibility_penalty_strength=args.feasibility_penalty_strength,
                use_assignment_head=args.use_assignment_head,
                assignment_head_hidden_width=(
                    args.assignment_head_hidden_width
                ),
                use_temporal_adapter=args.use_temporal_adapter,
                temporal_adapter_hidden_width=(
                    args.temporal_adapter_hidden_width
                ),
                temporal_residual_scale=args.temporal_residual_scale,
            ),
            tensorboard_log=str(work_dir),
            seed=args.seed,
            device=device,
            **config.algorithm,
        )
    else:
        assert args.load_ppo_from is not None
        algorithm = PPO.load(
            path=args.load_ppo_from,
            env=environment,
            device=device,
        )

    observations = environment.reset()
    scene_recorders: dict[int, SceneRecorder] = {}
    scene_results: list[dict[str, object]] = []
    for i in count():
        if i % config.log_interval == 0:
            todd.logger.info("rank %s step %d", os.environ['RANK'], i)

        if (
            args.coordination_diagnostics_top_k is not None
            or args.owner_assignment
        ):
            GLOBALS['capture_actor_logits'] = True
        actions, _ = algorithm.predict(
            observations,
            deterministic=True,  # type: ignore[arg-type]
        )
        actor_logits = GLOBALS.pop('actor_logits', None)
        pred_mask = GLOBALS.pop('pred_mask', None)
        if pred_mask is not None or args.owner_assignment:
            if args.owner_assignment:
                assert actor_logits is not None
            action_payloads: list[dict[str, object]] = []
            for env_index in range(len(actions)):
                payload: dict[str, object] = {}
                if pred_mask is not None:
                    payload['pred_mask'] = pred_mask[env_index].cpu()
                if args.owner_assignment:
                    payload['owner_logits'] = (
                        actor_logits[env_index].numpy()
                    )
                action_payloads.append(payload)
            actions = list(zip(actions, action_payloads))
        observations, _, dones, infos = environment.step(actions)

        if args.coordination_diagnostics_top_k is not None:
            assert actor_logits is not None
            for env_index, (done, info) in enumerate(zip(dones, infos)):
                step_diagnostics = info.get('coordination_diagnostics')
                if step_diagnostics is None:
                    continue
                scene_id = int(info['id'])
                recorder = scene_recorders.setdefault(
                    scene_id,
                    SceneRecorder(
                        scene_id=scene_id,
                        top_k=args.coordination_diagnostics_top_k,
                    ),
                )
                topk_task_ids = map_topk_task_ids(
                    actor_logits[env_index:env_index + 1],
                    ongoing_task_ids=[
                        step_diagnostics['ongoing_task_ids']
                    ],
                    num_satellites=[
                        len(step_diagnostics['assignment'])
                    ],
                    top_k=args.coordination_diagnostics_top_k,
                )[0]
                recorder.record_step(
                    time_step=step_diagnostics['time_step'],
                    assignment=step_diagnostics['assignment'],
                    topk_task_ids=topk_task_ids,
                    selected_visible=step_diagnostics['selected_visible'],
                    progress_made_task_ids=(
                        step_diagnostics['progress_made_task_ids']
                    ),
                )
                if done:
                    scene_results.append(recorder.finalize(
                        succeeded_task_ids=(
                            step_diagnostics['succeeded_task_ids']
                        ),
                        failed_task_ids=(
                            step_diagnostics['failed_task_ids']
                        ),
                        open_task_ids=(
                            step_diagnostics['open_task_ids']
                        ),
                    ))
                    del scene_recorders[scene_id]

        for done, info in zip(dones, infos):
            if done and not info.get('all_done', False):
                id_ = info['id']
                metrics = info['metrics']

                todd.logger.info(
                    f"rank %s step %d {id_=}\n{metrics=}",
                    os.environ['RANK'],
                    i,
                )

                json_path = gen_trajectory_dir / f'{id_ // 1000:02d}' / f'{id_:05d}.json'
                json_dump(metrics, str(json_path))

        if all(info.get('all_done', False) for info in infos):
            todd.logger.info(
                "rank %s step %d all done",
                os.environ['RANK'],
                i,
            )
            break
    GLOBALS.pop('capture_actor_logits', None)
    if args.coordination_diagnostics_top_k is not None:
        json_dump(
            dict(
                split=config.environment.split,
                top_k=args.coordination_diagnostics_top_k,
                summary=summarize_scene_results(scene_results),
                scenes=scene_results,
            ),
            str(work_dir / 'coordination_diagnostics.json'),
        )
    environment.close()


if __name__ == '__main__':
    main()
