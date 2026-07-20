"""批量生成受控局部 Graph-Q 训练数据。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def discover_reference_trajectories(
    reference_root: Path,
    *,
    split: str,
    limit: int | None,
) -> list[Path]:
    paths = sorted((reference_root / split).glob('*/*.pth'))
    if limit is not None:
        if limit <= 0:
            raise ValueError('limit must be positive')
        paths = paths[:limit]
    if not paths:
        raise ValueError('no reference trajectories found')
    return paths


def build_scene_command(
    *,
    python: Path,
    checkpoint: Path,
    reference: Path,
    output_root: Path,
    split: str,
    horizons: tuple[int, ...],
    primary_horizon: int,
    max_decisions: int,
    top_k: int,
    device: str,
    overwrite: bool,
) -> list[str]:
    scene_id = int(reference.stem)
    command = [
        str(python),
        'tools/generate_local_action_branches.py',
        str(checkpoint),
        str(reference),
        str(output_root / f'scene_{scene_id:05}'),
        '--split',
        split,
        '--scene-id',
        str(scene_id),
        '--device',
        device,
        '--horizons',
        *[str(horizon) for horizon in horizons],
        '--primary-horizon',
        str(primary_horizon),
        '--max-decisions',
        str(max_decisions),
        '--top-k',
        str(top_k),
    ]
    if overwrite:
        command.append('--overwrite')
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate a multi-scene controlled local Graph-Q dataset'
    )
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reference_root', type=Path)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--split', default='train')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument(
        '--horizons', type=int, nargs='+', default=[180, 300, 600]
    )
    parser.add_argument('--primary-horizon', type=int, default=300)
    parser.add_argument('--max-decisions', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--scene-workers', type=int, default=2)
    parser.add_argument('--threads-per-scene', type=int, default=24)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.scene_workers, args.threads_per_scene) <= 0:
        raise ValueError('worker and thread counts must be positive')
    horizons = tuple(sorted(set(args.horizons)))
    if not horizons or args.primary_horizon not in horizons:
        raise ValueError('primary horizon must be included in horizons')
    references = discover_reference_trajectories(
        args.reference_root,
        split=args.split,
        limit=args.limit,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_root = args.output_root / 'logs'
    logs_root.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = str(args.threads_per_scene)
    environment['MKL_NUM_THREADS'] = str(args.threads_per_scene)
    environment['PYTHONPATH'] = str(ROOT)
    environment.setdefault('MPLCONFIGDIR', '/tmp/aeos_mpl')

    def run(reference: Path) -> dict[str, object]:
        scene_id = int(reference.stem)
        scene_root = args.output_root / f'scene_{scene_id:05}'
        summary_path = scene_root / 'summary.json'
        if summary_path.is_file() and not args.overwrite:
            return {'scene_id': scene_id, 'status': 'cached', 'returncode': 0}
        command = build_scene_command(
            python=Path(sys.executable),
            checkpoint=args.checkpoint,
            reference=reference,
            output_root=args.output_root,
            split=args.split,
            horizons=horizons,
            primary_horizon=args.primary_horizon,
            max_decisions=args.max_decisions,
            top_k=args.top_k,
            device=args.device,
            overwrite=args.overwrite,
        )
        log_path = logs_root / f'scene_{scene_id:05}.log'
        with log_path.open('w', encoding='utf-8') as log_file:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {
            'scene_id': scene_id,
            'status': 'completed' if completed.returncode == 0 else 'failed',
            'returncode': completed.returncode,
            'log': str(log_path),
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.scene_workers) as executor:
        futures = {
            executor.submit(run, reference): reference
            for reference in references
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[local-dataset] scene={result['scene_id']} "
                f"status={result['status']}",
                flush=True,
            )
    results.sort(key=lambda item: int(item['scene_id']))
    summary = {
        'checkpoint': str(args.checkpoint),
        'reference_root': str(args.reference_root),
        'split': args.split,
        'horizons': horizons,
        'primary_horizon': args.primary_horizon,
        'max_decisions': args.max_decisions,
        'top_k': args.top_k,
        'scene_workers': args.scene_workers,
        'threads_per_scene': args.threads_per_scene,
        'results': results,
    }
    (args.output_root / 'dataset_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    failures = [result for result in results if result['returncode'] != 0]
    if failures:
        raise RuntimeError(f'{len(failures)} scene rollouts failed')


if __name__ == '__main__':
    main()
