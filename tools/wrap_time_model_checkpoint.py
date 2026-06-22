"""包装独立的 TimeModel checkpoint 以供主模型加载。

动作模型期望 TimeModel 参数位于嵌套键前缀下。此脚本为 checkpoint 键
添加该前缀并保存轻量级包装后的 checkpoint。
"""

import argparse
import pathlib

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Wrap a standalone TimeModel checkpoint for nested loading',
    )
    parser.add_argument('input', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument(
        '--prefix',
        default='_transformer._time_model.',
        help='Prefix added to every parameter key',
    )
    return parser.parse_args()


def extract_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and 'state_dict' in obj:
        state_dict = obj['state_dict']
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(obj, dict):
        return obj
    raise TypeError(f'Unsupported checkpoint format: {type(obj)!r}')


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location='cpu')
    state_dict = extract_state_dict(checkpoint)

    wrapped = {
        f'{args.prefix}{key}': value
        for key, value in state_dict.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapped, args.output)
    print(
        f'Saved wrapped checkpoint with {len(wrapped)} tensors to '
        f'{args.output}',
    )


if __name__ == '__main__':
    main()
