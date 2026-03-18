#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def summarize(result_dir):
    metrics = {'CR': [], 'WCR': [], 'PCR': [], 'WPCR': [], 'TAT': [], 'PC': []}

    for json_file in Path(result_dir).rglob('*.json'):
        with open(json_file) as f:
            data = json.load(f)
            for k in metrics:
                if k in data:
                    metrics[k].append(data[k])

    print(f"评估结果汇总 - {result_dir}")
    print("=" * 60)
    print(f"场景数量: {len(metrics['CR'])}")
    print()
    for k, v in metrics.items():
        if v:
            avg = sum(v) / len(v)
            print(f"{k:6s}: 平均={avg:8.2f}  最小={min(v):8.2f}  最大={max(v):8.2f}")

if __name__ == '__main__':
    summarize(sys.argv[1])
