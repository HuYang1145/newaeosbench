"""使用临时无 TAT 评分汇总评估输出。

当前复现辅助脚本，计算划分指标均值并按
``CS_no_TAT = 1/(0.6*CR+0.2*PCR+0.2*WCR) + PC_Wh/100`` 计算综合评分，
用于论文 TAT 定义尚未确定期间的临时方案。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRIC_KEYS = ("CR", "PCR", "WCR")


def _load_metrics(path: Path) -> dict[str, float] | None:
    if path.name.endswith("_trace.json") or path.name == "_trace_summary.json":
        return None
    try:
        data: Any = json.loads(path.read_text())
    except Exception:
        return None
    metrics = data.get("metrics", data) if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        return None
    if not all(key in metrics for key in METRIC_KEYS):
        return None
    output: dict[str, float] = {}
    for key in ("CR", "PCR", "WCR", "WPCR", "TAT", "PC", "PC_Wh"):
        if key in metrics and metrics[key] is not None:
            try:
                value = float(metrics[key])
            except (TypeError, ValueError):
                continue
            output[key] = value
    return output


def summarize_split(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob("*.json")):
        metrics = _load_metrics(path)
        if metrics is not None:
            records.append(metrics)
    if not records:
        raise RuntimeError(f"No metric records found under {root}")

    means: dict[str, float] = {}
    for key in ("CR", "PCR", "WCR", "WPCR", "TAT", "PC", "PC_Wh"):
        values = [record[key] for record in records if key in record and math.isfinite(record[key])]
        if values:
            means[key] = sum(values) / len(values)

    if "PC_Wh" not in means:
        if "PC" not in means:
            raise RuntimeError(f"No PC or PC_Wh values found under {root}")
        means["PC_Wh"] = means["PC"] / 3600.0

    quality = 0.6 * means["CR"] + 0.2 * means["PCR"] + 0.2 * means["WCR"]
    if quality <= 0:
        cs_no_tat = math.inf
    else:
        cs_no_tat = 1.0 / quality + means["PC_Wh"] / 100.0

    return {
        "root": str(root),
        "scene_count": len(records),
        "means": means,
        "table": {
            "CS_no_TAT": cs_no_tat,
            "CR_percent": means["CR"] * 100.0,
            "PCR_percent": means["PCR"] * 100.0,
            "WCR_percent": means["WCR"] * 100.0,
            "PC_Wh": means["PC_Wh"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("split_roots", nargs="+", type=Path)
    args = parser.parse_args()

    summaries = {root.name: summarize_split(root) for root in args.split_roots}
    output = {
        "score_definition": "CS_no_TAT = (0.6*CR + 0.2*PCR + 0.2*WCR)^(-1) + PC_Wh/100",
        "note": "TAT is logged when available but excluded from this temporary calibrated score.",
        "splits": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
