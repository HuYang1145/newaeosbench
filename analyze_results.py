#!/usr/bin/env python3
"""分析评估结果并与论文对比"""
import re
import sys
import json
from pathlib import Path

# 论文目标指标 (Val Unseen - AEOS-Former)
TARGET_METRICS = {
    'CS': 4.43,
    'CR': 35.42,
    'PC': 68.99
}

# CS 计算权重 (来自论文公式10)
W_CR = 0.6
W_PCR = 0.2
W_WCR = 0.2
W_TAT = 1/7
W_PC = 1/100

def calculate_cs(cr, pcr, wcr, tat, pc):
    """根据论文公式计算 CS"""
    cs = (W_CR * cr + W_PCR * pcr + W_WCR * wcr)**(-1) + W_TAT * tat + W_PC * pc
    return cs

def parse_eval_log(log_file):
    """从评估日志中提取指标"""
    metrics = {}
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # 查找指标 (支持百分比和小数格式)
        cr_match = re.search(r"'CR'[:\s]+([0-9.]+)", content)
        pcr_match = re.search(r"'PCR'[:\s]+([0-9.]+)", content)
        wcr_match = re.search(r"'WCR'[:\s]+([0-9.]+)", content)
        tat_match = re.search(r"'TAT'[:\s]+([0-9.]+)", content)
        pc_match = re.search(r"'PC'[:\s]+([0-9.]+)", content)

        if cr_match:
            metrics['CR'] = float(cr_match.group(1))
        if pcr_match:
            metrics['PCR'] = float(pcr_match.group(1))
        if wcr_match:
            metrics['WCR'] = float(wcr_match.group(1))
        if tat_match:
            metrics['TAT'] = float(tat_match.group(1))
        if pc_match:
            metrics['PC'] = float(pc_match.group(1))

        # 如果所有指标都找到了，计算 CS
        if all(k in metrics for k in ['CR', 'PCR', 'WCR', 'TAT', 'PC']):
            metrics['CS'] = calculate_cs(
                metrics['CR'], metrics['PCR'], metrics['WCR'],
                metrics['TAT'], metrics['PC']
            )

    except Exception as e:
        print(f"解析日志出错: {e}")

    return metrics

def compare_with_paper(metrics):
    """与论文指标对比"""
    print("\n" + "="*70)
    print("评估结果对比 (Val Unseen)")
    print("="*70)

    # 显示所有提取的指标
    print("\n提取的指标:")
    print("-"*70)
    for key in ['CR', 'PCR', 'WCR', 'TAT', 'PC', 'CS']:
        value = metrics.get(key, 'N/A')
        if isinstance(value, float):
            if key == 'CR':
                print(f"{key:<10} = {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"{key:<10} = {value:.4f}")
        else:
            print(f"{key:<10} = 未找到")

    # 与论文对比
    print("\n" + "="*70)
    print("与论文 AEOS-Former 对比:")
    print("-"*70)
    print(f"{'指标':<10} {'论文目标':<15} {'当前结果':<15} {'差异':<20}")
    print("-"*70)

    for key in ['CS', 'CR', 'PC']:
        target = TARGET_METRICS[key]
        current = metrics.get(key, 'N/A')

        if isinstance(current, float):
            if key == 'CR':
                current_pct = current * 100
                diff = current_pct - target
                diff_pct = (diff / target) * 100 if target != 0 else 0
                print(f"{key:<10} {target:<15.2f}% {current_pct:<15.2f}% {diff:+.2f}% ({diff_pct:+.1f}%)")
            else:
                diff = current - target
                diff_pct = (diff / target) * 100 if target != 0 else 0
                print(f"{key:<10} {target:<15.2f} {current:<15.2f} {diff:+.2f} ({diff_pct:+.1f}%)")
        else:
            print(f"{key:<10} {target:<15} {'未找到':<15} {'N/A':<20}")

    print("="*70)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python analyze_results.py <eval_log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    metrics = parse_eval_log(log_file)

    if metrics:
        compare_with_paper(metrics)

        # 生成优化建议
        from generate_report import generate_report
        report = generate_report(metrics, TARGET_METRICS)
        print("\n" + report)

        # 保存报告
        report_file = Path(log_file).parent / "optimization_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n报告已保存到: {report_file}")
    else:
        print("未能从日志中提取指标，请检查日志格式")
