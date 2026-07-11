"""TimeModel 可行性校准与推理门控工具。"""

import torch

__all__ = [
    'apply_feasibility_threshold',
    'binary_calibration_metrics',
    'hard_negative_indices',
]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def apply_feasibility_threshold(
    task_logits: torch.Tensor,
    feasibility_logits: torch.Tensor | None,
    threshold: float | None,
) -> torch.Tensor:
    """根据 TimeModel feasibility score 过滤任务 logits。"""
    if threshold is None:
        return task_logits
    if not 0 <= threshold <= 1:
        raise ValueError('feasibility threshold must be in [0, 1]')
    if feasibility_logits is None:
        raise ValueError(
            'feasibility_logits are required when threshold is enabled',
        )

    infeasible = feasibility_logits.sigmoid() <= threshold
    return task_logits.masked_fill(infeasible, float('-inf'))


def binary_calibration_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
    num_bins: int = 10,
) -> dict[str, object]:
    """计算二分类判别指标和 expected calibration error。"""
    probabilities = probabilities.detach().float().flatten().cpu()
    targets = targets.detach().bool().flatten().cpu()
    if probabilities.numel() == 0:
        raise ValueError('cannot calculate calibration metrics for empty input')
    if probabilities.shape != targets.shape:
        raise ValueError('probabilities and targets must have the same shape')
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1]')
    if num_bins <= 0:
        raise ValueError('num_bins must be positive')

    predictions = probabilities > threshold
    tp = int((predictions & targets).sum().item())
    fp = int((predictions & ~targets).sum().item())
    fn = int((~predictions & targets).sum().item())
    tn = int((~predictions & ~targets).sum().item())

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    fpr = _safe_ratio(fp, fp + tn)
    fnr = _safe_ratio(fn, fn + tp)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)

    calibration_bins: list[dict[str, float | int]] = []
    ece = 0.0
    support = probabilities.numel()
    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        mask = probabilities >= lower
        if index == num_bins - 1:
            mask &= probabilities <= upper
        else:
            mask &= probabilities < upper

        count = int(mask.sum().item())
        if count == 0:
            confidence = accuracy = 0.0
        else:
            confidence = float(probabilities[mask].mean().item())
            accuracy = float(targets[mask].float().mean().item())
            ece += count / support * abs(confidence - accuracy)
        calibration_bins.append(dict(
            lower=lower,
            upper=upper,
            count=count,
            confidence=confidence,
            accuracy=accuracy,
        ))

    return dict(
        threshold=threshold,
        support=support,
        positive_support=int(targets.sum().item()),
        negative_support=int((~targets).sum().item()),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        specificity=1 - fpr,
        fpr=fpr,
        fnr=fnr,
        f1=f1,
        accuracy=_safe_ratio(tp + tn, support),
        brier_score=float(((probabilities - targets.float())**2).mean().item()),
        ece=ece,
        calibration_bins=calibration_bins,
    )


def hard_negative_indices(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """返回高置信度预测可行、但离线标签为不可行的样本索引。"""
    probabilities = probabilities.detach().float().flatten()
    targets = targets.detach().bool().flatten()
    if probabilities.shape != targets.shape:
        raise ValueError('probabilities and targets must have the same shape')
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1]')
    return ((probabilities > threshold) & ~targets).nonzero().flatten()
