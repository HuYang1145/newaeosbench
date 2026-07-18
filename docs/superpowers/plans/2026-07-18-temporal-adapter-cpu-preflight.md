# Temporal Adapter CPU Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不占用 GPU 的前提下，完成 Stage3 Temporal Adapter 标签覆盖率、scene-level 数据隔离审计，并准备可重复运行的 10k pilot 与 8+8 Val 包装脚本。

**Architecture:** 扩展现有 `audit_multi_horizon_edge_labels.py`，使其严格按 annotation 中的 `(split, id, epoch)` 路由轨迹，并支持受控多进程 CPU 聚合。新增独立 scene split 审计，只比较场景静态输入的内容指纹；训练和评估包装脚本只做前置检查和命令固化，本阶段不启动 GPU 任务。

**Tech Stack:** Python 3.11、PyTorch CPU、pytest、Bash、JSON。

---

## Task 1: Annotation-aware 标签覆盖率审计

**Files:**
- Modify: `tools/audit_multi_horizon_edge_labels.py`
- Modify: `tests/test_multi_horizon_edge_labels.py`

- [ ] 先写 annotation 路由、缺失轨迹和重复 ID 的失败测试。
- [ ] 运行定向测试，确认新 API 缺失导致 RED。
- [ ] 实现 `(annotation, split, data_root)` 路由、完整性元数据和 `--workers` 多进程审计，保留旧 positional root 用法。
- [ ] 运行定向测试并确认 GREEN。

## Task 2: Scene-level 隔离审计

**Files:**
- Create: `tools/audit_temporal_scene_splits.py`
- Create: `tests/test_temporal_scene_splits.py`

- [ ] 先写精确场景指纹重叠、annotation 内重复 ID、缺失静态文件的失败测试。
- [ ] 运行测试，确认模块缺失导致 RED。
- [ ] 实现 train/val_seen/val_unseen 的 annotation 计数、路径完整性、静态场景 SHA256 指纹和重叠报告。
- [ ] 运行测试并确认 GREEN。

## Task 3: 固化 10k pilot 与 8+8 Val 命令

**Files:**
- Create: `scripts/train_temporal_adapter_p0_10k.sh`
- Create: `scripts/eval_temporal_adapter_p0_8.sh`
- Create: `tests/test_temporal_preflight_scripts.py`

- [ ] 先写脚本配置、checkpoint、输出目录、Temporal Adapter CLI 参数和 GPU 忙时拒绝启动的静态测试。
- [ ] 运行测试，确认脚本缺失导致 RED。
- [ ] 实现包装脚本；只准备，不在本计划中执行 GPU 训练或 Basilisk Val。
- [ ] 运行脚本测试和 `bash -n`，确认 GREEN。

## Task 4: 执行 CPU 预检并保存结果

**Outputs:**
- `/home/hy/data/newaeosbench/work_dirs/temporal_adapter_p0_preflight/stage3_label_coverage.json`
- `/home/hy/data/newaeosbench/work_dirs/temporal_adapter_p0_preflight/scene_split_audit.json`
- `/home/hy/data/newaeosbench/work_dirs/temporal_adapter_p0_preflight/*.log`

- [ ] 先运行小样本审计并估算全量耗时。
- [ ] 运行 Stage3 annotation 全量 5/15/30/300 秒标签审计。
- [ ] 运行 train/val_seen/val_unseen scene-level 隔离审计。
- [ ] 汇总 observed/censored、正样本率、数据数量偏差和进入 P0-B 前的阻塞项。
- [ ] 运行相关测试、`py_compile`、`bash -n` 与 `git diff --check`。
