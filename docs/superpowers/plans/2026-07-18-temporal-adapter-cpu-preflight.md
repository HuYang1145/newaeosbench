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

- [x] 先写 annotation 路由、缺失轨迹和重复 ID 的失败测试。
- [x] 运行定向测试，确认新 API 缺失导致 RED。
- [x] 实现 `(annotation, split, data_root)` 路由、完整性元数据和 `--workers` 多进程审计，保留旧 positional root 用法。
- [x] 运行定向测试并确认 GREEN。

## Task 2: Scene-level 隔离审计

**Files:**
- Create: `tools/audit_temporal_scene_splits.py`
- Create: `tests/test_temporal_scene_splits.py`

- [x] 先写精确场景指纹重叠、annotation 内重复 ID、缺失静态文件的失败测试。
- [x] 运行测试，确认模块缺失导致 RED。
- [x] 实现 train/val_seen/val_unseen 的 annotation 计数、路径完整性、静态场景 SHA256 指纹和重叠报告。
- [x] 运行测试并确认 GREEN。

## Task 3: 固化 10k pilot 与 8+8 Val 命令

**Files:**
- Create: `scripts/train_temporal_adapter_p0_10k.sh`
- Create: `scripts/eval_temporal_adapter_p0_8.sh`
- Create: `tests/test_temporal_preflight_scripts.py`

- [x] 先写脚本配置、checkpoint、输出目录、Temporal Adapter CLI 参数和 GPU 忙时拒绝启动的静态测试。
- [x] 运行测试，确认脚本缺失导致 RED。
- [x] 实现包装脚本；只准备，不在本计划中执行 GPU 训练或 Basilisk Val。
- [x] 运行脚本测试和 `bash -n`，确认 GREEN。

## Task 4: 执行 CPU 预检并保存结果

**Outputs:**
- `/home/hy/data/newaeosbench/work_dirs/temporal_adapter_p0_preflight/stage3_label_coverage.json`
- `/home/hy/data/newaeosbench/work_dirs/temporal_adapter_p0_preflight/scene_split_audit.json`
- `/home/hy/data/newaeosbench/work_dirs/temporal_adapter_p0_preflight/*.log`

- [x] 先运行小样本审计并估算全量耗时。
- [x] 运行 Stage3 annotation 全量 5/15/30/300 秒标签审计。
- [x] 运行 train/val_seen/val_unseen scene-level 隔离审计。
- [x] 汇总 observed/censored、正样本率、数据数量偏差和进入 P0-B 前的阻塞项。
- [x] 运行相关测试、`py_compile`、`bash -n` 与 `git diff --check`。

## Task 5: 根据全量计数修正类别不平衡

**Files:**
- Modify: `constellation/new_transformers/temporal_adapter.py`
- Modify: `constellation/new_transformers/model.py`
- Modify: `constellation/new_transformers/config_temporal_adapter_p0.py`
- Modify: `tests/test_temporal_adapter.py`
- Modify: `tests/test_temporal_model.py`

- [x] 先写 masked BCE 正类权重和配置值失败测试。
- [x] 运行测试，确认权重 API 缺失导致 RED。
- [x] 为 next 和各 horizon 分别接入训练集 `negative / positive` 权重，默认 `None` 时保持旧行为。
- [x] 将 Stage3 全量计数固化到 P0-B 配置，运行定向测试并确认 GREEN。

## 2026-07-18 CPU 预检结果

- Stage3 annotation：13,849 个唯一 scene；epoch 分布为 1: 3,596、2: 3,987、3: 6,266。
- 完整扫描 835,337,638 条非空执行边；总耗时 3,122.55 秒，8 个单线程 worker，未使用 GPU。
- train/val_seen/val_unseen 数量分别为 13,849/64/64；完整静态场景 SHA256 指纹两两重叠均为 0。
- 5/15/30/300 秒 `visible` censored 率分别为 1.71%/5.40%/10.55%/51.63%。
- 5/15/30/300 秒 `progress` censored 率分别为 1.09%/2.80%/5.12%/23.04%。
- 5/15/30/300 秒 `completion` censored 率分别为 1.20%/3.26%/6.13%/30.98%。
- next-step completion 正样本率只有 0.335%，5 秒 completion 在 observed 样本中的正样本率只有 1.69%；已按全量训练集 `negative / positive` 接入独立正类权重。
- 重复执行边占 67.51%，其中 93.19% 下一秒不可见；Temporal Adapter 可继续做 P0-B outcome 训练，但正式行为评估必须把重复冗余作为硬门槛，P0 本身不宣称解决跨卫星联合分配。
- 13,849 是 Stage3 `tau_e` 筛选后的 annotation 数量，不等同于论文原始 train 16,218 条轨迹；论文对齐报告必须保留这一区别。
- 最终 CPU 回归：`185 passed`；生产文件 `py_compile`、两个包装脚本 `bash -n`、`git diff --check` 均退出 0。
