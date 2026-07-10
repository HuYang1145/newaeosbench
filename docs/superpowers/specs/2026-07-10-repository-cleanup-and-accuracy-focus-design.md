# 仓库清理与模型准确率主线设计

## 目标

在不破坏已有实验状态的前提下，完成当前工作树整理，修复本机
`data/satellites/val_seen` 失效符号链接，统一 `README.md` 与 `TODO.md`
的研究主线，验证 `JointDataset` I/O 优化，并将整理后的提交使用
`git push --force-with-lease origin main` 发布到 GitHub。

项目后续主任务统一为：探索如何提高 AEOS-Former 的调度完成率，重点观察
`CR`、`PCR`、`WCR`，同时保留 `PC_Wh` 作为资源代价约束。benchmark
分层和失败原因统计用于诊断模型，而不是替代模型改进本身。

## 当前事实

- `data/satellites/val_seen` 指向旧机器绝对路径
  `/data/wlt/projects/AEOSBench/data/satellites/train`，当前目标不存在。
- 本机存在可用的 `data/satellites/train`，因此应把链接改成相对链接
  `val_seen -> train`。
- `data/` 被 `.gitignore` 排除，符号链接修复只属于本机实验环境，不进入 Git。
- 当前工作树包含 `JointDataset` 重复轨迹读取优化、对应测试、停止旧可观测性
  过滤主线的代码和脚本删除，以及 `TODO.md` 研究方向调整。
- `docs/实验复现报告.md` 的论文式 200k 联合训练结果表明：Val Seen 的
  `CR` 为 36.81% 到 37.50%，Val Unseen 为 41.80% 到 42.82%，Test 为
  21.24% 到 23.28%。因此“准确率 30% 多”只适合概括部分 split，严格表述
  应使用任务完成率 `CR/PCR/WCR` 并按 split 分别报告。

## 变更范围

### 1. 本机数据链接

使用相对链接替换失效绝对链接：

```text
data/satellites/val_seen -> train
```

验证链接能够解析到 `data/satellites/train`，并确认至少能枚举候选卫星 JSON。
不修改或删除 `data/` 中其他文件。

### 2. `JointDataset` I/O 优化

保留 `constellation/new_transformers/dataset.py` 中的现有重构，使
`JointDataset.__getitem__()` 对同一轨迹只调用一次 `torch.load`，并复用完整
轨迹构造动作 batch 与约束 batch。

保留 `tests/test_joint_dataset_io.py`，运行定向测试、语法检查和真实数据兼容性
检查。临时备份 `dataset.py.bak_20260623_102131` 不进入提交；通过 `.gitignore`
忽略此类时间戳备份，避免污染工作树，不删除备份实体。

### 3. 退出旧可观测性过滤主线

接受当前工作树对 `tools/generate_constellations_and_tasksets.py` 的回退，使场景
生成工具恢复为基础生成职责。删除已经退出主线的两个实验脚本：

- `scripts/eval_observable_filtered/run_stage3_200k_96core_eval.sh`
- `scripts/taskset_filtering/generate_filtered_annotation_tasksets.py`

已有筛选数据、日志和评估摘要继续作为历史实验状态保留，不删除
`data/`、`work_dirs/` 或 `docs/observable_filtered_stage3_eval_summary.md`。

### 4. 文档统一

`README.md` 与 `TODO.md` 使用同一研究口径：

1. 当前第一目标是提升模型调度完成率，而不是继续扩展可观测性过滤。
2. `CR/PCR/WCR` 是“准确率”讨论中的正式指标名称，不把三者混成单一 accuracy。
3. benchmark 难度分层、物理可观测性和失败原因统计是诊断手段。
4. 优先研究监督信号质量、动作分配目标、约束建模、专家迭代与推理策略。
5. 所有提升必须在 Val Seen、Val Unseen、Test 上分别报告完成率，并同时记录
   `PC_Wh`，避免以功耗显著上升换取不可解释的完成率增益。

`docs/实验复现报告.md` 中的已测数值保持不改，作为当前基线证据。

## 提交和发布

变更按职责拆分：

1. 设计说明提交。
2. `JointDataset` 优化、测试和备份忽略规则提交。
3. 旧可观测性过滤流程清理与研究主线文档提交。

发布前运行全部相关定向测试，检查 `git diff --check`，重新获取远端状态，并确认
`origin/main` 没有在本地审查后被第三方更新。最后执行：

```bash
git push --force-with-lease origin main
```

不使用裸 `--force`，不删除实验数据，不创建 PR。

## 验收标准

- `data/satellites/val_seen` 能解析到当前仓库的 `data/satellites/train`。
- `JointDataset` 定向测试通过，且同一轨迹只读取一次。
- 场景生成相关定向测试与 Python 语法检查通过。
- `README.md` 和 `TODO.md` 对当前主任务的描述一致。
- 实验报告中的现有数值不被改写或夸大。
- 临时备份不进入提交，实验数据和输出不被删除。
- 本地 `main` 的预期提交成功通过 `--force-with-lease` 推送到 `origin/main`。
