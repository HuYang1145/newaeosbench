# AGENTS.md

本文件为在本仓库内工作的编码助手提供协作规则。

## 环境

- 使用已有的 `aeos` conda 环境，其可执行文件位于 `/home/hy/miniconda3/envs/aeos/bin`。
- 优先从仓库根目录使用下面的前缀运行命令：

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" <command>
```

- 该环境提供 Python 3.11 和项目依赖，包括 PyTorch。除非用户明确要求，不要重建或重装环境。
- 运行 Python 脚本时，优先使用 `/home/hy/miniconda3/envs/aeos/bin/python`，或使用上面的 `PATH` 前缀。

## 项目说明

- 本仓库实现了带 Basilisk 仿真的 AEOS 星座调度方法，以及基于 Transformer 的模型。
- 关键训练和评估入口记录在 `CLAUDE.md`、`README.md` 和 `TODO.md` 中。
- 当前复现工作重点是尽量对齐论文 Table 2 和 Table 3 的结果。助手的主要任务是帮助复现论文数据、诊断本地指标与论文指标的差距，并选择更接近论文结果的训练和评估步骤。
- 当本地结果与论文不同，应优先查明原因，再启动新的无关实验。需要检查评估协议、数据划分、标注池、检查点来源、模型配置、损失定义、滚动生成/筛选规则和指标聚合公式。
- 默认以论文对齐作为成功标准：每个实验都应记录它对应论文哪一行、哪些指标一致、哪些指标有偏差，以及可能原因。
- 旧 200k CE-only 模型应保留为历史基线，但它不是严格论文复现模型。
- 当前可观测性过滤任务集工作中，要清楚区分三层数据：
  - `constellation` / `satellites` / `orbits` 是卫星物理场景；如果只是修正任务点有效性，通常应复用这些数据。
  - `tasksets` 是生成的地面观测任务；可观测性过滤修改的是这一层。
  - `trajectories.*` 是在特定 `taskset` 上生成的滚动生成轨迹、专家轨迹或控制轨迹。评估已有检查点不需要它们；如果要基于新 `tasksets` 重新训练，则必须重新生成它们。
- 可观测性过滤应使用快速物理几何检查，而不是完整 Basilisk 滚动生成。当前实现使用轨道传播、地球自转、地球遮挡/偏离星下点角约束、传感器类型匹配，以及任务 `release`/`due` 时间窗内的连续可见窗口。
- taskset 过滤不改变正式模型评估流程：评估仍然运行 `Policy + Controller + BasiliskEnvironment + TaskManager + Evaluators`。`TaskManager` 只是运行时任务状态账本，用于记录 release、ongoing、progress、succeeded、failed 和 closed 状态；它不是任务生成器。
- 可观测性过滤后的评估输出必须与未过滤输出分开。命名应包含 `observable_filtered`，例如 `work_dirs/rl_eval_*_observable_filtered` 和 `work_dirs/eval_summaries/*_observable_filtered.json`，避免新旧指标混在一起。
- 论文说明 train/val/test 的划分为：train 有 16,218 条轨迹，val-seen 有 64 个场景，val-unseen 有 64 个场景，test 有 64 个场景。本地指标对比前，应先按这些数量检查评估划分。
- 论文使用 96 个并行仿真环境进行评估。正式复现验证或评估时，如果资源允许，优先使用 `environment.world_size=96`。命令或日志中必须保留具体并行设置，方便追溯结果。
- 长时间正式评估应放在 `tmux` 等托管会话中运行，日志放在 `work_dirs/eval_logs/` 下，避免交互式会话关闭后任务中断。当前论文 Stage-3 全模型评估辅助脚本是 `scripts/run_stage3_96core_eval_managed.sh`。
- 预计运行超过几分钟的任务，尤其是训练、滚动生成、大规模评估或长时间数据处理，默认应放入 `tmux` 等后台托管会话，而不是在前台直接运行。
- 启动长任务前，优先在 `scripts/` 下创建专用包装脚本，并给会话使用清晰可识别的名称，方便恢复、检查和后续对比。
- 以托管方式启动长任务后，应在 `TODO.md` 中记录会话名称、命令或脚本、日志路径和预期输出路径。
- 不要依赖交互式编辑器会话一直打开。应假设用户可能随时关闭 VSCode 或断开连接，并据此选择托管/后台运行方式。

## 安全

- 工作区可能包含用户改动和实验输出。不要回滚或删除无关改动。
- 将 `data/`、`work_dirs/` 以及生成的轨迹/标注文件视为实验状态。替换正在使用的标注前必须先备份。
- 修改 tasksets 时不要删除旧生成数据。优先使用明确后缀重命名旧目录，例如 `*_unfiltered_YYYYMMDD_HHMMSS`，再把新数据生成到预期的活动路径。

## 沟通

- 默认用清楚、有条理的中文回复用户。
- 如果中文术语已经足够清楚，避免中英混杂。
- 给出命令或实现细节前，先用自然、分步骤的方式解释项目状态和技术结论。
- 如果必须使用英文技术术语，首次出现时应简要说明中文含义。
