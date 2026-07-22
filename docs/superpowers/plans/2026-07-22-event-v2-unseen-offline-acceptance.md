# V2-0 未见轨迹离线验收实施计划

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**目标：** 在固定 64 条 `val_unseen` 事实轨迹上，以同批样本和严格 checkpoint 审计比较随机 V2 与 V2-0 10k，并生成可复核的离线验收结论。

**架构：** 新增一个与训练器解耦的只读评估工具。工具复用 `EventV2OfflineDataset`、`EventJointActorCritic` 和 `event_v2_offline_loss`，通过纯函数完成 support 加权和验收判定；Slurm 包装先做结果无关的 GPU batch 探测，再锁定档位运行 64 场正式评估。

**技术栈：** Python 3.11、PyTorch、pytest、Slurm、BF16 autocast、SDPA、JSON。

---

## Task 1：实现 support 加权与严格验收判定

**文件：**

- 新建：`tests/test_evaluate_event_v2_unseen_offline.py`
- 新建：`tools/evaluate_event_v2_unseen_offline.py`

1. 先写失败测试，覆盖：四分量按各自 support 加权、`total` 从聚合分量重算、相对降幅、任一分量未严格下降即拒绝、非有限值拒绝、少于 64 场拒绝、support 为零拒绝。
2. 运行：`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_evaluate_event_v2_unseen_offline.py -q`；预期因评估模块不存在而失败。
3. 最小实现 `aggregate_weighted_losses()` 与 `decide_acceptance()`，不得导入 Basilisk，也不得读取 Test。
4. 重跑同一测试，预期全部通过。
5. 提交：`git add tools/evaluate_event_v2_unseen_offline.py tests/test_evaluate_event_v2_unseen_offline.py && git commit -m "feat: add V2 offline acceptance metrics"`。

## Task 2：实现严格 checkpoint 审计与公平模型构建

**文件：**

- 修改：`tests/test_evaluate_event_v2_unseen_offline.py`
- 修改：`tools/evaluate_event_v2_unseen_offline.py`

1. 先写失败测试，覆盖 seed `3407` 可复现、随机基线与训练模型 backbone 完全相同、训练 checkpoint 的 `stage/steps/schema/config` 不匹配时拒绝、模型状态严格加载。
2. 运行目标测试，确认新断言先失败。
3. 实现 `audit_checkpoint()`、`build_paired_models()` 和 backbone 逐 tensor 相等检查；随机基线先加载同一 Stage3 checkpoint，训练模型再严格加载 V2-0 10k 状态。
4. 重跑目标测试，预期全部通过。
5. 提交：`git add tools/evaluate_event_v2_unseen_offline.py tests/test_evaluate_event_v2_unseen_offline.py && git commit -m "feat: audit paired V2 checkpoints"`。

## Task 3：实现同批 64 场评估与 JSON 审计输出

**文件：**

- 修改：`tests/test_evaluate_event_v2_unseen_offline.py`
- 修改：`tools/evaluate_event_v2_unseen_offline.py`

1. 先写失败测试，使用 tiny fake dataset/model 验证每场 batch 只构造一次、两模型事件时间点和 support 相同、正式模式必须恰好 64 场、已有输出默认拒绝覆盖、probe 模式输出 GPU 资源字段。
2. 运行目标测试并确认失败。
3. 实现 CLI：固定默认 `val_unseen.json`、checkpoint、Stage3 checkpoint 与输出路径；逐场用确定性 scene seed 构造一次 CPU batch；两模型同时常驻 GPU；使用 `torch.inference_mode()` 与 BF16 autocast；记录逐场 loss/support/event times、全局聚合、指纹和 `called_basilisk=false`、`read_test=false`。
4. 重跑测试，预期全部通过；随后运行 `git diff --check`。
5. 提交：`git add tools/evaluate_event_v2_unseen_offline.py tests/test_evaluate_event_v2_unseen_offline.py && git commit -m "feat: evaluate paired V2 models on unseen trajectories"`。

## Task 4：实现 GPU batch 自适应探测与 Slurm 包装

**文件：**

- 新建：`scripts/evaluate_event_v2_unseen_offline_slurm.sh`
- 新建：`tests/test_event_v2_unseen_scripts.py`

1. 先写静态失败测试，要求脚本使用 `local-10`、`aeos`、单独日志和结果目录，按 `{8,16,32,64,128,256,512}` 探测，限制 `max_memory_reserved/total_memory <= 0.90`，正式运行前锁定 batch，且不出现 Basilisk/Test 路径。
2. 运行：`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_unseen_scripts.py -q`；预期脚本缺失而失败。
3. 实现包装脚本：先检查当前 Slurm 分配内 GPU 可用性；每档用独立 Python 进程跑单场 probe 以便 OOM 后释放显存；选择最大安全档；正式运行一次 64 场；保存 probe JSON 与日志。禁止通过无用 tensor 人为占用显存。
4. 重跑脚本测试和评估工具测试，预期全部通过；运行 `bash -n scripts/evaluate_event_v2_unseen_offline_slurm.sh`。
5. 提交：`git add scripts/evaluate_event_v2_unseen_offline_slurm.sh tests/test_event_v2_unseen_scripts.py && git commit -m "feat: add Slurm V2 unseen acceptance runner"`。

## Task 5：回归验证、合并并运行正式验收

**文件：**

- 修改：`TODO.md`
- 生成但不提交：`work_dirs/event_joint_transformer_v2/v2_0_unseen_offline/summary.json`
- 生成但不提交：`work_dirs/eval_logs/event_v2_unseen_offline_<job>.log`

1. 在原分支运行：`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_evaluate_event_v2_unseen_offline.py tests/test_event_v2_unseen_scripts.py tests/test_event_v2_offline.py tests/test_event_v2_warm_start.py -q`。
2. 运行全量相关回归：`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2*.py -q`，并运行 `git diff --check`。
3. 按用户决定直接保留在 `codex/offline-critic-ranking`；只暂存本轮 V2 文件，不覆盖现有 M3 和用户改动。
4. 使用 `nvidia-smi`、`nvidia-smi pmon -s um -c 3` 和进程列表获取新鲜 GPU 状态，然后用 `sbatch scripts/evaluate_event_v2_unseen_offline_slurm.sh` 提交；等待至完成，不中途终止。
5. 校验 Slurm exit code、日志、summary JSON、64 scene、四类 support、有限值、checkpoint/schema/config 指纹、选定 batch 与峰值显存。`accepted=false` 是有效实验结论，不是运行失败。
6. 将真实 job、batch、显存、四分量及验收结论写入 `TODO.md`；由于 `改进日志.md` 有用户未提交改动，本轮不自动修改它。
7. 运行最终测试与 `git diff --check`，只提交本轮 `TODO.md` 更新：`git add TODO.md && git commit -m "docs: record V2 unseen offline acceptance"`。

## 成功标准

- 评估基础设施测试和 V2 相关回归全部通过；
- 正式作业恰好处理 64 条 `val_unseen`，无 Basilisk、无 Test；
- 同一场景的两模型输入、事件时间点和 support 完全一致；
- GPU batch 由无 OOM且峰值 reserved 不超过 90% 的最大有效档位确定；
- 无论 `accepted` 真或假，都保留完整、可复核的 JSON 和日志，不据结果更换 seed、scene 或 checkpoint。
