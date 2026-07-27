# Event V2-Large Tail Top-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 V2-2-Large 尾部非终止小 batch，并从 update 1600 恢复两个 seed 后自动运行完整验证链。

**Architecture:** 在严格同步协议层加入同 policy 多轮聚合器，训练入口在部分 actor 完成时用存活 actor 补齐 `min_update_events`。checkpoint 只落在没有待训练 events 的安全 barrier；Slurm 恢复入口允许一次性覆盖恢复 checkpoint。

**Tech Stack:** Python 3.11、PyTorch、pytest、Bash、Slurm。

---

### Task 1: 用回归测试固定尾部行为

**Files:**
- Modify: `tests/test_event_v2_distributed_sync.py`
- Modify: `tests/test_event_v2_large_sync_scripts.py`

- [ ] 新增同 policy 多轮聚合测试：第一轮 58 events 且一个 actor 完成，第二轮由 7 个 actor 补采至少 6 events。
- [ ] 新增跨轮 policy、round、actor 集合和 transition 重复校验测试。
- [ ] 新增恢复脚本允许 `RESUME_A`/`RESUME_B` 覆盖的测试。
- [ ] 运行定向测试并确认因缺少聚合器和恢复覆盖而失败。

### Task 2: 实现安全尾部补采

**Files:**
- Modify: `constellation/new_transformers/event_v2/distributed_sync.py`
- Modify: `tools/train_event_v2_large_sync_ppo.py`

- [ ] 实现 `StrictSyncUpdateAccumulator`，只聚合同 policy 的连续同步轮次。
- [ ] 把单轮训练循环改为同 policy 的一个或多个采样轮，按剩余缺口计算补采目标。
- [ ] 达到 64 events 后更新一次；全部 actor 完成时允许跳过最终残余。
- [ ] checkpoint 前断言没有待训练 events；异常发生在待补采状态时不覆盖安全 checkpoint。
- [ ] 运行定向测试并确认通过。

### Task 3: 固定恢复点并保护后续续训

**Files:**
- Modify: `scripts/resume_event_v2_large_sync_ppo_full_slurm.sh`
- Modify: `TODO.md`

- [ ] 让恢复脚本默认使用 `checkpoint_latest.pth`，但支持 `RESUME_A`/`RESUME_B` 一次性覆盖。
- [ ] 记录根因、恢复点、验证命令和新 Slurm 依赖链。
- [ ] 运行 `bash -n` 和 Slurm 脚本测试。

### Task 4: 验证并提交作业链

**Files:**
- Test: `tests/test_event_v2_distributed_sync.py`
- Test: `tests/test_event_v2_large_sync_checkpoint.py`
- Test: `tests/test_train_event_v2_large_sync_ppo.py`
- Test: `tests/test_event_v2_large_sync_scripts.py`

- [ ] 使用 `aeos` 环境运行全部定向回归。
- [ ] 运行 synthetic preflight 或等价的真实 checkpoint 恢复检查。
- [ ] 取消 jobs `4243–4246` 的失效依赖链。
- [ ] 从两个 `checkpoint_update_001600.pth` 提交训练，并以 `afterok` 串联 heldout、Gate、完整 Val 和 Test。
- [ ] 用 `squeue`/`scontrol` 核对资源、依赖、恢复参数和日志路径。
- [ ] 只暂存和提交本计划涉及的文件。
