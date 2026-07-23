# Event V2-2 Full Synchronous PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从已通过 V2-1 正确性验收的 checkpoint 启动四个互相独立的 V2-2 同步 PPO replica，在 12–16 小时预算内覆盖 192 个预注册 train scenes，并保留 8 个固定 held-out train scenes 供 checkpoint 选择。

**Architecture:** V2-2 只继承 V2-1 的模型与 optimizer 状态，不继承旧场景 runtime、计数器或 RNG；每个 replica 使用独立 seed 和 48 个不重叠场景，四个 Slurm step 各占一张 GPU。V2-2 checkpoint 保持完整的阶段、配置、场景和 runtime 精确恢复语义；官方 Val/Test 在训练期间不可访问。

**Tech Stack:** Python 3.11、PyTorch、Basilisk、pytest、Slurm `local-10`、4×GPU、BF16 AMP；统一使用 `/home/hy/miniconda3/envs/aeos/bin/python`。

---

## 固定实验协议

- V2-1 源 checkpoint：`work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/checkpoint_update_000101.pth`。
- replica 0–3 分别使用 train scene `4–51`、`52–99`、`100–147`、`148–195`。
- 固定 held-out train scenes 为 `196–203`，不得参与梯度更新。
- 每个 replica 最多 `1,400` 次 update、每次 `64` 个事件、每 `200` 次 update 保存 checkpoint。
- Stage3 Encoder 继续冻结；动作、reward、PPO clipping、KL、BF16 和 5 秒安全复核不变。
- V2-2 训练不访问 `val_seen`、`val_unseen` 或 `test`。
- “全量训练”指按本阶段完整资源与时间预算运行上述 192 个场景，不声称穷举本地 100,000 个 train scenes。

### Task 1: 阶段化 checkpoint 与 V2-1 → V2-2 bootstrap

**Files:**
- Modify: `constellation/new_transformers/event_v2/checkpoint.py`
- Modify: `tests/test_event_v2_checkpoint.py`

- [ ] **Step 1: 写失败测试**

新增测试要求 `build_sync_ppo_checkpoint(..., stage='V2-2')` 写入 V2-2 阶段，`load_sync_ppo_checkpoint(..., expected_stage='V2-2')` 只接受同阶段；再新增 bootstrap 测试，要求从 V2-1 加载 model/optimizer，但不恢复源 RNG、scheduler、runtime 和计数器。

- [ ] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_checkpoint.py -q
```

Expected: FAIL，原因是 `stage` / `expected_stage` / bootstrap API 尚不存在。

- [ ] **Step 3: 最小实现**

给 `build_sync_ppo_checkpoint` 增加默认值为 `V2-1` 的 `stage` 参数；给精确恢复 loader 增加默认值为 `V2-1` 的 `expected_stage` 参数。新增 `load_sync_ppo_bootstrap_checkpoint`，只验证 checkpoint version、源 stage、transition schema 和冻结边界，然后加载 model/optimizer 并返回只读源元数据。

- [ ] **Step 4: 验证 GREEN**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_checkpoint.py -q
```

Expected: PASS。

### Task 2: V2-2 配置与训练入口

**Files:**
- Create: `constellation/new_transformers/config_event_v2_sync_ppo_full.py`
- Modify: `tools/train_event_v2_sync_ppo.py`
- Modify: `tests/test_event_v2_sync_ppo_scripts.py`

- [ ] **Step 1: 写失败测试**

新增测试固定以下行为：CLI 暴露 `--bootstrap-checkpoint` 与 `--seed`；V2-2 配置只含 train split、四个互斥的 48-scene 分片、固定 held-out `196–203`、`max_hours=16`、`max_updates=1400`、`checkpoint_interval=200`、Stage3 冻结和 BF16。

- [ ] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_sync_ppo_scripts.py -q
```

Expected: FAIL，原因是配置和 CLI 参数尚不存在。

- [ ] **Step 3: 最小实现**

训练入口允许 `V2-1`、`V2-2` 两个同步阶段。V2-2 创建新 runtime 时使用 bootstrap loader 继承 V2-1 model/optimizer；exact resume 仍要求目标阶段、目标 config fingerprint 和目标 scene IDs 完全相同。所有日志、checkpoint 和 summary 使用配置中的真实 stage；`--seed` 在计算 config fingerprint 前覆盖配置 seed。

- [ ] **Step 4: 验证 GREEN**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_checkpoint.py \
  tests/test_event_v2_sync_ppo_scripts.py -q
```

Expected: PASS。

### Task 3: 四 GPU Slurm 包装与静态防泄漏

**Files:**
- Create: `scripts/smoke_event_v2_2_sync_ppo_slurm.sh`
- Create: `scripts/train_event_v2_2_full_slurm.sh`
- Modify: `tests/test_event_v2_sync_ppo_scripts.py`

- [ ] **Step 1: 写失败测试**

新增脚本测试，要求正式脚本申请 `local-10`、4 张 GPU、16 小时，使用四个互斥 train 分片和四个不同 seed，调用 V2-1 checkpoint 的 bootstrap 参数，且脚本文本不含任何 Val/Test split。

- [ ] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_sync_ppo_scripts.py -q
```

Expected: FAIL，原因是两个包装脚本尚不存在。

- [ ] **Step 3: 最小实现**

smoke 脚本申请 1 张 GPU，在 train scene 4 上运行 60 秒/1 update。正式脚本申请 4 张 GPU、96 CPU、160 GiB 内存，用四个互斥 `srun --exclusive --gres=gpu:1` step 启动 replica；每个 replica 写独立日志与输出目录，父脚本等待全部 step 并在任一失败时非零退出。

- [ ] **Step 4: 验证 GREEN**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_sync_ppo_scripts.py -q
bash -n scripts/smoke_event_v2_2_sync_ppo_slurm.sh
bash -n scripts/train_event_v2_2_full_slurm.sh
```

Expected: 全部 PASS。

### Task 4: 启动前回归与真实 bootstrap smoke

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: 运行 V2 相关回归**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2*.py -q
git diff --check
```

Expected: 全部 PASS，且无 whitespace error。

- [ ] **Step 2: 提交并等待 smoke**

Run:

```bash
sbatch scripts/smoke_event_v2_2_sync_ppo_slurm.sh
```

验收 `summary.json`：stage 为 `V2-2`，有限数、log-prob 重放、冻结参数、动作合法性和 bootstrap 源元数据全部正确。失败时不提交正式训练。

- [ ] **Step 3: 在 TODO 记录证据**

记录 smoke job ID、日志、输出目录和结论；不得提前勾选收益门槛。

### Task 5: 提交并监控 V2-2 全预算训练

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: 提交正式任务**

Run:

```bash
sbatch scripts/train_event_v2_2_full_slurm.sh
```

- [ ] **Step 2: 确认四个 replica 正常推进**

检查 `squeue`、四个 replica 日志、GPU 显存/利用率和首个 update；确认 scene 分片、seed、stage、bootstrap checkpoint 与输出目录正确。

- [ ] **Step 3: 记录运行状态**

在 `TODO.md` 写入正式 job ID、资源、四个日志目录、checkpoint 周期和预期 wall time。训练完成前不访问官方 Val/Test。

- [ ] **Step 4: 训练后 checkpoint 选择**

仅在训练完成后，用固定 held-out train scenes `196–203` 比较候选 checkpoint 的 `Q=0.6CR+0.2PCR+0.2WCR` 与稳定性；选择一个候选，运行一个完整 3,600 秒 train smoke，然后只运行一次 Val Seen/Unseen 8+8。两个 split 的加权 Q 均至少提高 0.5 个百分点且任一 CR/PCR/WCR 不下降，才进入 V2-3。
