# Event V2-3 APPO Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从已通过 Val 8+8 的 V2-2 replica 0 checkpoint 启动真正的异步
Actor/Learner PPO，解冻 Stage3 Encoder/Decoder 最后一层，以最多 120 个 train
环境扩大在线训练，并在训练后重新通过 Val 8+8。

**Architecture:** 一个 learner 持有唯一可训练策略，1–4 个 actor 进程各持有只读
行为策略和一组 Basilisk runtime。actor 按固定 event chunk 采样并保存
`behavior_log_prob/policy_version`；learner 丢弃超过 policy-lag 上限的事件，再沿用
现有 event reward、time-aware GAE 和 clipped PPO 更新。策略权重通过共享 CPU
policy store 原子发布，不在 Actor/Critic forward 中调用额外 Basilisk，也不为候选
生成反事实轨迹。

**Tech Stack:** Python 3.11、PyTorch、`torch.multiprocessing`、Basilisk、
pytest、Slurm、BF16。

---

## 预注册参数

- bootstrap：
  `work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth`
- train scenes：固定 `205–324`，共 120 场；不访问 Val/Test。
- 解冻范围：Stage3 Encoder 最后 1 层和 Decoder 最后 1 层；输入 embedding、
  TimeModel 和其他层继续冻结。
- 新模块学习率：`1e-6`；解冻 Stage3 参数学习率：`1e-7`。
- `actor_chunk_events=32`，`learner_batch_events=128`。
- `max_policy_lag=2`；未来版本样本立即报错，超过 2 个版本的旧样本直接丢弃。
- `clip_ratio=0.2`、`max_kl=0.03`、`ppo_epochs=2`、
  `minibatch_events=32`，其余 reward/GAE 系数与 V2-2 相同。
- learner 每次成功 update 后原子发布一次共享策略；actor 只在 chunk 边界刷新，
  同一 chunk 内的 `policy_version` 必须一致。
- actor 在发送 chunk 前用自己的行为策略重放 log-prob，最大误差仍须
  `<=1e-6`；learner 不要求当前策略重放值等于旧 behavior log-prob，而是通过
  importance ratio 和 PPO clipping 使用它。
- 正式训练最多申请 4 张 GPU、120 CPU、200 GiB、28 小时。物理空闲 GPU 少于
  Slurm 可见 GPU 时自动排除占用超过 4 GiB 的卡；至少需要 2 张物理空闲 GPU。
- 1–4 个 actor 均匀分配 120 个 scene；learner 独占最后一张空闲 GPU，只有物理
  空闲卡不足 4 张时才与最后一个 actor 共享该卡。

### Task 1: APPO policy-lag 与 learner 核心

**Files:**
- Create: `constellation/new_transformers/event_v2/appo.py`
- Modify: `constellation/new_transformers/event_v2/ppo.py`
- Modify: `constellation/new_transformers/event_v2/__init__.py`
- Test: `tests/test_event_v2_appo.py`
- Test: `tests/test_event_v2_ppo.py`

- [ ] **Step 1: 写 policy-lag 的失败测试**

测试固定构造 policy version 为 `7/6/5/4` 的事件，并断言当前版本 7、
`max_policy_lag=2` 时保留前三个、丢弃版本 4；事件版本大于 learner 当前版本时抛出
`ValueError`；空输入和负 lag 配置也必须拒绝。

- [ ] **Step 2: 运行 RED**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_appo.py
```

预期：因 `event_v2.appo` 尚不存在而失败。

- [ ] **Step 3: 实现最小 policy-lag API**

在 `appo.py` 定义：

```python
@dataclass(frozen=True)
class APPOConfig:
    max_policy_lag: int = 2

class PolicyLagFilterResult(NamedTuple):
    accepted: tuple[StoredEventStep, ...]
    stale_dropped: int
    minimum_version: int
    maximum_version: int

def filter_policy_lag(
    steps: Sequence[StoredEventStep],
    *,
    current_policy_version: int,
    max_policy_lag: int,
) -> PolicyLagFilterResult:
    ...
```

实现只能基于 transition 内已保存的 `policy_version`，不得重新标记版本。

- [ ] **Step 4: 运行 GREEN**

运行 Task 1 Step 2 的命令，预期新增 policy-lag 测试通过。

- [ ] **Step 5: 写异步 learner 的失败测试**

扩展 `tests/test_event_v2_appo.py`：

- V2-2 policy 在 `unfreeze_last_layers(1, 1)` 后可构造 learner；
- optimizer 必须包含 `1e-6` 和 `1e-7` 两个参数组；
- learner 接受落后 1 个版本的 behavior transition；
- learner 不执行“当前策略必须精确等于 behavior 策略”的同步重放门槛；
- 更新后只有新模块和指定尾层可变化，其他 Stage3 参数逐值不变；
- stale drop 数和 policy version 出现在 metrics。

- [ ] **Step 6: 运行 RED**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_appo.py -k learner
```

预期：因 `AsynchronousPPOLearner` 尚不存在而失败。

- [ ] **Step 7: 泛化 PPO 更新器并实现 APPO learner**

给 `SynchronousPPOTrainer` 增加默认保持旧行为的两个显式参数：

```python
require_fully_frozen_backbone: bool = True
verify_behavior_replay: bool = True
```

冻结参数审计改为只比较 `requires_grad=False` 的 Stage3 named parameters；同步 PPO
默认路径的语义和测试不得变化。`AsynchronousPPOLearner` 先调用
`filter_policy_lag`，再用 `verify_behavior_replay=False` 的 PPO 更新器消费保留
事件；importance ratio 继续由
`exp(new_log_prob - behavior_log_prob)` 计算。

- [ ] **Step 8: 运行 APPO/PPO 回归**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_appo.py tests/test_event_v2_ppo.py \
    tests/test_event_v2_model.py tests/test_event_v2_backbone.py
```

预期：全部通过。

- [ ] **Step 9: 提交 learner 核心**

只暂存本 Task 列出的文件，提交信息：

```text
feat: add policy-lag bounded event v2 appo learner
```

### Task 2: APPO checkpoint 与共享策略发布

**Files:**
- Modify: `constellation/new_transformers/event_v2/checkpoint.py`
- Modify: `constellation/new_transformers/event_v2/appo.py`
- Test: `tests/test_event_v2_checkpoint.py`
- Test: `tests/test_event_v2_appo.py`

- [ ] **Step 1: 写 APPO checkpoint 的失败测试**

测试 checkpoint 必须保存并严格恢复：

```text
stage=V2-3
model/optimizer/scheduler/AMP
policy_version/updates/accepted_events/stale_dropped_events
processed_physical_seconds/completed_episodes
config/schema fingerprint
encoder_layers=1/decoder_layers=1/backbone_lr_scale=0.1
Python/NumPy/PyTorch/CUDA RNG
actor scene 分片和 actor runtime state
```

任一 stage、schema、config、scene shard 或解冻状态不一致都必须拒绝。同步 V2-1/V2-2
checkpoint 的现有加载测试必须继续通过。

- [ ] **Step 2: 运行 RED**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_checkpoint.py -k appo
```

预期：因 APPO checkpoint API 不存在而失败。

- [ ] **Step 3: 实现独立 APPO checkpoint schema**

在 `checkpoint.py` 增加 `APPO_CHECKPOINT_VERSION=1`、`APPOCounters`、
`build_appo_checkpoint()` 和 `load_appo_checkpoint()`。不修改已有同步 checkpoint
版本号，不让 V2-3 冒充同步 V2-1/V2-2。

- [ ] **Step 4: 写共享策略 store 的失败测试**

测试 learner 发布版本 3 后 actor 能精确刷新所有参数；相同版本不重复复制；版本只能
单调递增；写入期间由同一把进程锁保护参数和版本。

- [ ] **Step 5: 运行 RED**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_appo.py -k shared_policy
```

预期：因 `SharedPolicyStore` 不存在而失败。

- [ ] **Step 6: 实现共享 CPU policy store**

`SharedPolicyStore` 使用 `torch.nn.Module.share_memory()`、`multiprocessing.Value`
和 `multiprocessing.Lock`。`publish()` 在锁内从 learner state dict 复制到共享
CPU tensors 后再更新版本；`refresh()` 在锁内把同一版本完整复制到 actor model。

- [ ] **Step 7: 运行 checkpoint/store 回归**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_checkpoint.py tests/test_event_v2_appo.py
```

预期：全部通过。

- [ ] **Step 8: 提交 checkpoint/store**

提交信息：

```text
feat: checkpoint and publish event v2 appo policies
```

### Task 3: 异步 actor worker 与训练入口

**Files:**
- Modify: `constellation/new_transformers/event_v2/appo.py`
- Create: `tools/train_event_v2_appo.py`
- Create: `constellation/new_transformers/config_event_v2_appo.py`
- Test: `tests/test_event_v2_appo.py`
- Create: `tests/test_train_event_v2_appo.py`

- [ ] **Step 1: 写 actor chunk 的失败测试**

使用两个合成 runtime，断言 actor：

- chunk 内所有事件使用同一 behavior policy version；
- 发送前 log-prob 重放误差 `<=1e-6`；
- chunk 保存 actor id、scene ids、事件数、物理秒和完成 episode 数；
- 只在 chunk 边界刷新共享策略；
- stop event 能在边界安全退出；
- worker 异常通过 error message 传给 learner，不能静默丢失。

- [ ] **Step 2: 运行 RED**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_appo.py -k actor
```

预期：因 actor worker API 不存在而失败。

- [ ] **Step 3: 实现 actor 数据协议和 worker loop**

新增不可变消息 `APPORolloutChunk/APPOSnapshot/APPODone/APPOWorkerError`。worker
持有本地推理模型和固定 scene shard，沿用 `collect_synchronous_rollout()`；
每个 chunk 在行为模型上先调用 `replay_rollout_log_probs()`，再进入有界
`multiprocessing.Queue`。

- [ ] **Step 4: 写 learner orchestration 的失败测试**

用 `spawn` 启动两个合成 actor，断言 main loop 能：

- 累积到 `learner_batch_events=128` 才更新；
- 丢弃 lag 超限事件但保留新鲜事件；
- 每次成功 update 后发布新版本；
- 收到所有 actor done 后排空新鲜事件并正常结束；
- 收到任一 worker error 后设置 stop event、join 全部进程并失败；
- checkpoint 请求只在 actor chunk 边界返回 runtime states。

- [ ] **Step 5: 运行 RED**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_train_event_v2_appo.py
```

预期：因 APPO orchestration 尚未实现而失败。

- [ ] **Step 6: 实现训练入口**

`tools/train_event_v2_appo.py` 必须：

- 只接受 V2-2 selected checkpoint 作为 bootstrap；
- 先加载 policy，再调用 `unfreeze_last_layers(1, 1)`，创建全新 optimizer；
- 使用 `spawn`，不在 fork 后复用 CUDA context；
- 把 205–324 按 actor 数确定性分片；
- learner/actor 使用命令行指定的物理空闲 GPU；
- 持续记录 update、lag 分布、stale drop、queue wait、KL、clip fraction、
  gradient norm、完成场景和物理秒；
- 原子保存 V2-3 checkpoint、`metrics.jsonl` 和最终 `summary.json`；
- `accepted=true` 需要数值有限、非法动作 0、reward 重建误差 `<=1e-6`、
  actor replay 误差 `<=1e-6`、冻结参数变化 0、所有预注册 scene 完成且至少一次
  learner update。

- [ ] **Step 7: 运行合成 APPO**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  /home/hy/miniconda3/envs/aeos/bin/python tools/train_event_v2_appo.py \
    --config constellation/new_transformers/config_event_v2_appo.py \
    --synthetic-preflight --device cpu \
    --output /tmp/event_v2_appo_synthetic
```

预期：`accepted=true`，至少两个 policy version，至少产生并丢弃一个刻意构造的 stale
chunk，保存/恢复后的第一动作一致。

- [ ] **Step 8: 运行相关回归**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_appo.py tests/test_train_event_v2_appo.py \
    tests/test_event_v2_ppo.py tests/test_event_v2_checkpoint.py \
    tests/test_event_v2_rollout.py
```

预期：全部通过。

- [ ] **Step 9: 提交异步训练入口**

提交信息：

```text
feat: run event v2 actors and learner asynchronously
```

### Task 4: Slurm smoke 和正式训练链

**Files:**
- Create: `scripts/smoke_event_v2_appo_slurm.sh`
- Create: `scripts/train_event_v2_appo_full_slurm.sh`
- Modify: `tests/test_event_v2_sync_ppo_scripts.py`
- Modify: `TODO.md`

- [ ] **Step 1: 写 Slurm 包装的失败测试**

静态测试必须断言：

- 两个脚本都使用 `local-10/lab_team`、`aeos` Python 和绝对日志路径；
- 申请 4 GPU，但按 `nvidia-smi` 排除占用超过 4 GiB 的物理卡；
- 少于 2 张空闲 GPU 时明确失败；
- smoke 使用 train scene 205、完整 3,600 秒和独立输出目录；
- full 固定 scenes 205–324、最多 28 小时、最多 120 actor 环境；
- full 只允许通过的 smoke 以 `afterok` 依赖启动。

- [ ] **Step 2: 运行 RED**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_sync_ppo_scripts.py -k appo
```

预期：因 APPO Slurm 脚本不存在而失败。

- [ ] **Step 3: 实现两个 Slurm 包装**

smoke 使用一个 actor 和一个 learner；full 根据 2/3/4 张空闲卡启动 2/3/4 个 actor，
learner 使用最后一张卡。两个脚本都设置 BF16、SDPA、
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 和独立日志/输出目录。

- [ ] **Step 4: 运行完整 CPU 测试**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_event_v2_*.py tests/test_train_event_v2_appo.py
```

预期：V2 新旧测试全部通过。

- [ ] **Step 5: 提交并运行真实 smoke**

提交信息：

```text
ops: launch event v2 appo smoke
```

通过 `sbatch scripts/smoke_event_v2_appo_slurm.sh` 启动。只在完整 scene 205 的
`summary.json accepted=true` 后继续；若是代码/资源错误，保留日志、按系统化排错修复
后重试；若模型稳定性门槛失败，停止正式扩展。

- [ ] **Step 6: smoke 通过后自动提交正式训练**

通过：

```bash
sbatch --dependency=afterok:<smoke_job_id> \
  scripts/train_event_v2_appo_full_slurm.sh
```

将 job id、命令、日志、checkpoint 和输出路径写入 `TODO.md`，不等待用户再次确认。

### Task 5: APPO 后验收

**Files:**
- Create: `scripts/evaluate_event_v2_appo_val8_slurm.sh`
- Modify: `tools/evaluate_event_v2_policy.py`
- Modify: `constellation/new_transformers/event_v2/checkpoint.py`
- Modify: `tests/test_evaluate_event_v2_policy.py`
- Modify: `tests/test_event_v2_sync_ppo_scripts.py`
- Modify: `TODO.md`

- [ ] **Step 1: 写 V2-3 policy-only loader 的失败测试**

评估器必须能只读加载 `stage=V2-3`、解冻状态为 `1/1` 的 policy，而不恢复 optimizer、
actor runtime 或 RNG；同步 V2-1/V2-2 loader 保持严格不变。

- [ ] **Step 2: 运行 RED**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH="/home/hy/data/newaeosbench" \
  pytest -q tests/test_evaluate_event_v2_policy.py -k v2_3
```

预期：当前 evaluator 不认识 APPO checkpoint，测试失败。

- [ ] **Step 3: 实现 V2-3 只读评估**

配置按 checkpoint 解冻 1/1 层后加载 policy，随后全模型 `eval()` 和
`requires_grad_(False)`；确定性 Actor、Basilisk runtime 和 Q 聚合协议不变。

- [ ] **Step 4: 写并运行 APPO Val 8+8 包装**

APPO 训练成功后自动提交同一 Val scene `0–7` 的唯一一次 V2-2 vs V2-3 对照。
继续条件为两个 split：

```text
V2-3 Q - V2-2 Q >= 0.005
V2-3 CR/PCR/WCR 均不低于 V2-2
```

未通过则保留 V2-2 为第一阶段最佳 checkpoint，并停止扩大；通过才进入
Val Seen/Unseen 64+64。

- [ ] **Step 5: 记录结果并提交**

只写实际运行结果，不预填指标。提交信息：

```text
docs: record event v2 appo training and val gate
```
