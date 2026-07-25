# Event V2 大规模严格同步 PPO 实施计划

> **For Codex:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** 从当前最佳 V2-2 checkpoint 出发，训练两个严格同步、互相独立的 PPO
模型；充分利用 4 张 GPU 和最多 120 个 CPU 核心，并通过预注册 heldout/Val 门槛
判断是否真正提高完成率。

**Architecture:** 每个 seed 是一套独立 learner，绑定 2 张 GPU；每个 learner
控制 12 个采样进程，每个采样进程同时维护 5 个 Basilisk 环境，并从自己的 10 个
scene 队列中自动补充完成的环境。因此每个 seed 同时运行 60 个环境、最终覆盖固定
的 120 个 train scenes `205–324`。采样进程只在收到指定 policy version 的命令后
收集 8 个事件，随后进入 barrier；learner 只聚合同一 version 的完整轮次，完成
一次 PPO update 后才发布下一 version。系统不接收 stale rollout，也不允许采样器
自行刷新权重。

**Tech Stack:** Python 3.11、PyTorch、Basilisk、multiprocessing `spawn`、BF16、
Slurm、pytest。

**固定资源和训练协议**

- 模型数：2 个独立 seed，不做“训练一段后只留下最好模型”的 population
  replacement。
- GPU：每个 seed 2 张，共 4 张；Actor 在两张卡上轮转，learner 固定在该 seed
  的第二张卡。
- CPU：Slurm 整项最多申请 120 核；24 个采样进程并发推进环境，其余核心供
  Basilisk、数据准备和进程通信使用。
- 环境：每个 seed 60 个活跃环境，两个 seed 共 120；每个 seed 都完整覆盖
  scenes `205–324`，而不是把场景拆成两个模型各看一半。
- 同步 batch：每个 seed 12 个 actor × 每轮最多 8 events，完整轮次最多
  96 events/update；聚合不足 64 events 时不做残缺 PPO update，只保存最终状态。
- 初始模型：
  `work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth`。
- 冻结边界：冻结旧 Stage3 Transformer Encoder、Decoder、TimeModel 和约束模块；
  只训练 edge projection、`EventStateEncoder`、联合 Actor 和 centralized Critic。
- PPO：`gamma=1.0`、time-aware GAE、`clip_ratio=0.2`、`max_kl=0.03`、
  4 epochs、minibatch 16、BF16。
- Checkpoint：每 100 updates 永久保存一次；另维护 `latest`，选择完成后创建
  `best`，周期文件永不被覆盖。
- 时限：不设置研究层面的人为上限。若集群限制单次 Slurm 时长，则从严格 barrier
  checkpoint 续跑。

---

### Task 1: 实现严格同步轮次协议

**Files:**

- Create: `constellation/new_transformers/event_v2/distributed_sync.py`
- Test: `tests/test_event_v2_distributed_sync.py`

**Step 1: 写失败测试**

测试必须覆盖：

1. 同一轮的所有 chunk 必须具有相同 `round_id` 和 `policy_version`；
2. actor id 重复、缺失、未来 version、旧 version 都立即报错；
3. 所有 chunk 聚合后保持 actor/事件的确定性顺序；
4. 聚合事件数小于 64 时返回“只保存、不更新”，不能偷偷缩小 PPO batch；
5. learner 更新前 actor 不能越过 barrier 开始下一轮；
6. actor 异常必须通过结构化错误消息传回父进程。

Run:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=":${PYTHONPATH:-}" \
  pytest -q tests/test_event_v2_distributed_sync.py
```

Expected: FAIL，因为同步协议模块尚不存在。

**Step 2: 实现最小协议数据结构与校验器**

实现下列公开接口：

```python
@dataclass(frozen=True)
class SyncRoundCommand:
    round_id: int
    policy_version: int
    stop: bool = False

@dataclass
class SyncActorChunk:
    actor_id: int
    round_id: int
    policy_version: int
    steps: list[StoredEventStep]
    completed_scene_ids: tuple[int, ...]
    replay_max_abs_error: float

@dataclass
class SyncActorDone:
    actor_id: int
    completed_scene_ids: tuple[int, ...]
    state: Mapping[str, Any]

@dataclass
class SyncWorkerError:
    actor_id: int
    error_type: str
    message: str
    traceback: str

def validate_and_merge_sync_round(
    chunks: Sequence[SyncActorChunk],
    *,
    expected_actor_ids: Collection[int],
    round_id: int,
    policy_version: int,
    min_batch_events: int,
) -> SyncRoundBatch:
    ...
```

`SyncRoundBatch` 必须包含确定性排序后的 `steps`、总事件数、是否允许 update、各
actor 的 replay 误差与新完成 scene。

**Step 3: 运行测试并提交**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=":${PYTHONPATH:-}" \
  pytest -q tests/test_event_v2_distributed_sync.py
git add constellation/new_transformers/event_v2/distributed_sync.py \
  tests/test_event_v2_distributed_sync.py
git commit -m "feat: add strict synchronous PPO round protocol"
```

---

### Task 2: 实现场景队列、环境回收和 actor barrier

**Files:**

- Modify: `constellation/new_transformers/event_v2/distributed_sync.py`
- Modify: `constellation/new_transformers/event_v2/rollout.py`
- Test: `tests/test_event_v2_distributed_sync.py`

**Step 1: 写失败测试**

使用合成 runtime factory 验证：

1. 一个 actor 分配 10 个 scene 时最多同时存在 5 个 runtime；
2. runtime 完成后立即从自己的待运行队列补充，但 scene 只运行一次；
3. 每轮最多收集 8 个 event；
4. actor 收到 version `v` 后只用 `v` 采样，并在收到 `v+1` 前保持等待；
5. actor 结束前返回每个活动 runtime、待运行 scene、已完成 scene 和 RNG；
6. behavior log-prob 重放误差超过容差时立即停止，不能把 chunk 交给 learner；
7. 父进程发 stop 后 actor 先确认终止，再退出，避免队列 tensor 被提前释放。

**Step 2: 增加可回收 rollout collector**

实现：

```python
class QueuedEventRuntimePool:
    def __init__(
        self,
        *,
        assigned_scene_ids: Sequence[int],
        max_active_environments: int,
        runtime_factory: Callable[[int], EventRuntime],
    ) -> None:
        ...

    def collect(
        self,
        *,
        model: EventJointActorCritic,
        policy_version: int,
        max_events: int,
        device: torch.device,
    ) -> SyncActorChunkPayload:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...
```

Actor loop 必须由父进程命令驱动，命令之间不采样；每次命令开始时从
`SharedPolicyStore` 加载精确 version，并在收集结束后向 learner queue 发送
`SyncActorChunk`。

**Step 3: 运行测试并提交**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=":${PYTHONPATH:-}" \
  pytest -q tests/test_event_v2_distributed_sync.py
git add constellation/new_transformers/event_v2/distributed_sync.py \
  constellation/new_transformers/event_v2/rollout.py \
  tests/test_event_v2_distributed_sync.py
git commit -m "feat: add queued synchronous event collectors"
```

---

### Task 3: 实现 barrier 边界的完整 checkpoint 与精确恢复

**Files:**

- Create: `constellation/new_transformers/event_v2/large_sync_checkpoint.py`
- Test: `tests/test_event_v2_large_sync_checkpoint.py`

**Step 1: 写失败测试**

测试 checkpoint 往返必须保存并恢复：

- model、optimizer、scheduler、BF16/AMP scaler；
- `round_id`、`policy_version`、update、episode、event 和 physical seconds；
- 12 个 actor 的 scene assignment、pending/active/completed scene；
- 每个 actor 的 Python、NumPy、Torch 和 CUDA RNG；
- schema fingerprint、冻结边界和参数名称；
- permanent checkpoint 路径不能覆盖已有同 update 文件；
- `checkpoint_latest.pth` 只能通过原子替换更新；
- schema、scene assignment 或冻结边界不一致时拒绝恢复。

**Step 2: 实现版本化 schema**

实现：

```python
LARGE_SYNC_CHECKPOINT_VERSION = 1

def build_large_sync_checkpoint_payload(...) -> dict[str, Any]:
    ...

def save_large_sync_checkpoint(
    path: Path,
    *,
    payload: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    ...

def update_latest_checkpoint(*, source: Path, latest: Path) -> None:
    ...

def load_large_sync_checkpoint(
    path: Path,
    *,
    expected_schema_fingerprint: str,
    expected_scene_assignments: Mapping[int, Sequence[int]],
    expected_trainable_parameter_names: Sequence[str],
) -> LargeSyncResumeState:
    ...
```

保存只允许发生在全体活跃 actor 已经到达 barrier、learner 尚未发布下一轮命令时。

**Step 3: 运行测试并提交**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=":${PYTHONPATH:-}" \
  pytest -q tests/test_event_v2_large_sync_checkpoint.py
git add constellation/new_transformers/event_v2/large_sync_checkpoint.py \
  tests/test_event_v2_large_sync_checkpoint.py
git commit -m "feat: checkpoint large synchronous PPO at barriers"
```

---

### Task 4: 增加双 GPU 单 seed 训练入口和冻结审计

**Files:**

- Create: `constellation/new_transformers/config_event_v2_large_sync_ppo.py`
- Create: `tools/train_event_v2_large_sync_ppo.py`
- Test: `tests/test_train_event_v2_large_sync_ppo.py`

**Step 1: 写失败测试**

测试配置和 CLI：

1. train scenes 恰好为 `205–324`；
2. 默认 `actor_count=12`、`max_active_environments=60`、
   `events_per_actor_round=8`、`min_update_events=64`；
3. 初始 checkpoint 是 V2-2 replica 0 update 1046；
4. 只加载兼容模型/optimizer 状态，不加载旧 runtime、计数器和 RNG；
5. 输出总参数、可训练参数、冻结参数和完整可训练参数名；
6. 冻结参数在 synthetic 两轮 update 后逐值不变；
7. 两轮 strict barrier 的 policy version 正好从 0 递增到 2；
8. 出现 NaN/Inf、invalid action、重放误差或 KL 超限时保存审计并停止；
9. 剩余聚合事件小于 64 时不做 partial update；
10. `--resume` 从 large-sync checkpoint 恢复，不能从普通 V2-2 checkpoint
    冒充断点续训。

**Step 2: 实现配置与训练编排**

CLI 至少支持：

```text
--config
--seed
--learner-device
--actor-devices
--actors
--active-environments
--scene-start
--scene-end
--output-dir
--resume
--synthetic-preflight
--max-updates
--checkpoint-every-updates
```

训练入口使用 `multiprocessing.get_context("spawn")`，创建一个
`SharedPolicyStore`、12 个 actor command queue 和一个结果 queue。每轮严格执行：

```text
发布 round(v) -> 等待所有活跃 actor 的 chunk/done
-> 校验并聚合 version v -> PPO update
-> checkpoint（若到周期）-> 发布 version v+1
```

所有 scene 完成后保存最终永久 checkpoint、`latest` 和冻结参数差异审计。

**Step 3: 运行测试并提交**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=":${PYTHONPATH:-}" \
  pytest -q tests/test_train_event_v2_large_sync_ppo.py \
    tests/test_event_v2_distributed_sync.py \
    tests/test_event_v2_large_sync_checkpoint.py
git add constellation/new_transformers/config_event_v2_large_sync_ppo.py \
  tools/train_event_v2_large_sync_ppo.py \
  tests/test_train_event_v2_large_sync_ppo.py
git commit -m "feat: train one large strict-sync Event V2 policy"
```

---

### Task 5: 增加 Slurm smoke、双 seed 正式训练和自动续跑

**Files:**

- Create: `scripts/smoke_event_v2_large_sync_ppo_slurm.sh`
- Create: `scripts/train_event_v2_large_sync_ppo_full_slurm.sh`
- Create: `scripts/resume_event_v2_large_sync_ppo_full_slurm.sh`
- Test: `tests/test_event_v2_large_sync_scripts.py`
- Modify: `TODO.md`

**Step 1: 写失败测试**

静态检查脚本必须保证：

- smoke 在真实 scene 上运行 3,600 物理秒；
- full 总计请求 4 GPU、CPU 不超过 120；
- seed A 只看到本作业的 GPU 0/1，seed B 只看到 GPU 2/3；
- 两个 seed 都覆盖 scenes `205–324`；
- 每个 seed 为 12 actors、60 活跃环境；
- `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 和 `OPENBLAS_NUM_THREADS` 总预算不会
  显式超过 120；
- 脚本不硬编码 48 小时研究停止条件；
- 若集群单次作业有时限，resume 脚本读取各自 `checkpoint_latest.pth`；
- 任一 seed 非零退出时整项作业失败，不能把另一 seed 的完成掩盖为整体成功；
- 日志、checkpoint、Slurm job id 和命令写回 `TODO.md`。

**Step 2: 实现脚本**

正式脚本在单个 4-GPU Slurm allocation 内并发启动两个训练进程：

```bash
CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" ... --seed 5408 ...
CUDA_VISIBLE_DEVICES="${GPU_C},${GPU_D}" ... --seed 5409 ...
wait_and_propagate_failures
```

每个进程内部逻辑 GPU `cuda:0/cuda:1`，不把物理 GPU id 写进 Python 配置。
脚本优先使用分区允许的最长时间；达到 Slurm 时限前由 signal handler 在下一
barrier 保存 checkpoint 并正常退出。续跑脚本只接受 schema 校验通过的
`checkpoint_latest.pth`。

**Step 3: 运行测试并提交**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=":${PYTHONPATH:-}" \
  pytest -q tests/test_event_v2_large_sync_scripts.py
bash -n scripts/smoke_event_v2_large_sync_ppo_slurm.sh
bash -n scripts/train_event_v2_large_sync_ppo_full_slurm.sh
bash -n scripts/resume_event_v2_large_sync_ppo_full_slurm.sh
git add scripts/smoke_event_v2_large_sync_ppo_slurm.sh \
  scripts/train_event_v2_large_sync_ppo_full_slurm.sh \
  scripts/resume_event_v2_large_sync_ppo_full_slurm.sh \
  tests/test_event_v2_large_sync_scripts.py TODO.md
git commit -m "feat: launch dual-seed large synchronous PPO"
```

---

### Task 6: 运行 preflight 和真实 smoke

**Files:**

- Modify: `TODO.md`
- Modify: `改进日志.md`

**Step 1: 运行全套相关测试**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=":${PYTHONPATH:-}" \
  pytest -q tests/test_event_v2_*.py \
    tests/test_train_event_v2_*.py
```

**Step 2: 运行 CPU 合成 strict-barrier preflight**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=":${PYTHONPATH:-}" \
  /home/hy/miniconda3/envs/aeos/bin/python \
  tools/train_event_v2_large_sync_ppo.py \
  --config constellation/new_transformers/config_event_v2_large_sync_ppo.py \
  --synthetic-preflight --actors 2 --active-environments 4 \
  --max-updates 2 --output-dir work_dirs/event_joint_transformer_v2/preflight_large_sync
```

**Step 3: 提交 Slurm 单场 3,600 秒 smoke**

```bash
sbatch scripts/smoke_event_v2_large_sync_ppo_slurm.sh
```

smoke 通过条件全部满足后才进入 Task 7：

- reward 重建误差在既有容差内；
- behavior log-prob 重放误差在既有容差内；
- invalid action 为 0；
- 冻结参数变化数为 0；
- checkpoint 可恢复并继续一轮；
- event time 单调推进且 3,600 物理秒完整结束。

失败时在 `改进日志.md` 记录根因并修复，不提交正式训练。

---

### Task 7: 提交双 seed 正式训练并自动监控

**Files:**

- Modify: `TODO.md`
- Modify: `改进日志.md`

**Step 1: 记录启动前资源快照**

```bash
nvidia-smi
sinfo -o "%P %a %l %D %G %C"
scontrol show partition
```

**Step 2: 提交正式作业**

```bash
sbatch scripts/train_event_v2_large_sync_ppo_full_slurm.sh
```

在 `TODO.md` 记录 job id、精确命令、两个 seed 的输出目录和日志。训练过程中每
100 updates 检查永久 checkpoint 能否加载，但不得依据在线训练曲线提前选择模型。

**Step 3: 节点/作业时限后的自动恢复**

若未完成 120 scenes 且停止原因只是 Slurm 时限或节点离线：

```bash
sbatch scripts/resume_event_v2_large_sync_ppo_full_slurm.sh
```

若出现稳定性停止条件，则不自动掩盖失败，保留最后一个健康 checkpoint 和审计。

---

### Task 8: 只用 train-heldout 选择 seed 和周期 checkpoint

**Files:**

- Create: `tools/select_event_v2_large_sync_heldout.py`
- Create: `scripts/select_event_v2_large_sync_heldout_slurm.sh`
- Test: `tests/test_select_event_v2_large_sync_heldout.py`
- Modify: `TODO.md`

**Step 1: 写失败测试**

选择器必须：

- 只接受 scene ids `196–203`；
- 枚举两个 seed 的所有永久周期 checkpoint 和 final checkpoint；
- 每个 checkpoint 使用同一评估协议；
- 按 `Q=0.6CR+0.2PCR+0.2WCR` 排序；
- 保存完整 CR/PCR/WCR/TAT/PC/CS，而不是只保存 Q；
- 不访问 Val/Test；
- 并列时依次优先更高的最小单项提升、更早 update、更小 seed；
- 原子写入 `selection.json` 并建立 `checkpoint_best.pth`。

**Step 2: 实现并运行 heldout 选择**

```bash
sbatch scripts/select_event_v2_large_sync_heldout_slurm.sh
```

只有 `selection.json` 锁定 checkpoint 后才允许进入 Val。

---

### Task 9: 执行一次新的 Val 8+8 gate

**Files:**

- Create: `scripts/eval_event_v2_large_sync_gate_slurm.sh`
- Modify: `TODO.md`
- Modify: `改进日志.md`

使用锁定 checkpoint 和当前最佳 V2-2，分别在：

- Val Seen scenes `8–15`
- Val Unseen scenes `8–15`

计算 `CR/PCR/WCR/Q/TAT_s/PC_Wh/CS_paper`。两个 split 必须同时满足：

```text
Q_new - Q_v2_2 >= 0.005
CR_new  >= CR_v2_2
PCR_new >= PCR_v2_2
WCR_new >= WCR_v2_2
```

任一条件失败就停止完整 Val/Test，保留 V2-2 为正式最佳模型，并把大规模同步 PPO
记为有审计证据的负结果。

---

### Task 10: gate 通过后运行完整 Val，最后只运行一次 Test

**Files:**

- Create: `scripts/eval_event_v2_large_sync_full_val_slurm.sh`
- Create: `scripts/eval_event_v2_large_sync_test_once_slurm.sh`
- Modify: `TODO.md`
- Modify: `改进日志.md`
- Modify: `docs/实验复现报告.md`

完整 Val 必须分开报告：

1. 历史诊断 scenes `0–7`；
2. 本轮 gate scenes `8–15`；
3. 其余 scenes `16–63`；
4. 全部 `0–63` 聚合。

只有完整 Val 仍满足预注册完成率门槛，才执行一次 Test。最终报告明确区分
“heldout 选择结果”“Val gate”“完整 Val”和“Test”，并保留所有负面权衡，不把
TAT/功耗下降或上升隐藏在单一 Q 中。

**Final verification:**

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=":${PYTHONPATH:-}" \
  pytest -q tests/test_event_v2_*.py \
    tests/test_train_event_v2_*.py \
    tests/test_select_event_v2_large_sync_heldout.py
git diff --check
git status --short --branch
```
