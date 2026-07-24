# Event V2-2 Held-out Checkpoint Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定 train scenes `196–203` 上确定性比较 V2-1 基线与四个 V2-2 最终 checkpoint，按 `Q=0.6CR+0.2PCR+0.2WCR` 自动选出一个稳定候选。

**Architecture:** 单 checkpoint evaluator 只加载模型权重，不恢复 optimizer、训练 RNG 或训练 runtime；每个场景创建一条真实 Basilisk 轨迹并使用 deterministic Actor。Slurm 包装并行运行五个 checkpoint，随后 selector 校验完全相同的场景集合、指标公式和阶段边界，只在四个 V2-2 候选中选最大 Q，同时报告相对 V2-1 的 held-out 变化。

**Tech Stack:** Python 3.11、PyTorch、Basilisk、pytest、Slurm `local-10`、BF16 AMP。

---

### Task 1: 只读 policy checkpoint loader

**Files:**
- Modify: `constellation/new_transformers/event_v2/checkpoint.py`
- Modify: `tests/test_event_v2_checkpoint.py`

- [ ] **Step 1: 写失败测试**

```python
metadata = load_sync_ppo_policy_checkpoint(
    path=path,
    model=target_model,
    expected_stages=('V2-1', 'V2-2'),
)
assert metadata.stage == 'V2-2'
assert metadata.updates == 914
assert target_model.state_dict()['actor.weight'].equal(source_weight)
```

测试同时要求 loader 拒绝错误 version、transition schema、未冻结 checkpoint 和非同步 PPO stage。

- [ ] **Step 2: 运行 RED**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_checkpoint.py -q
```

Expected: FAIL，原因是只读 loader 尚不存在。

- [ ] **Step 3: 最小实现并运行 GREEN**

新增只读 `SyncPPOPolicyMetadata` 和 `load_sync_ppo_policy_checkpoint`。函数只验证并加载 `model`，返回 stage、updates、policy_version、source scene IDs 和 config fingerprint；不得恢复 optimizer、scheduler、scaler、RNG 或 runtime。

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_checkpoint.py -q
```

Expected: PASS。

### Task 2: 确定性单 checkpoint 评估

**Files:**
- Create: `tools/evaluate_event_v2_policy.py`
- Create: `tests/test_evaluate_event_v2_policy.py`

- [ ] **Step 1: 写失败测试固定指标**

```python
snapshot = CompletionSnapshot(
    progress=torch.tensor([10., 5.]),
    required_duration=torch.tensor([10., 10.]),
    completed=torch.tensor([True, False]),
)
metrics = completion_metrics(snapshot)
assert metrics == pytest.approx({
    'CR': .5,
    'PCR': .75,
    'WCR': .5,
    'Q': .55,
})
```

再用 fake runtime 固定 evaluator 必须传 `deterministic=True`、每场只创建一个 runtime，并输出每场指标与 macro mean。

- [ ] **Step 2: 运行 RED**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_evaluate_event_v2_policy.py -q
```

Expected: collection FAIL，工具尚不存在。

- [ ] **Step 3: 最小实现并运行 GREEN**

CLI 参数固定为 `--config`、`--checkpoint`、`--label`、`--split`、`--scene-ids`、`--max-time-step`、`--device`、`--output`。只允许明确传入的 split；held-out 包装只传 `train`。每次 policy event 使用 BF16 autocast 和 deterministic Actor，动作 detach 到 CPU 后交给 runtime。

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_evaluate_event_v2_policy.py -q
```

Expected: PASS。

### Task 3: 严格 selector

**Files:**
- Create: `tools/select_event_v2_heldout.py`
- Create: `tests/test_select_event_v2_heldout.py`

- [ ] **Step 1: 写失败测试**

构造一个 V2-1 baseline 和四个 V2-2 summary。要求 selector：

- 所有 summary 的 split 都是 `train`；
- scene IDs 必须严格等于 `196–203`；
- `Q` 必须能由 `CR/PCR/WCR` 以 `1e-9` 重建；
- 只在 V2-2 中选最大 Q，Q 相等时按 label 排序；
- 输出相对 baseline 的 CR/PCR/WCR/Q 差值。

- [ ] **Step 2: 运行 RED**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_select_event_v2_heldout.py -q
```

Expected: collection FAIL。

- [ ] **Step 3: 最小实现并运行 GREEN**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_select_event_v2_heldout.py -q
```

Expected: PASS。

### Task 4: Slurm held-out 比较

**Files:**
- Create: `scripts/evaluate_event_v2_heldout_slurm.sh`
- Modify: `tests/test_event_v2_sync_ppo_scripts.py`
- Modify: `TODO.md`

- [ ] **Step 1: 写失败测试**

脚本必须只包含 train scenes `196–203`，并行运行：

- `v2_1`：`checkpoint_update_000101.pth`
- `v2_2_replica_0`：`checkpoint_update_001046.pth`
- `v2_2_replica_1`：`checkpoint_update_000950.pth`
- `v2_2_replica_2`：`checkpoint_update_000924.pth`
- `v2_2_replica_3`：`checkpoint_update_000914.pth`

脚本不得出现 `val_seen`、`val_unseen` 或 test split，五个进程结束后必须调用 selector。

- [ ] **Step 2: 运行 RED**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_sync_ppo_scripts.py -q
```

Expected: FAIL，脚本尚不存在。

- [ ] **Step 3: 实现、回归并提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_checkpoint.py \
  tests/test_evaluate_event_v2_policy.py \
  tests/test_select_event_v2_heldout.py \
  tests/test_event_v2_sync_ppo_scripts.py -q
bash -n scripts/evaluate_event_v2_heldout_slurm.sh
git diff --check
```

Expected: 全部 PASS。

- [ ] **Step 4: 真实单场 smoke 后提交正式 held-out**

先在 scene `196` 上对 V2-1 与一个 V2-2 checkpoint 各运行一次完整 3,600 秒确定性评估；通过后提交五模型 × 八场景 Slurm 作业。作业完成后自动生成 `selection.json`，然后按 TODO 继续候选 3,600 秒 smoke。

### Task 5: 候选 smoke 与唯一一次 Val 8+8

**Files:**
- Create: `scripts/smoke_event_v2_selected_slurm.sh`
- Create: `scripts/evaluate_event_v2_val8_gate_slurm.sh`
- Create: `tools/compare_event_v2_val_gate.py`
- Create: `tests/test_compare_event_v2_val_gate.py`
- Modify: `tests/test_event_v2_sync_ppo_scripts.py`
- Modify: `TODO.md`

- [ ] **Step 1: 写失败测试固定门槛**

`compare_val_gate` 必须分别比较 Val Seen 和 Val Unseen；两个 split 的 Q 都至少比
V2-1 提高 `0.005`，并且各自的 CR/PCR/WCR 都不得下降，才返回 `passed=true`。

- [ ] **Step 2: 实现 selector 驱动的候选 smoke**

脚本只从 `heldout_2212/selection.json` 读取 selected checkpoint，并在新的 train
scene 204 上确定性运行一次 3,600 秒；不重新选择 checkpoint。

- [ ] **Step 3: 实现唯一一次 Val 8+8**

同一 Slurm 作业内对 V2-1 与 selected V2-2 分别运行 `val_seen 0–7` 和
`val_unseen 0–7`，随后调用 `compare_event_v2_val_gate.py`。gate 失败时保存
`gate.json` 并以 exit 2 停止，不进入 APPO；gate 通过才允许 V2-3。

- [ ] **Step 4: 回归与依赖提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_compare_event_v2_val_gate.py \
  tests/test_event_v2_sync_ppo_scripts.py -q
bash -n scripts/smoke_event_v2_selected_slurm.sh
bash -n scripts/evaluate_event_v2_val8_gate_slurm.sh
git diff --check
```

正式 smoke 使用 `afterok:<heldout_job>`，Val 使用 `afterok:<selected_smoke_job>`，
确保上一阶段失败时不会继续消耗官方 Val。
