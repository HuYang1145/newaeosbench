# Fixed Action Hold Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 Stage3-200k 模型参数的前提下，实现非空任务固定保持 `1/5/30` 秒的因果消融，完成同场景 Basilisk 评估，并把设计、协议、结果和结论写入 `改进日志.md`。

**Architecture:** 保持模型和 Basilisk 每秒运行，在 `GreedyModelAlgorithm` 将相对动作映射为全局 task id 后增加逐卫星保持状态。空动作不锁定；非空任务在仍属于 ongoing taskset 时执行满配置时长。使用独立汇总工具读取指标 JSON 和轨迹 PTH，比较正式指标与动作持续性指标。

**Tech Stack:** Python 3.11、PyTorch、Basilisk、pytest、现有 `aeos` Conda 环境。

---

## 文件边界

- Modify: `tools/rollout_model_trajectories.py`：CLI、固定保持状态、rollout 参数传递和元数据。
- Modify: `tests/test_rollout_model_candidates.py`：保持状态和算法集成回归测试。
- Create: `tools/summarize_action_hold.py`：汇总正式指标和轨迹行为指标。
- Create: `tests/test_summarize_action_hold.py`：汇总公式与片段统计测试。
- Create: `scripts/run_action_hold_stage1.sh`：固定 checkpoint、split、场景数和输出目录。
- Modify: `改进日志.md`：先记录设计和启动协议，实验结束后补真实结果与结论。

当前前三个 Modify 文件已有未提交改动。实施前复制时间戳备份到 `/tmp`，所有补丁基于
当前工作树增量应用，不覆盖、不格式化、不暂存既有改动。

### Task 1: 建立备份并写入设计记录

**Files:**
- Backup: `/tmp/action_hold_stage1_<timestamp>/`
- Modify: `改进日志.md`

- [ ] **Step 1: 建立三个重叠文件的时间戳备份**

Run:

```bash
backup_dir=/tmp/action_hold_stage1_$(date +%Y%m%d_%H%M%S)
mkdir -p "$backup_dir/tools" "$backup_dir/tests"
cp 改进日志.md "$backup_dir/改进日志.md"
cp tools/rollout_model_trajectories.py "$backup_dir/tools/rollout_model_trajectories.py"
cp tests/test_rollout_model_candidates.py "$backup_dir/tests/test_rollout_model_candidates.py"
```

Expected: 三个备份文件存在，原文件未变化。

- [ ] **Step 2: 在改进日志追加阶段一设计和启动状态**

追加独立小节，内容必须包括：

```markdown
#### P3.2：多时间尺度动作持续预测与固定保持消融

当前目标不是改变 Basilisk 的 1 秒物理步长，而是验证非空任务的最小持续承诺能否
减少几乎无贡献的一秒短脉冲。阶段一冻结 Stage3-200k，不训练新参数，比较 H=1、
H=5、H=30；空动作不锁定，任务完成、截止或退出 ongoing taskset 时立即解锁。

H=5 是主实验，H=30 只作为过长保持风险对照。成功标准以
CR/PCR/WCR/TAT_s/PC_Wh/CS_paper 为主，动作 top-1 accuracy 仅作诊断。
```

Expected: 不改写 P3.0/P3.1 已有内容，只在末尾新增 P3.2。

- [ ] **Step 3: 检查日志增量**

Run:

```bash
git diff --check -- 改进日志.md
git diff -- 改进日志.md
```

Expected: 无空白错误，diff 只包含当前新增 P3.2 和实施前已有改动。

### Task 2: 用 TDD 实现固定保持状态

**Files:**
- Modify: `tests/test_rollout_model_candidates.py`
- Modify: `tools/rollout_model_trajectories.py`

- [ ] **Step 1: 写保持状态失败测试**

在测试文件中先加入以下行为测试：

```python
def test_fixed_action_hold_repeats_non_idle_for_exact_horizon() -> None:
    hold = FixedActionHold(hold_seconds=3)
    assert hold.apply([31], ongoing_task_ids=[31, 44]) == [31]
    assert hold.apply([44], ongoing_task_ids=[31, 44]) == [31]
    assert hold.apply([44], ongoing_task_ids=[31, 44]) == [31]
    assert hold.apply([44], ongoing_task_ids=[31, 44]) == [44]


def test_fixed_action_hold_never_locks_idle() -> None:
    hold = FixedActionHold(hold_seconds=5)
    assert hold.apply([-1], ongoing_task_ids=[31]) == [-1]
    assert hold.apply([31], ongoing_task_ids=[31]) == [31]


def test_fixed_action_hold_releases_task_that_is_no_longer_ongoing() -> None:
    hold = FixedActionHold(hold_seconds=30)
    assert hold.apply([31], ongoing_task_ids=[31, 44]) == [31]
    assert hold.apply([44], ongoing_task_ids=[44]) == [44]


def test_one_second_hold_matches_every_step_proposal() -> None:
    hold = FixedActionHold(hold_seconds=1)
    assert hold.apply([31], ongoing_task_ids=[31, 44]) == [31]
    assert hold.apply([44], ongoing_task_ids=[31, 44]) == [44]
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_rollout_model_candidates.py -q
```

Expected: FAIL，原因是 `FixedActionHold` 尚不存在。

- [ ] **Step 3: 实现最小保持状态**

在 rollout 工具中加入：

```python
class FixedActionHold:

    def __init__(self, hold_seconds: int) -> None:
        if hold_seconds <= 0:
            raise ValueError('hold_seconds must be positive')
        self._hold_seconds = hold_seconds
        self._held_task_ids: list[int] = []
        self._remaining_steps: list[int] = []

    def apply(
        self,
        proposed_task_ids: list[int],
        *,
        ongoing_task_ids: list[int],
    ) -> list[int]:
        if len(self._held_task_ids) != len(proposed_task_ids):
            self._held_task_ids = [-1] * len(proposed_task_ids)
            self._remaining_steps = [0] * len(proposed_task_ids)
        ongoing = set(ongoing_task_ids)
        output: list[int] = []
        for index, proposed in enumerate(proposed_task_ids):
            held = self._held_task_ids[index]
            if self._remaining_steps[index] > 0 and held in ongoing:
                output.append(held)
                self._remaining_steps[index] -= 1
                continue
            output.append(proposed)
            self._held_task_ids[index] = proposed
            self._remaining_steps[index] = (
                self._hold_seconds - 1 if proposed >= 0 else 0
            )
        return output
```

- [ ] **Step 4: 运行定向测试确认通过**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_rollout_model_candidates.py -q
```

Expected: 新增和既有测试全部 PASS。

### Task 3: 集成 rollout CLI 和全局 task id 映射

**Files:**
- Modify: `tests/test_rollout_model_candidates.py`
- Modify: `tools/rollout_model_trajectories.py`

- [ ] **Step 1: 写算法集成失败测试**

构造两步 logits，让第二步模型建议 task 44，但 `hold_seconds=5` 时仍执行全局 task 31；
再把 task 31 从 ongoing taskset 删除，断言立即切换到 44。测试必须检查返回的
assignment 和 `Action.target_location` 同时一致。

- [ ] **Step 2: 运行集成测试并确认失败**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_rollout_model_candidates.py -q
```

Expected: FAIL，因为 `GreedyModelAlgorithm` 尚未应用保持状态。

- [ ] **Step 3: 完成参数和算法集成**

修改项：

```python
parser.add_argument('--min-action-hold-seconds', type=int, default=1)
```

`GreedyModelAlgorithm.__init__` 创建：

```python
self._action_hold = FixedActionHold(min_action_hold_seconds)
```

`step()` 中先按现有 logits 得到全局 assignment，再执行：

```python
assignment = self._action_hold.apply(
    assignment,
    ongoing_task_ids=taskset.ids.tolist(),
)
task_index_by_id = {
    int(task_id): index for index, task_id in enumerate(taskset.ids.tolist())
}
relative_task_ids = torch.tensor([
    task_index_by_id.get(task_id, -1) for task_id in assignment
])
```

把参数贯穿 `rollout_one()`、`main()` 和 `rollout_metadata.json`。CLI 输入小于 1 时抛出
`ValueError`。

- [ ] **Step 4: 运行定向测试和语法检查**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_rollout_model_candidates.py -q
/home/hy/miniconda3/envs/aeos/bin/python -m py_compile \
  tools/rollout_model_trajectories.py \
  tests/test_rollout_model_candidates.py
```

Expected: 全部 PASS，语法检查无输出。

### Task 4: 实现可复现汇总工具

**Files:**
- Create: `tests/test_summarize_action_hold.py`
- Create: `tools/summarize_action_hold.py`

- [ ] **Step 1: 写轨迹统计失败测试**

构造包含 `[-1, 31, -1]`、`[44, 44, 44]` 和重复 assignment 的小型 tensor，断言：

```python
assert summary['one_second_non_idle_runs'] == 1
assert summary['non_idle_runs'] == 2
assert summary['one_second_work_fraction'] == pytest.approx(0.25)
assert summary['duplicate_edge_rate'] == pytest.approx(expected_rate)
```

另写指标汇总测试，复用 `tools.summarize_eval.compute_scores()`，确保
`CS_paper = quality**-1 + TAT_s/700 + PC_Wh/100`。

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_summarize_action_hold.py -q
```

Expected: FAIL，因为汇总模块尚不存在。

- [ ] **Step 3: 实现轨迹和指标汇总**

工具提供：

```python
def summarize_actions(actions: torch.Tensor) -> dict[str, float | int]: ...
def summarize_trajectory(path: Path) -> dict[str, float | int]: ...
def summarize_variant(root: Path) -> dict[str, object]: ...
```

`summarize_actions` 按卫星切分连续非空 `(task_id)` 片段；重复边定义为同一时刻分配给
出现次数大于 1 的 task id 的非空卫星边数，占全部非空边的比例。正式指标调用现有
`summarize_split()`，避免复制 `CS_paper` 公式。

- [ ] **Step 4: 运行汇总测试**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_summarize_action_hold.py tests/test_summarize_eval.py -q
```

Expected: 全部 PASS。

### Task 5: 真实 checkpoint smoke 与 2+2 实验

**Files:**
- Create: `scripts/run_action_hold_stage1.sh`
- Create under ignored experimental state: `work_dirs/action_hold_stage1/`

- [ ] **Step 1: 写专用运行脚本**

脚本固定：

```bash
CHECKPOINT=work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth
SCENE_LIMIT=2
SPLITS=(val_seen val_unseen)
HOLDS=(1 5 30)
```

每个组合使用独立输出：

```text
work_dirs/action_hold_stage1/h1/val_seen
work_dirs/action_hold_stage1/h5/val_seen
work_dirs/action_hold_stage1/h30/val_seen
```

命令使用 `aeos` Python、`--strategy greedy`、相同 seed 和
`--min-action-hold-seconds`。已有完整 JSON/PTH 时安全跳过。

- [ ] **Step 2: 检查 GPU 和进程**

Run:

```bash
nvidia-smi
pgrep -af 'rollout_model_trajectories|eval_all'
```

Expected: 确认 GPU 有足够空间且没有冲突实验；否则改用空闲 GPU 或 CPU，不终止用户
进程。

- [ ] **Step 3: 运行单场 smoke**

Run:

```bash
SCENE_LIMIT=1 HOLDS='1 5' bash scripts/run_action_hold_stage1.sh
```

Expected: 四个场景变体均生成 JSON 和 PTH，`H=1` 不报兼容错误。

- [ ] **Step 4: 运行完整 2+2**

Run:

```bash
SCENE_LIMIT=2 HOLDS='1 5 30' bash scripts/run_action_hold_stage1.sh
```

Expected: 12 个场景变体完成，日志、checkpoint、seed、split、hold seconds 可追溯。

- [ ] **Step 5: 汇总结果**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python tools/summarize_action_hold.py \
  --output work_dirs/action_hold_stage1/summary.json \
  h1=work_dirs/action_hold_stage1/h1 \
  h5=work_dirs/action_hold_stage1/h5 \
  h30=work_dirs/action_hold_stage1/h30
```

Expected: summary 同时包含两个 split 的正式指标和动作片段统计。

### Task 6: 写入真实结果并完成验证

**Files:**
- Modify: `改进日志.md`

- [ ] **Step 1: 根据 summary 写入结果表**

必须写入真实数值，不使用估计值。分别列 Val Seen、Val Unseen 的
`CR/PCR/WCR/TAT_s/PC_Wh/CS_paper`，并补充一秒片段与重复率变化。

- [ ] **Step 2: 按停止门槛给出结论**

只允许以下结论之一：

```text
H=5 通过 2+2，进入 8+8。
H=5 未通过，固定保持路线停止。
2+2 样本方向不一致，需要先扩至预先规定的 8+8 才能判断。
```

不得因为切换率降低就宣称预测准确度提高。

- [ ] **Step 3: 运行最终定向验证**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_rollout_model_candidates.py \
  tests/test_summarize_action_hold.py \
  tests/test_summarize_eval.py -q
/home/hy/miniconda3/envs/aeos/bin/python -m py_compile \
  tools/rollout_model_trajectories.py \
  tools/summarize_action_hold.py
git diff --check
```

Expected: 全部测试通过，语法检查和 diff 检查无错误。

- [ ] **Step 4: 报告工作状态和回滚边界**

报告当前分支、修改前基线 `30d3259`、备份目录、验证命令、实验输出和结论。由于三个
目标文件在本任务前已有未提交改动，除非用户另行批准 checkpoint 提交，不将其既有
内容混入本任务提交。
