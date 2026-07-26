# Event V2 Shared-Resource Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume both V2-2-Large seeds from their barrier checkpoints while limiting every formal training and downstream evaluation job to 2 GPU, 72 CPU, and 70 GiB.

**Architecture:** Preserve the checkpoint fingerprint by keeping 12 actors, 60 active environments, scene assignments, and PPO settings unchanged. Map each seed's learner and all actors onto one allocated GPU, then serialize evaluation work into batches of two so every downstream job remains valid with only two allocated GPUs.

**Tech Stack:** Bash/Slurm, Python 3.11 in the `aeos` Conda environment, pytest, PyTorch, Basilisk.

---

### Task 1: Lock the shared-resource contract in tests

**Files:**
- Modify: `tests/test_event_v2_large_sync_scripts.py`

- [ ] **Step 1: Write the failing resource and GPU-mapping tests**

Add assertions that the formal training/resume/evaluation scripts contain exactly:

```python
@pytest.mark.parametrize(
    'script_path',
    (FULL, RESUME),
)
def test_large_sync_formal_jobs_share_server_resources(
    script_path: pathlib.Path,
) -> None:
    script = script_path.read_text()
    assert '#SBATCH --gres=gpu:2' in script
    assert '#SBATCH --cpus-per-task=72' in script
    assert '#SBATCH --mem=70G' in script
    assert '#SBATCH --gres=gpu:4' not in script


def test_large_sync_resume_maps_one_seed_to_each_gpu() -> None:
    script = RESUME.read_text()
    assert 'GPU_A="${ALLOCATED_GPUS[0]}"' in script
    assert 'GPU_B="${ALLOCATED_GPUS[1]}"' in script
    assert script.count('--learner-device cuda:0') == 2
    assert script.count('--actor-devices cuda:0') == 2
    assert '--actors 12' in script
    assert '--active-environments 60' in script


def test_large_sync_smoke_keeps_small_cpu_request_and_uses_70g() -> None:
    script = SMOKE.read_text()
    assert '#SBATCH --gres=gpu:2' in script
    assert '#SBATCH --cpus-per-task=32' in script
    assert '#SBATCH --mem=70G' in script
```

Update the existing script-specific assertions from four GPUs and 120 CPUs to the approved limits. Add `import pytest`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH="${PWD}" \
  pytest -q tests/test_event_v2_large_sync_scripts.py
```

Expected: failures show the scripts still request `gpu:4`, `120/96 CPU`, `200/220/240G`, and use two-GPU pairs per seed.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_event_v2_large_sync_scripts.py
git commit -m "test: require shared resources for Event V2 jobs"
```

### Task 2: Change full and resume training to one GPU per seed

**Files:**
- Modify: `scripts/smoke_event_v2_large_sync_ppo_slurm.sh`
- Modify: `scripts/train_event_v2_large_sync_ppo_full_slurm.sh`
- Modify: `scripts/resume_event_v2_large_sync_ppo_full_slurm.sh`
- Test: `tests/test_event_v2_large_sync_scripts.py`

- [ ] **Step 1: Reduce Slurm resource requests**

Use these directives in both formal training scripts:

```bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=72
#SBATCH --mem=70G
```

Keep the smoke CPU count at 32, but reduce its memory directive to `#SBATCH --mem=70G`.

- [ ] **Step 2: Replace GPU pairs with one GPU per seed**

In both full and resume scripts, require two allocated GPUs and map them as:

```bash
if (( ${#ALLOCATED_GPUS[@]} < 2 )); then
  echo "[error] large sync training requires two allocated GPUs" >&2
  exit 1
fi
GPU_A="${ALLOCATED_GPUS[0]}"
GPU_B="${ALLOCATED_GPUS[1]}"
```

Launch each seed with its single visible GPU and local CUDA index zero:

```bash
CUDA_VISIBLE_DEVICES="${GPU_A}" \
  "${PYTHON}" tools/train_event_v2_large_sync_ppo.py \
    --learner-device cuda:0 \
    --actor-devices cuda:0 \
    --actors 12 \
    --active-environments 60
```

Repeat for seed `5409` with `CUDA_VISIBLE_DEVICES="${GPU_B}"`. Do not change PPO arguments, scenes, checkpoint interval, or resume paths.

- [ ] **Step 3: Run focused tests and verify GREEN**

Run:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH="${PWD}" \
  pytest -q tests/test_event_v2_large_sync_scripts.py
```

Expected: all current tests pass.

- [ ] **Step 4: Commit the training scripts**

```bash
git add scripts/smoke_event_v2_large_sync_ppo_slurm.sh \
  scripts/train_event_v2_large_sync_ppo_full_slurm.sh \
  scripts/resume_event_v2_large_sync_ppo_full_slurm.sh
git commit -m "ops: share GPUs in large sync training"
```

### Task 3: Run every downstream evaluation in two-GPU batches

**Files:**
- Modify: `scripts/select_event_v2_large_sync_heldout_slurm.sh`
- Modify: `scripts/eval_event_v2_large_sync_gate_slurm.sh`
- Modify: `scripts/eval_event_v2_large_sync_full_val_slurm.sh`
- Modify: `scripts/eval_event_v2_large_sync_test_once_slurm.sh`
- Test: `tests/test_event_v2_large_sync_scripts.py`

- [ ] **Step 1: Write failing evaluation resource and batching tests**

Add the resource contract for all downstream scripts:

```python
@pytest.mark.parametrize(
    'script_path',
    (HELDOUT, VAL_GATE, FULL_VAL, TEST_ONCE),
)
def test_large_sync_evaluations_share_server_resources(
    script_path: pathlib.Path,
) -> None:
    script = script_path.read_text()
    assert '#SBATCH --gres=gpu:2' in script
    assert '#SBATCH --cpus-per-task=72' in script
    assert '#SBATCH --mem=70G' in script
    assert '#SBATCH --gres=gpu:4' not in script
```

In the existing held-out, gate, full-Val, and Test tests, assert that GPU indices
use `% 2`, each batch barrier fires at two PIDs, and no `% 4` or direct GPU indices
2/3 remain.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH="${PWD}" \
  pytest -q tests/test_event_v2_large_sync_scripts.py
```

Expected: evaluation assertions fail because the four wrappers still request four GPUs and launch four-way batches.

- [ ] **Step 3: Apply the common Slurm limits**

Use exactly these directives in all four scripts:

```bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=72
#SBATCH --mem=70G
```

- [ ] **Step 4: Limit held-out and full-Val batches to two**

Replace `% 4` with `% 2` and replace the four-process barrier with:

```bash
if (( ${#pids[@]} == 2 )); then
  wait_batch
fi
```

- [ ] **Step 5: Add two-process barriers to the Val gate and Test**

For each loop, choose `gpu_index=$(( index % 2 ))` (or `shard_index % 2`), append each PID, and call this helper after every two launches and once after the loop:

```bash
wait_batch() {
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  pids=()
  if (( status != 0 )); then
    return 1
  fi
}
```

This preserves the four logical evaluations while running only two at a time.

- [ ] **Step 6: Run focused tests and shell syntax checks**

Run:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH="${PWD}" \
  pytest -q tests/test_event_v2_large_sync_scripts.py
```

Expected: all tests pass, including `bash -n` for every wrapper.

- [ ] **Step 7: Commit the tests and evaluation scripts**

```bash
git add tests/test_event_v2_large_sync_scripts.py \
  scripts/select_event_v2_large_sync_heldout_slurm.sh \
  scripts/eval_event_v2_large_sync_gate_slurm.sh \
  scripts/eval_event_v2_large_sync_full_val_slurm.sh \
  scripts/eval_event_v2_large_sync_test_once_slurm.sh
git commit -m "ops: evaluate Event V2 with two GPUs"
```

### Task 4: Verify checkpoint compatibility and update project tracking

**Files:**
- Modify: `TODO.md`
- Test: `tests/test_event_v2_large_sync_checkpoint.py`
- Test: `tests/test_event_v2_large_sync_scripts.py`
- Test: `tests/test_train_event_v2_large_sync_ppo.py`

- [ ] **Step 1: Run the relevant regression suite**

Run:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH="${PWD}" \
  pytest -q tests/test_event_v2_large_sync_checkpoint.py \
    tests/test_event_v2_large_sync_scripts.py \
    tests/test_train_event_v2_large_sync_ppo.py
```

Expected: all tests pass.

- [ ] **Step 2: Validate both live resume checkpoints without mutation**

Load both `checkpoint_latest.pth` files with `torch.load(..., mmap=True)` and assert:

```python
assert seed_5408['updates'] == 1420
assert seed_5409['updates'] == 1448
assert 'model' in checkpoint
assert 'optimizer' in checkpoint
assert 'actor_states' in checkpoint
```

- [ ] **Step 3: Update TODO with the paused and replacement job state**

Record that job `3296` stopped safely at barrier with exit `75`, old jobs `3306–3309` were cancelled, and the replacement job uses `2 GPU / 72 CPU / 70 GiB`. Preserve the original job history instead of overwriting it.

- [ ] **Step 4: Commit tracking changes**

```bash
git add TODO.md
git commit -m "docs: record shared-resource Event V2 resume"
```

### Task 5: Resume training and rebuild the gated dependency chain

**Files:**
- No source changes.

- [ ] **Step 1: Submit the resume job**

```bash
sbatch scripts/resume_event_v2_large_sync_ppo_full_slurm.sh
```

Record the returned job ID and verify with `scontrol show job` that it requests exactly `2 GPU`, `72 CPU`, `70G`, and unlimited time.

- [ ] **Step 2: Rebuild the `afterok` chain**

Submit held-out selection after the resume job, then Val gate, full Val, and Test using the generated job-specific JSON paths. Every downstream job must depend on `afterok` from the preceding job.

- [ ] **Step 3: Verify scheduler and runtime resource sharing**

Once running, check:

```bash
scontrol show job <resume-job-id>
sstat -j <resume-job-id>.batch --format=JobID,AveRSS,MaxRSS
nvidia-smi
```

Expected: only two GPUs allocated, CPU count 72, memory request 70G, two other GPUs remain schedulable, and each training GPU remains below its 24 GiB capacity.

- [ ] **Step 4: Update TODO with new job IDs and commit**

Record the resume and dependency-chain job IDs without marking unfinished training or validation as complete.

```bash
git add TODO.md
git commit -m "docs: record shared-resource Slurm chain"
```
