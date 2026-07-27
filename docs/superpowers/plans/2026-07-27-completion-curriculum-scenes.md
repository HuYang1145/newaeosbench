# Completion Curriculum Scenes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate isolated 600-second and 1800-second Event V2 curriculum scene splits from the accepted train satellite pool, verify all 256 static scenes, and record the candidate completion objective and generated assets without changing the running paper-Q protocol.

**Architecture:** Add one standalone generator under `tools/` so the formal `Task.sample()` and existing 3600-second generator remain unchanged. The generator loads the sorted accepted train satellite pool, applies fixed seeds, writes constellation/taskset pairs through temporary sibling directories, audits every file before publication, and writes a separate metadata record under `work_dirs/`. Event V2 continues to load the new data through the existing arbitrary split name accepted by `BasiliskSceneBackend.from_scene_id()`.

**Tech Stack:** Python 3.11 in the `aeos` Conda environment, `argparse`, `dataclasses`, `pathlib`, PyTorch seed control, existing `Constellation`/`Task`/`TaskSet` classes, pytest, Git.

---

### Task 1: Horizon-aware task sampling and validation

**Files:**
- Create: `tools/generate_curriculum_scenes.py`
- Create: `tests/test_generate_curriculum_scenes.py`

- [ ] **Step 1: Write failing tests for specification and task bounds**

Add tests that import `CurriculumSceneSpec`, `sample_curriculum_task`, and `validate_spec` from the new tool. Cover a valid 600-second spec and these failures: horizon below `3*60`, invalid satellite/task ranges, non-positive scene count, and a non-curriculum split name. Generate 2,000 tasks with a seeded `random.Random` and assert:

```python
assert 15 <= task.duration <= 60
assert 0 <= task.release_time < task.due_time <= 600
assert task.due_time - task.release_time >= 3 * task.duration
assert -90 <= task.coordinate.x <= 90
assert -180 <= task.coordinate.y <= 180
assert task.sensor_type is SensorType.VISIBLE
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=".:${PYTHONPATH:-}" \
  /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_generate_curriculum_scenes.py
```

Expected: collection fails because `tools.generate_curriculum_scenes` does not exist.

- [ ] **Step 3: Implement the immutable specification and sampler**

Define:

```python
@dataclasses.dataclass(frozen=True)
class CurriculumSceneSpec:
    split: str
    horizon: int
    num_scenes: int
    satellite_min: int
    satellite_max: int
    task_min: int
    task_max: int
    seed: int

def validate_spec(spec: CurriculumSceneSpec) -> None: ...

def sample_curriculum_task(
    task_id: int,
    *,
    horizon: int,
    rng: random.Random,
) -> Task: ...
```

Require `split.startswith('curriculum_')`, `horizon >= 180`, positive counts, ordered inclusive ranges, and `satellite_max <= len(pool)` when the full generator runs. Sample duration first, then release and due using the approved `3*duration` window.

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run the Task 1 pytest command. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the sampling unit**

```bash
git add tools/generate_curriculum_scenes.py tests/test_generate_curriculum_scenes.py
git commit -m "feat: add curriculum scene specification"
```

### Task 2: Safe deterministic split generation

**Files:**
- Modify: `tools/generate_curriculum_scenes.py`
- Modify: `tests/test_generate_curriculum_scenes.py`

- [ ] **Step 1: Add failing generation tests**

Use `tmp_path` and a small list of real `Satellite` objects loaded from the sorted train pool. Test `generate_curriculum_split()` with two scenes and assert the exact layout:

```text
constellations/curriculum_test/00/00000.json
constellations/curriculum_test/00/00001.json
tasksets/curriculum_test/00/00000.json
tasksets/curriculum_test/00/00001.json
metadata/curriculum_test/metadata.json
```

Load every JSON with `Constellation.load()` and `TaskSet.load()`. Run the generator in two independent temporary roots with the same seed and assert that the four scene JSON payloads are identical. Add a test proving that an existing non-empty target split raises `FileExistsError` without changing its sentinel file.

- [ ] **Step 2: Run the focused tests and confirm generation is not implemented**

Run the same pytest command. Expected: failures for missing `generate_curriculum_split` and audit helpers.

- [ ] **Step 3: Implement pool loading, temporary generation, audit, and publication**

Define focused helpers:

```python
def load_satellite_pool(root: pathlib.Path) -> Satellites: ...
def audit_generated_split(
    spec: CurriculumSceneSpec,
    *,
    constellation_dir: pathlib.Path,
    taskset_dir: pathlib.Path,
) -> dict[str, object]: ...
def generate_curriculum_split(
    spec: CurriculumSceneSpec,
    *,
    satellites_root: pathlib.Path,
    constellations_root: pathlib.Path,
    tasksets_root: pathlib.Path,
    metadata_root: pathlib.Path,
) -> dict[str, object]: ...
```

Sort satellite filenames before loading. At function entry seed Python, NumPy, and PyTorch through `todd.utils.init_seed(spec.seed)`. Reject pre-existing non-empty targets and metadata. Generate into temporary sibling directories, audit IDs/counts/ranges/loadability, publish with `Path.replace()`, then write sorted, indented metadata including train IDs `0–119` and held-out IDs `120–127` when `num_scenes == 128`.

- [ ] **Step 4: Implement the CLI**

Add exact argparse options from the approved design and call `generate_curriculum_split()` with repository constants. Print the returned metadata as JSON. The command must not accept an overwrite flag.

- [ ] **Step 5: Run generator tests and repository syntax checks**

Run:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=".:${PYTHONPATH:-}" \
  /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_generate_curriculum_scenes.py

env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=".:${PYTHONPATH:-}" \
  /home/hy/miniconda3/envs/aeos/bin/python -m py_compile \
  tools/generate_curriculum_scenes.py \
  tests/test_generate_curriculum_scenes.py
```

Expected: pytest passes and `py_compile` exits 0.

- [ ] **Step 6: Commit safe split generation**

```bash
git add tools/generate_curriculum_scenes.py tests/test_generate_curriculum_scenes.py
git commit -m "feat: generate isolated curriculum scenes"
```

### Task 3: Generate and audit the 256 approved scenes

**Files:**
- Generate: `data/constellations/curriculum_600/`
- Generate: `data/tasksets/curriculum_600/`
- Generate: `data/constellations/curriculum_1800/`
- Generate: `data/tasksets/curriculum_1800/`
- Generate: `work_dirs/curriculum_scenes/curriculum_600/metadata.json`
- Generate: `work_dirs/curriculum_scenes/curriculum_1800/metadata.json`

- [ ] **Step 1: Record the formal-data baseline**

Record file counts for `train`, `val_seen`, `val_unseen`, and `test` under both `data/constellations` and `data/tasksets`. Confirm the four curriculum targets and metadata files do not already exist. If any target exists, stop and report it rather than deleting or overwriting it.

- [ ] **Step 2: Generate the 600-second split**

Run:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" \
  PYTHONPATH=".:${PYTHONPATH:-}" \
  /home/hy/miniconda3/envs/aeos/bin/python \
  tools/generate_curriculum_scenes.py \
  --split curriculum_600 \
  --horizon 600 \
  --num-scenes 128 \
  --satellite-min 1 \
  --satellite-max 5 \
  --task-min 10 \
  --task-max 50 \
  --seed 3407
```

Expected: metadata reports 128 constellation files, 128 taskset files, train IDs `0–119`, held-out IDs `120–127`, and no audit errors.

- [ ] **Step 3: Generate the 1800-second split**

Run the same command with:

```text
--split curriculum_1800 --horizon 1800 --num-scenes 128
--satellite-min 5 --satellite-max 15
--task-min 25 --task-max 150 --seed 3408
```

Expected: the same count and ID audit succeeds for `curriculum_1800`.

- [ ] **Step 4: Independently audit both generated splits**

Run a separate Python audit that loads all 512 JSON files, checks scene IDs `0–127`, verifies every approved range and task window, and prints min/median/max satellite and task counts. Instantiate `BasiliskSceneBackend.from_scene_id()` for scene 0 of both splits without advancing a full trajectory.

- [ ] **Step 5: Confirm formal data remained untouched**

Repeat the Task 3 Step 1 counts and assert they exactly match the baseline. Report generated directory sizes with `du -sh`. Do not delete any generated data after successful verification.

### Task 4: Record the candidate metric and generated assets

**Files:**
- Modify: `改进日志.md`
- Modify: `TODO.md`

- [ ] **Step 1: Append the candidate-route record to `改进日志.md`**

Append a new section after the existing content. State:

- `Q_completion = 0.8*CR + 0.2*PCR` is a next-stage candidate, not a retroactive protocol change;
- job 4276 and its dependency chain keep paper Q;
- 600/1800-second static curriculum scenes were generated with their exact paths, counts, seeds, satellite/task ranges, and audit result;
- no new 3600-second scenes were generated;
- static data generation does not prove PPO improvement;
- future training starts from the Gate-approved V2-Large checkpoint, or V2-2 selected checkpoint on Gate failure.

- [ ] **Step 2: Add executable next actions and evidence to `TODO.md`**

Record the same generated assets and mark only data generation/audit complete. Leave curriculum PPO, 600-second held-out selection, 1800-second held-out selection, existing 3600-second continuation, formal Val, and Test unchecked. Preserve the current running chain and its existing acceptance gates.

- [ ] **Step 3: Validate documentation without absorbing unrelated worktree changes**

Run:

```bash
git diff --check -- TODO.md 改进日志.md
git diff -- TODO.md 改进日志.md
```

Inspect that the new sections are append-only and the pre-existing M3 diff in `改进日志.md` remains unchanged. Stage the new `改进日志.md` hunk separately from the old uncommitted M3 hunk.

- [ ] **Step 4: Commit only the generated-scene records**

Stage `TODO.md` normally. Build and apply an index-only patch for the new appended `改进日志.md` section so the earlier uncommitted M3 section is not included. Verify with `git diff --cached`, then commit:

```bash
git commit -m "docs: record completion curriculum scenes"
```

### Task 5: Final verification and handoff

**Files:**
- Verify: `tools/generate_curriculum_scenes.py`
- Verify: `tests/test_generate_curriculum_scenes.py`
- Verify: generated curriculum data and metadata
- Verify: `TODO.md`, `改进日志.md`

- [ ] **Step 1: Run the complete focused verification**

Run the generator pytest file, `py_compile`, independent data audit, `git diff --check`, and a fresh `git status --short --branch`. Expected: all executable checks pass; only pre-existing unrelated worktree changes remain unstaged.

- [ ] **Step 2: Report exact outputs and rollback boundary**

Report branch, design commit, implementation commits, verification commands, generated directories and sizes, and the current job 4276 chain status. Explain that Git can revert code/docs commits, but generated `data/` and `work_dirs/` assets are not recoverable through Git; removal would require separate explicit approval.
