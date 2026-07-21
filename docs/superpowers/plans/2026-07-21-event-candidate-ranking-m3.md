# M3 Event Candidate Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate event-level stay/switch-duration counterfactual labels and a scene-split candidate Critic without entering PPO or Test.

**Architecture:** Extend the deterministic local-branch wrapper so one satellite can hold a selected action for a bounded event commitment while all other actions continue to come from frozen Stage3. Generate one 300-second branch per candidate, derive stable 180/300-second pair labels, train a small scorer with scene-level folds, and only expose an Actor rerank path when preregistered offline gates pass.

**Tech Stack:** Python 3.11, PyTorch, pytest, Basilisk, existing AEOS `Controller`/`TaskManager`, Slurm CPU jobs, Git.

---

## File map

- Create `constellation/new_transformers/event_candidate.py`: candidate specifications, event-point selection, stable pair labels, and audits.
- Modify `constellation/new_transformers/local_action_branch.py`: multi-second `ControlledCommitmentAlgorithm`; preserve one-step compatibility.
- Create `tools/generate_event_candidate_branches_m3.py`: deterministic single-scene branch generation.
- Create `tools/generate_event_candidate_dataset_m3.py`: multi-scene orchestration and label gate.
- Create `constellation/new_transformers/event_candidate_critic.py`: samples, scorer, pairwise loss, baselines, and metrics.
- Create `tools/train_event_candidate_critic_m3.py`: scene-level folds and M3-C acceptance.
- Create `scripts/run_event_candidate_m3_pilot_slurm.sh`: M3-A/B/C CPU Slurm wrapper.
- Conditionally modify `tools/rollout_model_trajectories.py` only if M3-C passes.
- Add focused `tests/test_event_candidate*.py` files and update `TODO.md` / `改进日志.md` from real evidence.

### Task 1: Event candidates and stable preference labels

**Files:**
- Create: `constellation/new_transformers/event_candidate.py`
- Test: `tests/test_event_candidate.py`

- [ ] **Step 1: Write failing tests for candidate expansion and stable labels**

    def test_build_event_candidate_specs_limits_idle_to_one_second():
        specs = build_event_candidate_specs(stay_task_id=-1, switch_task_id=8)
        assert [(x.task_id, x.commitment_seconds) for x in specs] == [
            (-1, 1), (8, 1), (8, 5), (8, 15), (8, 30), (8, 60),
        ]

    def test_stable_preference_requires_180_300_agreement():
        branches = {
            'a': _branch(task_id=7, duration=15, cost180=3.0, cost300=2.8),
            'b': _branch(task_id=8, duration=30, cost180=3.2, cost300=3.1),
        }
        audit = audit_preference_pair('a', 'b', branches, min_margin=0.01)
        assert audit.accepted and audit.better_branch == 'a'

    def test_stable_preference_censors_reversal():
        branches = {
            'a': _branch(task_id=7, duration=15, cost180=3.0, cost300=3.2),
            'b': _branch(task_id=8, duration=30, cost180=3.1, cost300=3.0),
        }
        audit = audit_preference_pair('a', 'b', branches, min_margin=0.01)
        assert not audit.accepted
        assert audit.reason == 'horizon_reversal'

- [ ] **Step 2: Run the test and verify RED**

    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_event_candidate.py

Expected: collection fails because `event_candidate` does not exist.

- [ ] **Step 3: Implement the minimal API**

    @dataclasses.dataclass(frozen=True)
    class EventCandidateSpec:
        name: str
        task_id: int
        commitment_seconds: int
        action_kind: Literal['stay', 'switch']

    @dataclasses.dataclass(frozen=True)
    class PreferenceAudit:
        first_branch: str
        second_branch: str
        accepted: bool
        reason: str
        better_branch: str | None
        worse_branch: str | None
        margin_300: float | None

    def build_event_candidate_specs(
        *, stay_task_id: int, switch_task_id: int,
        commitments: Sequence[int] = (1, 5, 15, 30, 60),
    ) -> list[EventCandidateSpec]:
        """Return unique stay/switch-duration candidates; idle is one second."""

    def audit_preference_pair(
        first: str, second: str, branches: Mapping[str, Any],
        *, min_margin: float = 0.01,
    ) -> PreferenceAudit:
        """Accept only quality-protected pairs agreeing at 180 and 300 s."""

Use exact rejection reasons `identical_candidate`, `missing_window`, `small_margin`, `horizon_reversal`, `quality_protection`, and `accepted`.

- [ ] **Step 4: Add event-point stratification and audit tests, then implement**

Test transitions across 300-second bins, replayability, 300-second remaining horizon, accepted scene/pair counts, agreement, and winner-duration distribution. Implement `find_event_decisions()` and `summarize_preference_audits()` in the same module.

- [ ] **Step 5: Run focused tests and commit**

    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_event_candidate.py
    git add constellation/new_transformers/event_candidate.py tests/test_event_candidate.py
    git commit -m "feat: define m3 event candidates and stable labels"

### Task 2: Multi-second controlled commitment

**Files:**
- Modify: `constellation/new_transformers/local_action_branch.py`
- Test: `tests/test_event_commitment_branch.py`

- [ ] **Step 1: Write failing tests for bounded commitment**

    def test_commitment_overrides_only_target_for_requested_seconds():
        algorithm = ControlledCommitmentAlgorithm(
            timer=timer, base_algorithm=base, decision_time=5,
            satellite_index=0, forced_task_id=3, commitment_seconds=5,
        )
        assignments = [_step(algorithm, time) for time in range(4, 12)]
        assert [row[0] for row in assignments] == [-1, 3, 3, 3, 3, 3, -1, -1]
        assert all(row[1] == 5 for row in assignments)
        assert algorithm.actual_commitment_seconds == 5
        assert algorithm.interruption_reason == 'expired'

    def test_commitment_stops_when_task_leaves_ongoing_set():
        algorithm = _commitment(task_id=3, seconds=5)
        _step(algorithm, time=5, ongoing_task_ids=[3, 5])
        _step(algorithm, time=6, ongoing_task_ids=[3, 5])
        _step(algorithm, time=7, ongoing_task_ids=[5])
        assert algorithm.actual_commitment_seconds == 2
        assert algorithm.interruption_reason == 'task_unavailable'

- [ ] **Step 2: Verify RED**

    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_event_commitment_branch.py

Expected: import fails because `ControlledCommitmentAlgorithm` is absent.

- [ ] **Step 3: Implement the wrapper**

Call the frozen base algorithm every second, capture the state exactly once at `decision_time`, override only one satellite while the task remains ongoing and elapsed time is below the request, then expose:

    requested_commitment_seconds: int
    actual_commitment_seconds: int
    interruption_reason: str | None
    original_task_id: int | None
    applied_task_id: int | None
    decision_state_signature: str | None
    decision_context: dict[str, Any] | None

Reuse forced-action/state/context helpers without changing `ControlledActionAlgorithm` behavior.

- [ ] **Step 4: Verify GREEN and old branch regression**

    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
      tests/test_event_commitment_branch.py tests/test_local_action_branch.py

- [ ] **Step 5: Commit**

    git add constellation/new_transformers/local_action_branch.py \
      tests/test_event_commitment_branch.py
    git commit -m "feat: add controlled event commitment branches"

### Task 3: Single-scene M3 branch generator

**Files:**
- Create: `tools/generate_event_candidate_branches_m3.py`
- Test: `tests/test_generate_event_candidate_branches_m3.py`

- [ ] **Step 1: Write failing helper tests**

Cover the highest Stage3 candidate distinct from stay, deterministic branch names, common state/context validation, one-run prefix extraction, stable pair serialization, and preservation of rejected audits.

    def test_resolve_switch_uses_highest_candidate_different_from_stay():
        assert resolve_switch_task_id(
            stay_task_id=7,
            actor_logits=[5.0, 9.0, 8.0],
            ongoing_task_ids=[7, 8],
        ) == 8

- [ ] **Step 2: Verify RED**

    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
      tests/test_generate_event_candidate_branches_m3.py

- [ ] **Step 3: Implement generator helpers and CLI**

CLI contract:

    checkpoint reference_trajectory output_root
    --m2-checkpoint work_dirs/event_heads_m2_10k/checkpoints/iter_10000/model.pth
    --split train --scene-id N --device cpu
    --horizons 60 180 300 --max-decisions 2
    --commitments 1 5 15 30 60 --min-margin 0.01 --overwrite

Run stay-1 first to capture Stage3 logits, resolve one distinct switch, expand candidates, replay each branch once to `decision_time + max(horizons) + 1`, and save duration metadata, raw horizons, all audits, and accepted preferences.

When `--m2-checkpoint` is present, instantiate a frozen residual-zero M2 shadow
model, construct `TemporalHistoryTensors` from the same assignment prefix, and
store the selected-edge continue probability plus five duration logits in the
common decision context. The formal Slurm wrapper must provide this checkpoint;
the bounded correctness smoke may omit it and must then mark the M2 baseline
unavailable.

- [ ] **Step 4: Add bounded real smoke arguments**

Allow `--max-decisions 1 --horizons 5 10` for correctness smoke while preserving formal defaults. Never use Test.

- [ ] **Step 5: Run tests and commit**

    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
      tests/test_generate_event_candidate_branches_m3.py \
      tests/test_event_commitment_branch.py
    git add tools/generate_event_candidate_branches_m3.py \
      tests/test_generate_event_candidate_branches_m3.py
    git commit -m "feat: generate m3 event candidate branches"

### Task 4: Multi-scene dataset and label gate

**Files:**
- Create: `tools/generate_event_candidate_dataset_m3.py`
- Create: `scripts/run_event_candidate_m3_pilot_slurm.sh`
- Test: `tests/test_event_candidate_dataset_m3.py`
- Test: `tests/test_event_candidate_m3_scripts.py`

- [ ] **Step 1: Write failing command and audit tests**

Test Stage3 reference discovery, exact `aeos` interpreter propagation, absence of Test in the wrapper, worker validation, and the fields `accepted_scene_count`, `stable_pair_count`, `horizon_agreement`, duration distribution, and `gate.decision`.

- [ ] **Step 2: Verify RED, then implement orchestration**

Use `ThreadPoolExecutor` only for independent scene subprocesses. Propagate `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `PYTHONPATH`, write per-scene logs, and aggregate `dataset_summary.json` plus `label_audit.json`.

- [ ] **Step 3: Implement the exact M3-B gate**

    ready = (
        accepted_scene_count >= 6
        and stable_pair_count >= 32
        and horizon_agreement >= 0.70
        and winning_duration_class_count >= 3
        and max_winning_duration_fraction <= 0.85
    )

- [ ] **Step 4: Add and validate Slurm wrapper**

Request `local-10` / `lab_team` CPU resources, 8 train scenes, 2 decisions, 60/180/300 horizons, and existing Stage3 checkpoint/reference root. Run Critic only if the label gate says `ready_for_critic`.

    bash -n scripts/run_event_candidate_m3_pilot_slurm.sh
    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
      tests/test_event_candidate_dataset_m3.py tests/test_event_candidate_m3_scripts.py

- [ ] **Step 5: Commit**

    git add tools/generate_event_candidate_dataset_m3.py \
      scripts/run_event_candidate_m3_pilot_slurm.sh \
      tests/test_event_candidate_dataset_m3.py tests/test_event_candidate_m3_scripts.py
    git commit -m "feat: add m3 event candidate data gate"

### Task 5: Event candidate Critic

**Files:**
- Create: `constellation/new_transformers/event_candidate_critic.py`
- Test: `tests/test_event_candidate_critic.py`

- [ ] **Step 1: Write failing sample-encoding tests**

Test idle encoding, task lookup, finite logits, duration distinction, group identity, and rejection of state/context mismatch.

    def test_candidate_features_distinguish_commitment_duration():
        short = encode_candidate(_branch(duration=5), _context())
        long = encode_candidate(_branch(duration=60), _context())
        assert short.shape == long.shape
        assert not torch.equal(short, long)

- [ ] **Step 2: Verify RED, then implement scorer**

    class EventCandidateCritic(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.network(features).squeeze(-1)

    def pairwise_ranking_loss(better_cost, worse_cost, weights):
        return (F.softplus(better_cost - worse_cost) * weights).mean()

Features must include target satellite, task or idle row, sanitized Actor logit, idle flag, previous-task match, log duration, run length, switch counts, progress ratio, release/due features, and compatibility.

- [ ] **Step 3: Implement metrics and baselines with tests**

Report pairwise accuracy, group top-1 exact, regret, stay/switch and duration subgroups, winner distribution, Stage3-logit, always-stay, and M2-rule baseline when shadow predictions are present. Mark M2 unavailable when fields are absent.

- [ ] **Step 4: Run tests and commit**

    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
      tests/test_event_candidate_critic.py
    git add constellation/new_transformers/event_candidate_critic.py \
      tests/test_event_candidate_critic.py
    git commit -m "feat: add m3 event candidate critic"

### Task 6: Scene-level Critic training

**Files:**
- Create: `tools/train_event_candidate_critic_m3.py`
- Test: `tests/test_train_event_candidate_critic_m3.py`

- [ ] **Step 1: Write failing scene-fold and gate tests**

    def test_scene_folds_never_overlap():
        train, val = split_groups_by_scene(groups, num_folds=4, fold_index=0)
        assert {x.scene_id for x in train}.isdisjoint(
            {x.scene_id for x in val}
        )

    def test_combined_gate_requires_three_folds_and_baseline_gain():
        summary = summarize_cross_validation(
            _accepted_folds(3) + [_failed_fold()]
        )
        assert summary['accepted'] is True

- [ ] **Step 2: Verify RED, then implement loader/training**

Load only accepted preferences, normalize from train-fold statistics, train AdamW with pairwise loss, and save each fold bundle/summary. Keep scenes disjoint.

- [ ] **Step 3: Implement exact M3-C acceptance**

    accepted = (
        critic_accuracy >= 0.60
        and critic_accuracy - strongest_baseline_accuracy >= 0.05
        and critic_regret <= strongest_baseline_regret
        and accepted_folds >= 3
        and not subgroup_collapse
    )

Emit `ready_for_actor_smoke` or `stop_before_actor`.

- [ ] **Step 4: Run tests and commit**

    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
      tests/test_train_event_candidate_critic_m3.py \
      tests/test_event_candidate_critic.py
    git add tools/train_event_candidate_critic_m3.py \
      tests/test_train_event_candidate_critic_m3.py
    git commit -m "feat: train m3 event candidate critic"

### Task 7: Real M3-A/B/C execution

**Files:**
- Modify: `TODO.md`
- Modify: `改进日志.md`

- [ ] **Step 1: Submit bounded real correctness smoke**

Use a CPU Slurm job for train scene 0, one event and 5/10-second diagnostic horizons. Verify signatures, duration metadata, deterministic rerun, and schema.

- [ ] **Step 2: Submit formal 8-scene pilot**

    sbatch scripts/run_event_candidate_m3_pilot_slurm.sh

Wait for completion. Record job ID, elapsed time, exit code, output roots, failures, groups/branches/pairs, gate decision, and resources.

- [ ] **Step 3: Apply stop rule exactly**

If `label_audit.json` says `stop_before_critic`, do not train, rerank, run Val, or tune margin. If ready, run Task 6 and apply M3-C.

- [ ] **Step 4: Record results and commit**

    git add TODO.md 改进日志.md
    git commit -m "docs: record m3 event candidate pilot"

### Task 8: Conditional Actor smoke

**Files:**
- Modify only if M3-C passes: `tools/rollout_model_trajectories.py`
- Create only if M3-C passes: `scripts/run_event_candidate_m3_actor_smoke_slurm.sh`
- Test only if M3-C passes: `tests/test_event_candidate_actor_m3.py`
- Modify: `TODO.md`
- Modify: `改进日志.md`

- [ ] **Step 1: Read machine gate before touching Actor**

Proceed only when Critic summary decision is `ready_for_actor_smoke`.

- [ ] **Step 2: If passed, write RED tests for explicit-off reranking**

Require a Critic checkpoint, enumerate the same stay/switch-duration set, override Stage3 only above saved calibration margin, otherwise fall back. Reject incompatible M2 commitment flags.

- [ ] **Step 3: If passed, implement and run 3,600-second scene-0 smoke**

Stop if any completion metric drops over 0.5 percentage points or `CS_paper` does not improve. Only a passing smoke may enter 8+8 Val; never Test.

- [ ] **Step 4: If gate failed, document intentional skip**

Record the exact M3-B/M3-C failure and that no Actor code, Val, PPO, or Test ran.

### Task 9: Final verification and handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-21-event-candidate-ranking-m3.md`
- Modify: `TODO.md`
- Modify: `改进日志.md`

- [ ] **Step 1: Run the complete relevant suite**

    /home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
      tests/test_event_candidate.py \
      tests/test_event_commitment_branch.py \
      tests/test_generate_event_candidate_branches_m3.py \
      tests/test_event_candidate_dataset_m3.py \
      tests/test_event_candidate_m3_scripts.py \
      tests/test_event_candidate_critic.py \
      tests/test_train_event_candidate_critic_m3.py \
      tests/test_local_action_branch.py \
      tests/test_generate_local_action_branches_tool.py \
      tests/test_local_graph_q_critic.py \
      tests/test_event_action.py tests/test_event_policy.py \
      tests/test_rollout_model_candidates.py
    bash -n scripts/run_event_candidate_m3_pilot_slurm.sh
    git diff --check

- [ ] **Step 2: Review protected scope**

Confirm `.claude/settings.json`, `CLAUDE.md`, `dataset.py.bak_20260623_102131`, and Basilisk AutoTeX remain unstaged.

- [ ] **Step 3: Mark evidence-backed checkboxes and commit**

    git add docs/superpowers/plans/2026-07-21-event-candidate-ranking-m3.md \
      TODO.md 改进日志.md
    git commit -m "docs: complete m3 event candidate evaluation"

- [ ] **Step 4: Report branch, baseline, commits, Slurm jobs, tests, result, and revert targets**

Keep `codex/offline-critic-ranking`. Do not merge or push without explicit direction.
