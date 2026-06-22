# AGENTS.md

Guidance for coding agents working in this repository.

## Environment

- Use the existing `aeos` conda environment. Its binaries are in `/home/hy/miniconda3/envs/aeos/bin`.
- Prefer running commands from the repository root with:

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" <command>
```

- The environment provides Python 3.11 and project dependencies, including PyTorch. Do not recreate or reinstall the environment unless explicitly requested.
- For Python scripts, prefer `/home/hy/miniconda3/envs/aeos/bin/python` or the `PATH` prefix above.

## Project Notes

- This repository implements AEOS constellation scheduling with Basilisk simulation and Transformer-based models.
- Key training/evaluation entry points are documented in `CLAUDE.md`, `README.md`, and `TODO.md`.
- Current reproduction work centers on matching the paper's Table 2 and Table 3 results as closely as possible. The agent's main task is to help reproduce the paper data, diagnose gaps between local metrics and paper metrics, and choose training/evaluation steps that make local results closer to the paper.
- When local results differ from the paper, prioritize finding the cause before launching unrelated new experiments. Check evaluation protocol, data split, annotation pool, checkpoint lineage, model config, loss definition, rollout/filtering rules, and aggregation formula.
- Treat paper alignment as the default success criterion: for each experiment, record which paper row it corresponds to, which metrics match, which metrics diverge, and the likely reason.
- The old 200k CE-only model should be preserved as a historical baseline, but it is not the strict paper reproduction model.
- The paper states that train/val/test are split as: train has 16,218 trajectories, val-seen has 64 scenarios, val-unseen has 64 scenarios, and test has 64 scenarios. Local evaluation split sizes should be checked against these numbers before comparing metrics.
- The paper reports evaluation with 96 parallel simulator environments. For formal reproduction validation/evaluation runs, prefer `environment.world_size=96` when resources allow it. Keep the exact parallelism setting visible in the command or log so evaluation results can be traced.
- Long formal evaluation runs should be launched in a managed session such as `tmux`, with logs under `work_dirs/eval_logs/`, so they continue after the interactive agent session closes. The current helper for the paper Stage-3 full-model evaluation is `scripts/run_stage3_96core_eval_managed.sh`.
- Any task expected to run for more than a few minutes, especially training, rollout generation, large-scale evaluation, or long data processing, should by default be started in a managed background session such as `tmux` rather than in the foreground.
- Before launching a long-running task, prefer creating a dedicated wrapper script under `scripts/` and give the session a distinctive name so the run can be resumed, inspected, and compared later.
- When a long-running task is launched in managed mode, record the session name, command/script, log path, and expected output path in `TODO.md`.
- Do not rely on the interactive editor session staying open. Assume the user may close VSCode or disconnect at any time, and choose a managed/background launch method accordingly.

## Safety

- The worktree may contain user edits and experimental outputs. Do not revert or delete unrelated changes.
- Treat `data/`, `work_dirs/`, and generated trajectory/annotation files as experiment state. Back up active annotations before replacing them.

## Communication

- Reply to the user in clear, organized Chinese by default.
- Avoid mixing Chinese and English when a plain Chinese term is understandable.
- Explain project status and technical conclusions in a human, step-by-step way before giving commands or implementation details.
- If an English technical term is necessary, briefly explain what it means in Chinese the first time it appears.
