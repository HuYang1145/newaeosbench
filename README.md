# Constellation

This repository is the official implementation of "Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology".

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116515)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26297-b31b1b.svg)](https://arxiv.org/abs/2510.26297)

## Installation

```bash
sudo apt install ffmpeg libpq-dev swig
bash setup.sh
```

### Known Issues and Solutions

#### Basilisk Version Compatibility

**Issue**: The project requires Basilisk version compatible with numpy 1.x (required by todd-ai), but newer Basilisk versions require numpy 2.0+.

**Solution**: Use Basilisk commit `786cb285d` (last version before numpy 2.0 support). This is automatically handled in `setup.sh`.

#### lvis-api Dependency

**Issue**: todd-ai requires `lvis-api` with `boundary_utils` module, which is not available in PyPI version.

**Solution**: Install from GitHub source:
```bash
pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021 --no-build-isolation
```

#### pytest-html Missing

**Issue**: Basilisk compilation requires `pytest-html` package.

**Solution**: Install before compiling Basilisk:
```bash
pip install pytest-html
```

## data

If you want to use all data to reproduct our paper:

```bash
git clone git@hf.co:datasets/MessianX/AEOS-dataset ./data
find ./data -type f -name '*.tar' -print0 | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

Or, you can just download the val_seen/val_unseen/test from the trajectories inside our hf repo and unzip them to only evaluate your own model:

```bash
# TODO: urls
# suppose you have download these requested data
find ./data -type f -name '*.tar' -print0 | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

## Steps

### 1. Confirm the data

The right file tree should be like this：

```
data/
├── trajectories.1/
│   ├── test/
│   ├── train/
│   │   ├── 00/         # contains pth and json files
│   │   ├── 01/
│   │   ├── ...
│   ├── val_seen/
│   └── val_unseen/
├── trajectories.2/
├── trajectories.3/
├── annotations/
│   ├── test.json
│   ├── train.json
│   ├── val_seen.json
│   └── val_unseen.json
├── constellations/
│   ├── test/
│   ├── train/
│   ├── val_seen/
│   └── val_unseen/
├── orbits/
├── satellites/
└── tasksets/
```

### 2. Train the model

Use the command below to train our model:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train test constellation/new_transformers/config.py
```

This will continue till 200000 iters.

### 3. Eval the model

Use the command below to evaluate the model:

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    work_dir_name \
    constellation/rl/config_eval.py \
    --load-model-from 'work_dirs/test/checkpoints/iter_200000/model.pth'
```

### 4. Baseline Evaluation

Run baseline algorithms (OptimalAlgorithm) for comparison:

```bash
# Single process
PYTHONPATH=. python tools/test_baseline.py baseline_optimal 0

# Multi-process (4 workers)
PYTHONPATH=. python tools/test_baseline.py baseline_optimal 4
```

## Results

### Transformer Model (iter_200000)

| Split | Scenarios | CR (%) | PCR (%) | TAT (s) |
|-------|-----------|--------|---------|---------|
| Val Seen | 32 | 37 | 40 | 575.6 |
| Val Unseen | 16 | 39 | 42 | 540.6 |
| Test | 64 | Running | - | - |

### Baseline (OptimalAlgorithm)

| Split | Scenarios | Status |
|-------|-----------|--------|
| Val Seen | 32 | Running |
| Val Unseen | 32 | Pending |
| Test | 64 | Pending |

**Note**: Baseline evaluation is currently in progress. Results will be updated upon completion.

## Project Structure

```
.
├── constellation/          # Core source code
│   ├── algorithms/        # Scheduling algorithms (Optimal, Replay, etc.)
│   ├── callbacks/         # Training callbacks
│   ├── data/             # Data structures (Constellation, Task, etc.)
│   ├── environments/     # Basilisk simulation environment
│   ├── evaluators/       # Evaluation metrics (CR, TAT, Power)
│   ├── new_transformers/ # Transformer model implementation
│   ├── rl/              # RL training (PPO) and evaluation
│   └── task_managers.py  # Task scheduling management
├── data/                 # Dataset (not in git)
│   ├── annotations/      # Train/val/test split definitions
│   ├── constellations/   # Satellite constellation configs
│   ├── tasksets/        # Observation task definitions
│   └── trajectories.*/  # Pre-computed trajectories
├── tools/               # Utility scripts
│   └── test_baseline.py # Baseline algorithm evaluation
├── work_dirs/           # Training outputs (not in git)
│   ├── test/           # Main training run
│   │   └── checkpoints/ # Model checkpoints (iter_100000, iter_200000)
│   ├── rl_eval_*/      # Evaluation results
│   └── test_baseline/  # Baseline evaluation results
├── archive_logs/        # Historical evaluation logs
├── third_party/         # External dependencies
│   └── basilisk/       # Spacecraft simulation framework
└── EVALUATION_RESULTS.md # Consolidated evaluation results
```

