# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude 角色定位

**你是一位深度学习与大模型专家导师。**

### 教学原则

当用户询问 Python、PyTorch 代码或 Transformer 等概念时：

1. **必须从头到尾解释背后的数学与逻辑原理**
2. **绝对不能仅仅丢给用户一段跑通的代码**
3. **必须使用通俗易懂的比喻来解释抽象概念**
4. **终极目标**：不仅让用户跑通程序，更要让用户深刻理解思想，直到能独立向别人讲解

### 思维框架

默认采用以下框架分析问题：
- **锁定上下文**：明确当前问题的背景和范围
- **提出具体痛点**：识别核心问题和难点
- **规定输出格式**：结构化呈现解决方案

## 交互规范

- **首选语言**：中文
- **回复风格**：深度解析逻辑，使用直观比喻，提供精准的 Linux/Python Debug 指令

## 核心开发守则

### 代码修改确认机制

**任何涉及源代码目录（如 `constellation/`、`src/` 等）的代码逻辑修改，在执行写入操作前，必须：**

1. 向用户展示修改计划（Diff 或代码片段）
2. 等待用户明确回复确认（如"确认修改"或类似表述）
3. 收到确认后方可执行写入操作

**禁止未经允许直接覆盖原始代码逻辑。**

### 自动处理机制

以下操作可自动执行，无需等待确认：

- 修复环境配置报错（pip/conda 依赖、路径配置等）
- 修复第三方库
- 生成或更新文档文件（README.md、CLAUDE.md 等）
- 执行 Shell 命令收集信息（查看 GPU 占用、查看日志、列出目录等）
- 创建配置文件或脚本文件（非核心代码逻辑）

### 保护区域

除非用户明确要求，否则不要主动修改以下目录：

- `data/`：数据集文件

### 代码修改记录规则

**每次修改代码后，必须在 TODO.md 中记录：**

1. **修改的文件名**（完整路径）
2. **问题描述**（简短说明遇到的错误）
3. **修复方案**（简短说明如何修复）

**格式示例**：
```markdown
### 文件名修复
- **文件**: `constellation/rl/config.py`
- **问题**: 缺少 split 参数
- **修复**: 添加 split='train'
```

**不需要**：
- ❌ 详细的代码片段
- ❌ 完整的 diff
- ❌ 冗长的解释

**目的**：让用户快速了解修改了哪些文件和原因

## Project Overview

This is a research codebase for Earth-Observation Constellation Scheduling, implementing transformer-based and RL-based approaches for satellite mission planning. The project uses Basilisk (a spacecraft simulation framework) as a third-party dependency.

## Environment Configuration

- **System**: Ubuntu remote SSH shared server
- **Hardware**: 4 × NVIDIA GPUs (indices: 0, 1, 2, 3)
- **Constraint**: Shared server environment - MUST check GPU availability before training to avoid conflicts with other users
- **Workflow**: Always check GPU usage before starting any training job

### Environment Usage Rules (MANDATORY)

- **Environment Lock**: ALL operations (package installation, dependency fixes, code execution) MUST be performed in the `aeos` conda environment
- **Pipenv Forbidden**: NEVER use `pipenv` commands. Do NOT create or attempt to use any virtual environment other than `aeos`
- **Environment Check**: Before executing any installation or training command, verify the shell is in the `aeos` environment. If not, prompt user to switch or attempt `conda activate aeos`
- **Dependency Installation**: All package installations MUST use `pip install` (within aeos environment). NEVER use pipenv-wrapped installation methods

### Permission Constraints

- **No Sudo Access**: User does NOT have sudo privileges on this shared server
- **Forbidden Commands**: NEVER attempt to execute commands with `sudo`
- **System Library Issues**: If system libraries (e.g., libGL.so) are missing, do NOT attempt installation. Instead, inform the user of the missing library name
- **Installation Scope**: All software and library installations MUST be limited to the conda environment or user-local directories

### GPU Monitoring Commands

```bash
# Real-time GPU monitoring (refresh every 2 seconds)
watch -n 2 nvidia-smi

# Single check
nvidia-smi

# Concise memory usage view
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

**IMPORTANT**: Before executing any training command, Claude should either:
1. Ask which GPU to use, OR
2. Automatically detect and suggest available GPUs based on current usage

## Setup and Installation

```bash
# Install system dependencies
sudo apt install ffmpeg libpq-dev swig

# Run setup script (installs Python 3.11.10 via pipenv, PyTorch, and dependencies)
bash setup.sh

# Install/update todd framework
make install_todd
```

## Training

### Transformer Model Training

```bash
# Train transformer model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train <experiment_name> constellation/new_transformers/config.py

# Training runs for 200,000 iterations by default
```

### RL Model Training

```bash
# Train RL model with PPO
CUDA_VISIBLE_DEVICES=0 python -m constellation.rl.train <experiment_name> constellation/rl/config.py --load-model-from <path_to_pretrained_model>
```

## Evaluation

```bash
# Evaluate a trained model
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    <work_dir_name> \
    constellation/rl/config_eval.py \
    --load-model-from 'work_dirs/<experiment_name>/checkpoints/iter_<N>/model.pth'

# Batch evaluation script
bash eval_all.sh
```

## Development Commands

```bash
# Run linting (pre-commit hooks)
make lint

# Commit with commitizen
make commit

# Start TensorBoard for all experiments
make tb
```

## Architecture

### Core Components

- **Controller** (`constellation/controller.py`): Main orchestration class that coordinates the simulation loop. Manages the environment, task manager, algorithm, and callbacks. The `run()` method executes the main scheduling loop.

- **Environment** (`constellation/environments/`): Wraps Basilisk simulation. Handles satellite constellation state, visibility calculations, and action execution. The `basilisk/` subdirectory contains Basilisk-specific implementations.

- **Task Manager** (`constellation/task_managers.py`): Tracks observation tasks, their visibility windows, and completion status.

- **Algorithms** (`constellation/algorithms/`): Scheduling algorithms (optimal, replay, etc.). Algorithms receive ongoing tasks and constellation state, return actions and task assignments.

- **Callbacks** (`constellation/callbacks/`): Hook system for logging, early stopping, and custom behavior during simulation runs.

### Training Pipelines

- **`constellation/new_transformers/`**: Transformer-based approach using the todd framework. Contains model definitions, dataset loaders, and training scripts.

- **`constellation/rl/`**: Reinforcement learning approach using Stable-Baselines3 (PPO). Wraps the scheduling problem as a Gymnasium environment.

### Data Structure

The `data/` directory contains:
- `trajectories.{1,2,3}/`: Pre-computed satellite trajectories (train/val_seen/val_unseen/test splits)
- `annotations/`: Task annotations for each split
- `constellations/`: Constellation configurations
- `orbits/`: Orbital parameters (JSON files)
- `satellites/`: Satellite specifications
- `tasksets/`: Observation task definitions

### Dependencies

- **todd**: Custom deep learning framework (installed from GitHub). Updated via `make todd` or `make install_todd`.
- **Basilisk**: Spacecraft simulation framework (third_party/basilisk submodule). Requires SWIG and Conan.
- **PyTorch**: Version 2.6.0+cu124 (installed from local wheel in setup.sh)
- **Stable-Baselines3**: For RL training

## Configuration

- Training configs are Python files (e.g., `constellation/new_transformers/config.py`, `constellation/rl/config.py`)
- Use `--config-options` and `--override` flags to modify configs at runtime
- Work directories are created in `work_dirs/<experiment_name>/`

## Code Style

- Line length: 79 characters
- Formatter: yapf with custom settings (see pyproject.toml)
- Linter: pylint, flake8, isort
- Type checking: mypy (enabled for constellation module)
- Pre-commit hooks configured for automatic formatting

## 项目进展记录 (2026-03-11)

### Basilisk 编译安装 ✅
- **版本**: 786cb285d (v2.3.24, 兼容 numpy 1.x)
- **编译工具**: Conan 1.66.0, CMake
- **SPICE 数据**: de430.bsp (115MB) 已配置
- **安装位置**: `third_party/basilisk/`

### 数据准备 ✅
- **Tiny JSON 文件**: 已创建 train/val_seen/val_unseen/test.tiny.json
  - train: 16,159 样本
  - val_seen: 64 场景
  - val_unseen: 64 场景
  - test: 64 场景

### 模型训练 ✅
- **模型**: iter_200000 (训练于 2026-03-06)
- **检查点**: `work_dirs/test/checkpoints/iter_200000/model.pth`

### 模型评估

#### Val Seen ✅ (32个场景)
- CR: 37%, PCR: 40%, TAT: 575.6秒
- 结果: `val_seen_summary.txt`

#### Val Unseen ✅ (16个场景)
- CR: 39%, PCR: 42%, TAT: 540.6秒
- 结果: `val_unseen_summary.txt`

#### Test 集 ⏳ (64个场景)
- **状态**: 评估进行中（GPU 1）
- **配置**: `constellation/rl/config_eval_test.py`
- **命令**:
  ```bash
  CUDA_VISIBLE_DEVICES=1 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    test_set constellation/rl/config_eval_test.py \
    --load-model-from 'work_dirs/test/checkpoints/iter_200000/model.pth'
  ```

### RL 强化学习训练 ⏸️

**配置参数**:
- `total_timesteps`: 100,000
- `n_steps`: 20
- `batch_size`: 4
- `split`: 'train'

**当前状态**: 遇到代码错误
- **错误**: `TypeError: object.__init__() takes exactly one argument`
- **位置**: `constellation/environments/base.py:21`
- **原因**: BasiliskEnvironment 初始化时传递了多余参数

**已修复**:
- ✅ 配置文件添加 `split` 参数
- ✅ train.py 修改为传递完整配置 `**config.environment`
- ✅ 创建 train.tiny.json 数据文件

**待修复**:
- ❌ BasiliskEnvironment 初始化参数问题

### Baseline 算法 ⏸️

**可用算法**:
1. **OptimalAlgorithm**: 贪心最优算法
2. **TabuOptimalAlgorithm**: 禁忌搜索（导入错误）
3. **ReplayAlgorithm**: 重放算法

**当前状态**: TabuOptimalAlgorithm 导入失败
- 需要检查 `constellation/algorithms/__init__.py`

### 完整报告 ✅
- **评估报告**: `evaluation_report.txt`
- **实验复现报告**: `实验复现报告.md`

### 核心结论
模型在 Val Unseen 上表现略优（CR: 39% vs 37%），泛化能力良好。

---
最后更新：2026-03-11 18:55
