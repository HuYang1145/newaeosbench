#!/usr/bin/env bash
# 生成用于后续场景采样的卫星池。
#
# 流程：
# 1. 先生成筛选卫星用的共享 MRP 任务集：data/tasksets/mrp.json。
# 2. 创建 data/satellites/<split>/ 目录，其中 val_seen 复用 train 的卫星池。
# 3. 用 32 个 torchrun 进程并行运行 generate_satellites.py。
#
# 注意：这个脚本生成的是“候选卫星池”，不是最终训练场景；最终场景还需要
# generate_constellations_and_tasksets.py 从卫星池里采样星座并生成 taskset。

PYTHONPATH=:${PYTHONPATH} python tools/generate_mrp_taskset.py

mkdir -p data/satellites/train
ln -s ${PWD}/data/satellites/train data/satellites/val_seen
mkdir -p data/satellites/val_unseen
mkdir -p data/satellites/test

PYTHONPATH=:${PYTHONPATH} torchrun --nproc-per-node 32 tools/generate_satellites.py
