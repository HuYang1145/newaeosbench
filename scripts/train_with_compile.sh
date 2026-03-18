#!/bin/bash
# 训练脚本 - 启用 torch.compile 加速

# 设置环境变量（让 triton 编译器找到 libcuda.so）
export LIBRARY_PATH=~/.local/lib:/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=~/.local/lib:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 训练参数
EXPERIMENT_NAME=${1:-"production"}
GPU_IDS=${2:-"0,1,2,3"}

echo "🚀 启动训练 (torch.compile 已启用)"
echo "实验名称: $EXPERIMENT_NAME"
echo "使用 GPU: $GPU_IDS"

# 启动训练
CUDA_VISIBLE_DEVICES=$GPU_IDS auto_torchrun -m constellation.new_transformers.train \
    $EXPERIMENT_NAME \
    constellation/new_transformers/config.py \
    --autocast
