#!/bin/bash
# 自动检测空闲 GPU 并启动训练

# 设置环境变量（启用 torch.compile）
export LIBRARY_PATH=~/.local/lib:/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=~/.local/lib:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 检测空闲 GPU（显存使用 < 1000MB）
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$2 < 1000 {printf "%s,", $1}' | sed 's/,$//')

if [ -z "$FREE_GPUS" ]; then
    echo "❌ 没有空闲 GPU，请稍后再试"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=table
    exit 1
fi

EXPERIMENT_NAME=${1:-"production"}

echo "✅ 使用 GPU: $FREE_GPUS"
echo "📝 实验名称: $EXPERIMENT_NAME"
echo ""

# 启动训练
CUDA_VISIBLE_DEVICES=$FREE_GPUS conda run -n aeos --no-capture-output auto_torchrun -m constellation.new_transformers.train \
    $EXPERIMENT_NAME \
    constellation/new_transformers/config.py \
    --autocast
