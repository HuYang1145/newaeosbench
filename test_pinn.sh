#!/bin/bash
# PINN 量纲测试脚本

echo "🧪 开始 PINN 量纲测试..."

# 设置环境变量
export LIBRARY_PATH=~/.local/lib:/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=~/.local/lib:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 检测空闲 GPU
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$2 < 1000 {print $1; exit}')

if [ -z "$FREE_GPU" ]; then
    echo "❌ 没有空闲 GPU"
    exit 1
fi

echo "✅ 使用 GPU: $FREE_GPU"

# 运行 1 次迭代测试
CUDA_VISIBLE_DEVICES=$FREE_GPU PYTHONPATH=:${PYTHONPATH} conda run -n aeos auto_torchrun -m constellation.new_transformers.train \
    pinn_test \
    constellation/new_transformers/config_test_pinn.py

echo "✅ 测试完成！请检查上方的量纲输出"
