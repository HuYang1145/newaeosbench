#!/bin/bash
# 监控训练进度

WORK_DIR=${1:-"pinn_production"}

echo "=== 训练监控: $WORK_DIR ==="
echo ""

# GPU 使用情况
echo "📊 GPU 状态:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep -E "^(0|3)"
echo ""

# 训练进程
echo "🔄 训练进程:"
ps aux | grep "constellation.new_transformers.train" | grep -v grep | wc -l
echo ""

# 最新日志
echo "📝 最新训练日志:"
tail -n 10 work_dirs/$WORK_DIR/*.log 2>/dev/null | grep -E "iter|loss|物理" | tail -5
