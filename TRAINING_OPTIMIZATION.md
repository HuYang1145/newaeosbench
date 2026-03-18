# 训练加速优化指南（4×RTX 4090）

## 已实现的优化

### 1. torch.compile（✅ 已启用）

**原理**：PyTorch 2.0 的 JIT 编译器，将 Python 代码编译成优化的机器码。

**优化效果**：
- 算子融合：将多个小操作合并成一个 GPU kernel
- 内存优化：减少中间结果的显存分配
- 针对 4090 架构的优化代码生成

**预期加速**：1.5-2x

**如何启用**：
```python
# config.py 中已设置
model = dict(
    type='ConstellationModelRegistry.Model',
    use_compile=True,  # ← 已启用
)
```

**编译模式**：
- `mode='max-autotune'`：最大优化（首次运行会慢，后续快）
- `fullgraph=False`：允许图分割（更稳定）

---

### 2. PyTorch SDPA（Scaled Dot-Product Attention）

**原理**：PyTorch 内置的优化 Attention 实现，自动选择最优后端：
- Flash Attention（如果可用）
- Memory-Efficient Attention
- 标准实现（fallback）

**优化效果**：
- 显存占用：从 O(N²) 降到 O(N)
- 计算速度：2-4x（长序列更明显）

**如何启用**：
```bash
# 方法1：修改 config.py
model = dict(
    type='ConstellationModelRegistry.Model',
    use_compile=True,
    use_sdpa=True,  # ← 启用 SDPA
)

# 方法2：命令行覆盖
--override model.use_sdpa=True
```

**注意**：SDPA 与 torch.compile 可以同时使用，效果叠加。

---

### 3. 多 GPU 并行训练（✅ 已启用）

**配置**：
- 策略：`DDPStrategy`（分布式数据并行）
- 自动检测：`auto_torchrun` 会自动使用所有可见 GPU

**显存分配**（4×4090）：
- 每张卡：batch_size=64 → 总 batch=256
- 显存占用：~18GB/卡（FP32）或 ~12GB/卡（FP16）

---

### 4. 数据加载优化（✅ 已优化）

**配置**：
```python
dataloader=dict(
    type='PrefetchDataLoader',  # 预取数据到 GPU
    num_workers=4,              # 4 个进程并行加载
    sampler=dict(type='DistributedSampler'),  # 多 GPU 数据分片
)
```

---

## 训练命令

### 基础命令（使用所有优化）

```bash
# 使用 4 张 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 auto_torchrun -m constellation.new_transformers.train \
    test_optimized \
    constellation/new_transformers/config.py
```

### 高级命令（自定义配置）

```bash
# 启用所有优化 + 混合精度
CUDA_VISIBLE_DEVICES=0,1,2,3 auto_torchrun -m constellation.new_transformers.train \
    test_optimized \
    constellation/new_transformers/config.py \
    --autocast \
    --override model.use_sdpa=True \
    --override trainer.dataset.batch_size=96
```

### 性能测试命令

```bash
# 测试不同配置的速度
# 1. 基础配置（无优化）
CUDA_VISIBLE_DEVICES=0 auto_torchrun -m constellation.new_transformers.train \
    test_baseline \
    constellation/new_transformers/config.py \
    --override model.use_compile=False \
    --override trainer.dataset.batch_size=24

# 2. 启用 torch.compile
CUDA_VISIBLE_DEVICES=0 auto_torchrun -m constellation.new_transformers.train \
    test_compile \
    constellation/new_transformers/config.py \
    --override model.use_compile=True \
    --override trainer.dataset.batch_size=24

# 3. 启用 torch.compile + SDPA
CUDA_VISIBLE_DEVICES=0 auto_torchrun -m constellation.new_transformers.train \
    test_compile_sdpa \
    constellation/new_transformers/config.py \
    --override model.use_compile=True \
    --override model.use_sdpa=True \
    --override trainer.dataset.batch_size=24

# 4. 全部优化 + 4 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 auto_torchrun -m constellation.new_transformers.train \
    test_full_optimized \
    constellation/new_transformers/config.py \
    --autocast \
    --override model.use_sdpa=True \
    --override trainer.dataset.batch_size=96
```

---

## 性能对比（预估）

| 配置 | GPU | Batch Size | 混合精度 | torch.compile | SDPA | 训练速度 | 显存/卡 |
|------|-----|-----------|---------|--------------|------|---------|---------|
| 基础 | 1 | 24 | ❌ | ❌ | ❌ | 1.0x | 18GB |
| +compile | 1 | 24 | ❌ | ✅ | ❌ | 1.5x | 18GB |
| +SDPA | 1 | 24 | ❌ | ✅ | ✅ | 2.0x | 16GB |
| +FP16 | 1 | 48 | ✅ | ✅ | ✅ | 2.5x | 12GB |
| **推荐** | 4 | 64 | ✅ | ✅ | ✅ | **8-10x** | 12GB |

---

## 注意事项

### torch.compile 首次运行

首次运行时，torch.compile 会编译模型（需要 5-10 分钟）：
```
[INFO] Compiling model... (this may take a while)
```

后续运行会直接使用缓存的编译结果。

### 显存不足（OOM）

如果遇到 OOM 错误：
```bash
# 减小 batch_size
--override trainer.dataset.batch_size=32

# 或启用混合精度
--autocast
```

### 数值稳定性

混合精度可能影响数值稳定性，建议：
1. 先在小数据集上验证
2. 对比 FP32 和 FP16 的 loss 曲线
3. 如果 loss 不稳定，禁用 `--autocast`

---

## 监控训练速度

查看 TensorBoard：
```bash
tensorboard --logdir work_dirs/test_optimized
```

关注指标：
- `iter/s`：每秒迭代次数（越高越好）
- `samples/s`：每秒处理样本数
- `GPU utilization`：GPU 利用率（应接近 100%）

---

## 故障排查

### 问题1：torch.compile 报错

```
RuntimeError: Dynamo is not supported on Python 3.11+
```

**解决**：降级到 Python 3.10 或禁用 compile：
```bash
--override model.use_compile=False
```

### 问题2：SDPA 不生效

检查 PyTorch 版本：
```bash
python -c "import torch; print(torch.__version__)"
# 需要 >= 2.0.0
```

### 问题3：多 GPU 不均衡

检查 GPU 使用情况：
```bash
watch -n 1 nvidia-smi
```

确保所有 GPU 的利用率接近。

---

## 推荐配置（4×RTX 4090）

```bash
# 最优配置
CUDA_VISIBLE_DEVICES=0,1,2,3 auto_torchrun -m constellation.new_transformers.train \
    production \
    constellation/new_transformers/config.py \
    --autocast \
    --override model.use_sdpa=True \
    --override trainer.dataset.batch_size=64 \
    --override trainer.dataloader.num_workers=4
```

**预期性能**：
- 训练速度：~8-10x 基础配置
- 200k 迭代时间：~12-16 小时（vs 基础配置 ~5 天）
- 显存占用：~12GB/卡
- GPU 利用率：>95%
