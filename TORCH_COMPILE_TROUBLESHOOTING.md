# torch.compile 故障排查

## 问题

训练时遇到 torch.compile 编译错误：
```
CalledProcessError: Command '['/usr/bin/gcc', ...] returned non-zero exit status 1.
```

## 原因

torch.compile 需要编译 C++ 代码，但系统缺少编译依赖或配置不正确。

## 解决方案

### 方案 1：暂时禁用 torch.compile（推荐）

在 `config.py` 中设置：
```python
model = dict(
    type='ConstellationModelRegistry.Model',
    use_compile=False,  # 禁用
)
```

**影响**：损失 1.5-2x 加速，但训练仍然可以正常进行。

### 方案 2：修复编译环境

安装缺失的依赖：
```bash
# 检查 gcc 版本
gcc --version  # 需要 >= 7.0

# 安装编译工具
conda install -c conda-forge gcc_linux-64 gxx_linux-64

# 或使用系统包管理器
sudo apt install build-essential
```

然后重新启用：
```python
model = dict(
    type='ConstellationModelRegistry.Model',
    use_compile=True,
)
```

### 方案 3：使用 reduce-overhead 模式

如果 max-autotune 失败，尝试更简单的模式：

修改 `model.py` 第 446 行：
```python
self._transformer = torch.compile(
    self._transformer,
    mode='reduce-overhead',  # 更简单的优化
    fullgraph=False,
)
```

## 性能对比

| 配置 | 加速 | 稳定性 |
|------|------|--------|
| use_compile=False | 1.0x | ✅ 最稳定 |
| mode='reduce-overhead' | 1.3x | ✅ 较稳定 |
| mode='max-autotune' | 1.5-2x | ⚠️ 需要完整编译环境 |

## 推荐

**对于 4×4090**：
- 即使不用 torch.compile，4 GPU 并行 + 大 batch size 已经有 ~4x 加速
- 可以先不修复 compile，等训练稳定后再优化
