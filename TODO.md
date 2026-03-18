# TODO

## 已完成 ✅

### 训练加速优化（2026-03-17）
- ✅ 添加 torch.compile 支持（model.py）
- ✅ 添加 PyTorch SDPA 支持（model.py）
- ✅ 优化训练配置（config.py）
  - batch_size: 24 → 64
  - num_workers: 2 → 4
  - 启用 torch.compile
- ✅ 创建优化文档
  - TRAINING_OPTIMIZATION.md（详细指南）
  - OPTIMIZATION_SUMMARY.md（快速参考）
- ✅ 更新 README.md
  - 添加项目核心原理说明
  - 添加 Basilisk 作用说明
  - 添加 Transformer 输入输出规格
  - 添加训练加速指南

### 代码修复
- ✅ BasiliskEnvironment 初始化参数问题
- ✅ 配置文件添加 split 参数
- ✅ 创建 train.tiny.json 数据文件

### 模型评估
- ✅ Val Seen 评估（32个场景）
- ✅ Val Unseen 评估（16个场景）
- ✅ 生成评估报告

## 待完成 ⏳

### Test 集评估
- ⏳ Test 集评估（64个场景）
  - 状态：评估进行中（GPU 1）
  - 配置：constellation/rl/config_eval_test.py

### RL 强化学习训练
- ⏸️ 修复 BasiliskEnvironment 初始化参数问题
- ⏸️ 完成 RL 训练流程

### Baseline 算法
- ⏸️ 修复 TabuOptimalAlgorithm 导入错误
- ⏸️ 运行 Baseline 对比实验

## 优化建议 💡

### 进一步加速
1. 尝试 Gradient Checkpointing（节省显存）
2. 使用 torch.compile 的 reduce-overhead 模式
3. 优化数据预处理流程

### 实验验证
1. 对比不同优化配置的训练速度
2. 验证混合精度对精度的影响
3. 测试不同 batch size 的效果

---
最后更新：2026-03-17 22:10
