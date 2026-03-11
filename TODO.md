# TODO List

## ✅ 项目已完成 (2026-03-11)

### 环境配置
- [x] aeos conda 环境配置
- [x] Basilisk 编译安装（版本 786cb285d, v2.3.24）
- [x] SPICE 数据文件配置（de430.bsp）
- [x] 创建 tiny.json 数据文件（train/val_seen/val_unseen/test）

### 模型训练
- [x] Transformer 模型训练（iter_200000，完成于 2026-03-06）

### 模型评估
- [x] **Val Seen 评估完成**（32个场景）
  - 平均 CR: 37%, PCR: 40%
  - 结果: val_seen_summary.txt
- [x] **Val Unseen 评估完成**（16个场景）
  - 平均 CR: 39%, PCR: 42%
  - 结果: val_unseen_summary.txt
- [⏳] **Test 集评估进行中**（64个场景）
  - 状态: GPU 1 运行中
  - 预计完成时间: 待定

### 结果分析
- [x] 生成完整评估报告（evaluation_report.txt）
- [x] 对比 Val Seen 和 Val Unseen 性能
- [x] 模型泛化能力验证

## ⏳ 进行中任务 (2026-03-11 19:50)

### Test 集评估
- [⏳] **Test 集评估**（64个场景）
  - 状态: GPU 1 运行中（已运行 46+ 分钟）
  - 配置文件: constellation/rl/config_eval_test.py
  - 预计完成时间: 约 1 小时

## ⏸️ 待完成任务

### RL 强化学习训练
- [❌] **RL 训练**（PPO fine-tuning）
  - 状态: 代码不完整，无法运行
  - 已修复: 14 个代码错误
  - 剩余问题: `environment.py` 缺少 `end_time` 属性
  - 建议: 使用 `controller_environment.py` 替代
  - 详细问题记录:
    - 配置缺少参数、训练脚本未传递配置
    - BaseEnvironment/TaskManager 初始化错误
    - 导入不存在的 Evaluator (PCompletionRateEvaluator 等)
    - TaskManager 属性名错误 (valid_tasks → ongoing_tasks)
    - Evaluator 方法不存在 (log_progress)
    - 奖励计算空值错误、完成状态属性错误
    - BasiliskEnvironment 缺少 end_time 属性

### Baseline 对比
- [ ] **运行 Baseline 算法**
  - OptimalAlgorithm（贪心最优）
  - TabuOptimalAlgorithm（禁忌搜索）
  - ReplayAlgorithm（重放算法）
  - 状态: TabuOptimalAlgorithm 导入错误

## 核心结论

模型在 Val Unseen 上表现略优于 Val Seen（CR: 39% vs 37%），说明泛化能力良好

## 📝 代码修复记录 (2026-03-11)

为了启动 RL 训练，修复了以下代码问题：

### 配置与初始化修复（4个）
1. **constellation/rl/config.py** - 添加 `split='train'`
2. **constellation/rl/train.py** - 改为 `Environment.build(**config.environment)`
3. **constellation/environments/base.py** - 修复 `super().__init__()`
4. **constellation/task_managers.py** - 删除错误的 `super().__init__()`

### Environment 属性修复（7个）
5. **constellation/rl/environment.py** - 移除不存在的 PCompletionRateEvaluator 等
6. **constellation/rl/environment.py** - `valid_tasks` → `ongoing_tasks`（3处）
7. **constellation/rl/environment.py** - `num_valid_tasks` → `num_ongoing_tasks`
8. **constellation/rl/environment.py** - 注释 `evaluator_log_progress()`
9. **constellation/rl/environment.py** - `valid_labels` → `ongoing_flags`（2处）
10. **constellation/rl/environment.py** - `tasks` → `taskset`
11. **constellation/rl/environment.py** - 添加 `completed` 空值检查
12. **constellation/rl/environment.py** - `is_finished` → `all_closed`

### 数据文件创建（2个）
13. **data/annotations/*.tiny.json** - 从原始 JSON 提取 ids
14. **constellation/rl/config_eval_test.py** - Test 集评估配置

### 当前问题
- **constellation/rl/environment.py** - `BasiliskEnvironment` 缺少 `end_time` 属性
- **结论**: `environment.py` 代码不完整，需要参考 `controller_environment.py` 重写

## 重要文件位置

- **评估结果**: `eval_results_val_seen.txt`
- **模型检查点**: `work_dirs/test/checkpoints/iter_200000/model.pth`
- **配置文件**:
  - `constellation/rl/config_eval.py` (原始配置)
  - `constellation/rl/config_eval_quick.py` (4-GPU快速评估)
- **Basilisk**: `third_party/basilisk/` (版本 786cb285d)

## 关键命令

```bash
# 激活环境
conda activate aeos

# 单GPU评估
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
  test constellation/rl/config_eval.py \
  --load-model-from 'work_dirs/test/checkpoints/iter_200000/model.pth'

# 多GPU评估
CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 RANK=0 python -m constellation.rl.eval_all \
  test_unseen constellation/rl/config_eval_quick.py \
  --load-model-from 'work_dirs/test/checkpoints/iter_200000/model.pth'

# 检查GPU占用
nvidia-smi
```

## 注意事项

- **环境要求**：所有操作必须在 `aeos` conda 环境中执行
  - 验证命令：`which python` 应显示 `/home/hy/miniconda3/envs/aeos/bin/python`
  - 如未激活：提醒用户执行 `conda activate aeos`
- **GPU 使用**：训练前检查 GPU 占用，避免与其他用户冲突
- **依赖冲突**：Basilisk 与 todd-ai 存在 numpy 版本冲突
  - Basilisk 2.8.41 需要 numpy>=2.0
  - todd-ai 0.6.0 需要 numpy<2.0
  - 当前方案：使用 Basilisk commit c3624e0（支持 numpy 1.x）
- checkpoint 每 10,000 iter 保存一次
- 训练日志位于 `work_dirs/test/*.log`

## 数据状态

- ✅ trajectories.{1,2,3}/ 已解压
- ✅ annotations/ 已就绪
- ✅ constellations/ 已就绪
- ⚠️  data/orbits.zip 已删除（git status 显示）

## 已创建的文件

- `config_eval_val_unseen.py` - val_unseen 评估配置
- `analyze_results.py` - 结果分析脚本（含 CS 公式）
- `FINAL_EXECUTION_REPORT.md` - 执行报告
- `BASILISK_MULTIPROCESS_ISSUE.md` - Basilisk 问题详细报告
- `eval_single_process.py` - 单进程评估脚本（未成功）
- `test_multiprocess.py` - 多进程测试脚本

## 核心问题

**Basilisk 依赖冲突**：这是当前阻塞评估的主要问题
- Basilisk 最新版本与 todd-ai 的 numpy 要求不兼容
- 已尝试使用旧版本（c3624e0）但安装时遇到网络问题
- 需要解决 googletest 下载失败或找到替代安装方案

---
最后更新：2026-03-07 09:22
