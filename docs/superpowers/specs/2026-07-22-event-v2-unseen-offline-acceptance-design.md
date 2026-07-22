# V2-0 未见轨迹离线验收设计

## 目标

在固定的 64 条 `val_unseen` 旧轨迹上，比较以下两个模型：

1. **随机 V2 基线**：使用 seed `3407` 初始化 V2 新模块，并加载同一个
   Stage3-200k checkpoint；
2. **V2-0 10k**：加载
   `work_dirs/event_joint_transformer_v2/v2_0_warm_start/checkpoint_step_010000.pth`。

验收只回答“10k warm start 是否让 V2 新模块优于随机初始化”。它不运行 Basilisk，
不读取 Test，不评价 CR/PCR/WCR，也不据此选择或继续调节 checkpoint。

## 固定输入与公平比较

- split 固定为 `val_unseen`；annotation 固定为
  `data/annotations/val_unseen.json`，必须恰好包含 64 条轨迹；
- scene 顺序使用 annotation 原顺序；正式 `event_batch_size` 在结果无关的 GPU smoke
  中从 `{8,16,32,64,128,256,512}` 选择最大安全档位，随后锁定；
- 每个 scene 只构造一次 `OfflineEventBatch`，同一个 CPU batch 依次送入随机基线和
  10k 模型，保证 scene、事件时间点、mask 和事实标签逐值相同；
- 两个模型使用同一 V2 配置、同一 Stage3 checkpoint 和同一 loss 权重；唯一差异是
  V2 新模块采用随机初始化还是 10k checkpoint；
- 随机基线初始化前重置 Python、NumPy 和 PyTorch seed 为 `3407`；
- 10k checkpoint 必须通过 `stage=V2-0`、step `10000`、transition schema fingerprint
  和 config fingerprint 校验；不接受 `strict=False` 静默加载。

## 指标与聚合

使用现有 `event_v2_offline_loss()` 计算四个事实监督分量：

- `task_distillation`：V2 task logits 对冻结 Stage3 teacher 的 KL；
- `termination`：事实 segment 边界上的 BCE；
- `commitment`：有事实连续段标签的位置上的五档 CE；
- `value`：事件点到精确终点 Q 的 Smooth L1。

每个 scene 同时记录四类 support。跨 scene 聚合时：task、termination、commitment
分别按对应 observed label 数加权；value 按事件数加权。`total` 由四个聚合分量按正式
loss 权重重新相加，不直接平均 scene total，避免不同标签覆盖率造成偏差。

输出同时包含：

- 随机基线和 10k 的加权 loss；
- `delta = trained - random`，以及
  `relative_reduction = (random - trained) / max(random, 1e-12)`；
- 64 个 scene 的 id、事件时间点、四类 support 和两模型 loss；
- 标签总覆盖、缺失标签 scene、checkpoint/schema/config 指纹；
- 是否调用 Basilisk/Test 的固定审计字段，值必须为 `false`。

## 验收门槛

只有同时满足以下条件才标记 `accepted=true`：

1. 64 个 scene 全部处理完成，两个模型使用完全相同的事件时间点和 support；
2. 四类全局 support 均大于 0；
3. 所有 loss、delta 和相对变化均为有限数值；
4. 10k 的 `total` 以及四个分量均严格低于随机 V2 基线；
5. checkpoint、schema 和 config fingerprint 全部匹配。

该门槛只判定离线初始化是否有效。即使通过，也只能进入同步 PPO 正确性阶段，不能写成
“模型完成率提高”；如果失败，则保留完整结果并停止进入 PPO，不能改 seed、挑 scene
或改用其他 checkpoint 修饰结论。

## 实现边界

新增独立工具 `tools/evaluate_event_v2_unseen_offline.py`，不修改训练器或旧模型行为。
工具提供 `--limit` 仅用于 smoke；正式运行拒绝把少于 64 条的结果标为 accepted。
正式输出固定写入：

```text
work_dirs/event_joint_transformer_v2/v2_0_unseen_offline/summary.json
```

新增 Slurm 包装 `scripts/evaluate_event_v2_unseen_offline_slurm.sh`，使用 `aeos` 环境、
`local-10` 和独立日志。评估只做 forward，不写模型权重；GPU 不可用、显存已被占满、
checkpoint 缺失或输出已存在且未显式允许覆盖时直接失败。

## GPU 利用边界

随机基线和 10k 模型同时常驻 GPU，forward 使用 BF16 `inference_mode` 和 SDPA。
Slurm 包装先在固定 smoke scene 上按升序测试
`{8,16,32,64,128,256,512}`，记录 `max_memory_allocated`、
`max_memory_reserved` 和总显存；选择无 OOM 且峰值 reserved 不超过总显存 90% 的最大
档位。正式 64 场运行开始后不得再改变 batch size。

该探针只依据资源占用选择档位，不读取 loss 优劣，因此不会把 `val_unseen` 结果用于
调参。如果 scene 的事实事件数小于候选 batch size，工具必须记录实际事件数；不得通过
创建无用 tensor 或缓存重复模型来人为占满显存。

## 测试与运行顺序

1. 单元测试覆盖 support 加权、相对变化、严格门槛、非有限值和不足 64 scene 的拒绝；
2. tiny 模型测试覆盖随机/训练 checkpoint 严格加载及同 batch 对照；
3. CLI/Slurm 静态测试覆盖 `aeos`、`local-10`、无 Basilisk/Test 和独立输出路径；
4. 单 scene GPU smoke 并锁定最大安全 `event_batch_size`；
5. 64 条 `val_unseen` 正式运行一次；
6. 校验 JSON、日志、scene 数、support、fingerprint 与 `accepted`，再更新 `TODO.md`
   和 `改进日志.md`。
