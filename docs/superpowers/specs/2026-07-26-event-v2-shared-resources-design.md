# Event V2 大规模同步 PPO 共享资源设计

## 目标

从已安全保存的两个 V2-2-Large checkpoint 继续严格同步 PPO，同时减少 Slurm
资源占用，让同一节点上的其他用户仍可使用 CPU、内存和两张完整 GPU。

## 已确认基线

- 原 Slurm job `3296` 已在同步 barrier 停止；旧的 jobs `3306–3309` 已取消。
- seed `5408` 从 update `1420` 恢复，seed `5409` 从 update `1448` 恢复。
- 两个 `checkpoint_latest.pth` 均包含模型、optimizer、scheduler、RNG、actor 状态和
  场景进度。
- 原任务 `sstat` 实测峰值 RSS 为约 `57 GiB`，因此 `70 GiB` 仍保留约
  `13 GiB` 的内存余量。
- 当前 Slurm 仅配置整卡 `gpu` GRES，未配置 MPS；申请四张 GPU 即使只用少量显存，
  也会阻止其他 Slurm 作业获得这四张卡。

## 新资源边界

正式训练及后续 held-out、Val 和 Test 作业统一满足：

- 最多申请 `2 GPU`；
- 最多申请 `72 CPU`，即 server-10 的 144 个逻辑 CPU 的一半；
- 申请 `70 GiB` 内存；
- 单次训练作业时限为 6 小时；结束前 5 分钟通知主训练进程在同步 barrier 保存
  checkpoint，后续可从 latest 精确续训；
- 不直接在登录节点运行正式训练或评估。

## 训练映射

恢复作业继续并行训练两个 seed：

- seed `5408` 独占本作业获得的第一张 GPU；
- seed `5409` 独占本作业获得的第二张 GPU；
- 每个 seed 的 learner 和 12 个 actor 全部放在该 seed 的同一张 GPU；
- 每个 seed 仍使用 12 actors、60 个活跃环境和原 scenes `205–324`；
- reward、PPO 超参数、模型冻结范围、事件 batch、动作定义和 scene assignment 均不
  改变。

每张 GPU 的预计显存占用约 `11–12 GiB`，低于 24 GiB 卡容量；另外两张 GPU 不被本
作业申请，可由 Slurm 分配给其他用户。GPU 映射不参与训练指纹，因此不改变 checkpoint
兼容性。

## Checkpoint 与故障处理

- 从现有 `checkpoint_latest.pth` 精确恢复，不重新初始化或覆盖已完成进度。
- 每 100 updates 永久保存一个周期 checkpoint，并继续维护 latest 链接。
- 正常暂停或信号退出只在同步 barrier 保存最终 checkpoint。
- 6 小时训练作业使用 batch-shell 信号转发，只通知两个主训练进程，不直接终止 actor；
  预留 5 分钟完成 barrier checkpoint。
- 若 70 GiB 内存不足导致 Slurm 强制终止，禁止从头重训；从最近永久 checkpoint 恢复，
  同时记录实际 `MaxRSS`，再由用户决定是否增加内存。

## 后续评估链

held-out selection、Val 8+8 gate、完整 Val 和唯一一次 Test 全部改为最多两张 GPU。
原先四路并发的评估改成每轮两路并发，评估内容、场景和门槛不变：

```text
resume training
  -> held-out checkpoint selection
  -> Val 8+8 gate
  -> full Val
  -> exactly one Test
```

Slurm 继续使用 `afterok`：任何训练或验证门槛失败，后续任务不运行。

## 验证标准

修改实现前先增加失败测试，随后验证：

1. 所有相关 Slurm 脚本均不超过 `2 GPU / 72 CPU / 70 GiB`；
2. resume 脚本把两个 seed 分别映射到一张 GPU；
3. 12 actors、60 个活跃环境和训练指纹相关配置保持不变；
4. 四路评估在两张 GPU 上分两批运行，不发生 GPU 索引越界；
5. 恢复 preflight 能读取 update `1420/1448` 的 checkpoint；
6. 新作业启动后用 `scontrol`、`sstat` 和 `nvidia-smi` 核实实际资源占用。
