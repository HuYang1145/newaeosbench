# M2 事件监督设计

## 目标

在不修改 Stage3 任务排序、不在线运行候选 Basilisk、也不进入 PPO 的前提下，
使用现有专家轨迹训练事件式 Actor 所需的三个能力：

1. 判断当前非空任务下一秒应继续还是终止；
2. 在 `1/5/15/30/60 s` 中选择保守承诺时长；
3. 预测短窗口内的可见、进度和完成事实结果。

M2 的交付标准首先是“标签、模型、训练和推理路径闭环”，不是提前宣称正式指标
已经提升。

## 已有基础与选择

现有 Temporal Adapter 已经提供：

- 因果历史：上一全局任务、当前候选映射、连续执行时长和 30/60 秒切换次数；
- 冻结 Stage3 主干的训练方式；
- 卫星—任务边隐藏特征；
- 带 censor mask 的可见、进度、完成与事件时间结果头。

旧 Temporal Adapter 失败的主要实验形态是使用 residual 直接改写 Actor logits。
M2 不重复该做法。比较过三种实现后采用：

```text
复用因果历史和边特征
→ temporal_residual_scale = 0
→ Stage3 Actor logits 精确保持不变
→ 新增 continue 与 duration heads
→ 既有 outcome heads 只作短窗口事实辅助监督
→ M1 runtime 消费预测的承诺时长
```

不采用完全独立的离线 MLP，因为它会复制一套与正式 Actor 不一致的特征提取路径；
也不继续训练 residual，因为 8+8 Val 已证明该动作改写方式不稳定。

## 标签定义

训练决策在时间 `t`，模型输入只允许使用保存到 `t-1` 的卫星状态、任务状态和
`actions[:t]` 的历史，标签来自 `t` 之后。

### 终止标签

只监督真实执行的非空边：

```text
continue_target[t, s] = actions[t + 1, s] == actions[t, s]
```

这是专家行为标签，不伪装成反事实最优标签。它用于学习专家的持续性边界。

### 持续时间标签

计算从 `t` 开始同一任务的剩余连续长度，并向下映射到不越过专家切换点的最大承诺：

```text
1..4  -> 1 s
5..14 -> 5 s
15..29 -> 15 s
30..59 -> 30 s
>=60 -> 60 s
```

若连续段延伸到轨迹结尾且不足 60 秒，则精确结束时间未知，duration loss 对该位置
使用 censor mask；只要已观察到至少 60 秒，`60 s` 标签仍然有效。

idle 不进入 duration loss，正式推理第一轮仍固定为 1 秒，避免 M1 已验证的漏任务
问题。

### 事实结果标签

复用 `build_batched_edge_outcomes()`，窗口改为 `5/15/30/60 s`：

- 是否至少一次可见；
- 是否产生任务进度；
- 是否完成任务；
- 首次事件时间；
- 提前切换且未观察到事件的窗口保持 censored，不当作负样本。

功耗不单独训练回归头。对非空任务，`PowerUsageEvaluator` 的传感器功耗由
`sensor.power × commitment_seconds` 确定，推理和评估时直接计算，避免让模型学习
一个本来可以精确得到的量。

## 模型与损失

在 `TemporalAdapter.edge_hidden` 后增加：

- `continue_head: hidden -> 1`；
- `duration_head: hidden -> 5`。

既有 `outcome_head` 继续输出事实结果。训练总损失为：

```text
L_M2 =
  w_continue * BCE_continue
  + w_duration * CE_duration
  + w_visible * L_visible
  + w_progress * L_progress
  + w_completion * L_completion
  + w_event_time * L_event_time
```

Stage3 Encoder、Decoder、TimeModel 和原任务 logits 全部冻结；
`assignment_loss_weight=0`，`temporal_residual_scale=0`。因此即使 M2 新头尚未训练，
任务选择也与 Stage3 baseline 精确一致。

## 训练与推理阶段

### M2-A：标签和离线可训练性

1. 用少量真实轨迹审计 continue/duration 类别、censor 比例和短窗口结果覆盖；
2. 完成单 batch forward/backward/optimizer step；
3. 确认只有 Temporal Adapter 参数变化，Stage3 主干逐张量不变。

### M2-B：冻结主干训练

使用 Stage3-200k checkpoint 和
`train_paper_stage3_tau_e_existing.json`。正式训练通过 Slurm 运行，先保存
`1k/2k/5k/10k` checkpoint，不使用 Test。

### M2-C：事件推理

Actor 仍先选择任务；事件头只为已经选中的非空任务决定承诺：

```text
continue probability < threshold -> 1 s
otherwise -> argmax duration bucket
```

第一轮 outcome 头只用于记录校准，不用未经验证的硬阈值覆盖任务动作。后续只有在
离线校准可靠时，才增加 outcome-gated duration。

## 验收与停止条件

进入真实 M2 smoke 前必须满足：

- 标签严格因果对齐；
- duration 末尾 censor 正确；
- 关闭或零 residual 时 Stage3 logits 精确兼容；
- 冻结参数无梯度且 optimizer step 后不变化；
- 新头 loss 有限，真实轨迹单 batch 可反向传播。

单场 smoke 必须完整报告：

- `CR/PCR/WCR/TAT_s/PC_Wh/CS_paper`；
- 任务一秒承诺率、各时长分布、任务失效中断；
- continue 概率和 duration 分布；
- 全局模型调用与卫星重规划次数。

若 `PC_Wh` 再次大幅上升、`CS_paper` 恶化，或头只会预测单一时长，则停止扩大
Val，回到标签平衡与校准，不直接进入 M3/PPO。

## 非目标

- 不生成 stay/switch 反事实标签；
- 不修改任务选择 logits；
- 不学习多秒 idle；
- 不在线调用 Basilisk、轨道传播或重型几何预测；
- 不运行 Test；
- 不在 M2-A 阶段启动 PPO/APPO。
