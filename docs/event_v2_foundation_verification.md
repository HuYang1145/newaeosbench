# Transformer V2 Foundation 验证报告

## 结论

V2-0 foundation 已实现并通过代码正确性与单步真实数据训练前检。它现在具备：

- Stage3-200k 表征热启动；
- 显式事件历史、承诺、owner 和时间状态；
- termination、任务、minimum commitment 的自回归联合 Actor；
- 不使用 simulator privileged state 的 centralized value Critic；
- 精确 `Q=0.6CR+0.2PCR+0.2WCR` terminal correction；
- 可重放的 action order、mask、owner state 和联合 log-prob；
- V2-0 旧轨迹事件数据、离线 loss、恢复 checkpoint 和 Slurm 入口。

该结论只说明 V2-0 代码与单步训练链路可用，不说明完成率已经提高。正式 10k GPU
warm start、同步 PPO、Basilisk 3,600 秒 smoke、Val 和 Test 均未运行。

## Git 恢复点

```text
工作分支：codex/event-joint-transformer-v2
修改前基线：be61acf0a3fbc9b5b92d42acbe3a8e785f28bbf7
V2-0 runner 提交：799eb06
```

foundation 由职责单一的小提交组成：

```text
61f2147 state schema
8dda015 completion reward / GAE
b1ebdc6 transition schema
85f9e1b Stage3 token backbone
49425f3 event state encoder / Critic
72e214a autoregressive Actor
fca09f3 Actor-Critic assembly
faa4bf0 exact PCR terminal correction
98bc1a0 offline event data / loss
799eb06 warm-start trainer / Slurm
```

## 关键实现边界

### 无 Basilisk 热路径

`EventJointActorCritic.act/evaluate_actions` 只接收现有 Stage3 tensor、
`EventStateTensors` 和 mask。函数签名不含 `is_visible`、未来状态或 Basilisk 对象。
`is_visible` 也不在 `OfflineEventBatch` 中。

### Stage3 冻结与分阶段解冻

Stage3 `_encoder/_decoder/_time_model` 参数可严格载入并冻结；V2 新建 edge projection、
事件状态层、Actor 和 Critic 保持可训练。`unfreeze_last_layers(1, 1)` 只打开 Encoder /
Decoder 最后一层和最终 LayerNorm，输入 embedding 仍冻结。

真实 Stage3-200k checkpoint 早于独立 `_duration_head`，只缺该 head 的 weight/bias。
加载器对白名单中的这两个 legacy 缺键保留零初始化，其他 Stage3 缺键或未知
`_transformer.*` 键继续报错。

### 精确 completion reward

仓库 `CompletionRateEvaluator` 的 `PCR` 是全部任务终点进度比例均值，未完成任务也可能
贡献 PCR。V2 使用：

```text
Q_final = 0.6*mean(completed)
        + 0.2*mean(progress/required_duration)
        + 0.2*sum(duration*completed)/sum(duration)
```

potential 的 dense 代理权重为：

```text
omega_i = 0.8/N + 0.2*duration_i/sum(duration)
```

终点加入 `Q_final-Phi_terminal`，单元测试证明整条 event reward 精确 telescoping 到
Evaluator 的 Q。未完成任务的 CR/WCR 代理被收回，但真实 PCR 部分进度保留。

### 旧专家重复 owner

旧轨迹中确实存在超过 3 个 owner 的状态。V2-0 读取时把计数饱和为 3，只保留“容量已
满”的输入语义；owner marginal head 不读取这些重复 owner 作为正标签。第四个新 owner
在 V2 Actor 中始终被物理 mask。

## 自动测试

命令：

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_state.py \
  tests/test_event_v2_reward.py \
  tests/test_event_v2_transition.py \
  tests/test_event_v2_backbone.py \
  tests/test_event_v2_critic.py \
  tests/test_event_v2_actor.py \
  tests/test_event_v2_model.py \
  tests/test_event_v2_dataset.py \
  tests/test_event_v2_offline.py \
  tests/test_event_v2_warm_start.py \
  tests/test_event_v2_scripts.py \
  tests/test_event_action.py \
  tests/test_event_policy.py \
  tests/test_temporal_model.py \
  tests/test_bipartite_assignment.py -q
```

结果：

```text
119 passed, 0 failed
```

同时通过：

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m compileall -q \
  constellation/new_transformers/event_v2 \
  tools/train_event_v2_warm_start.py
git diff --check
```

## 真实轨迹数据 smoke

使用 `train_paper_stage3_tau_e_existing.json` 第一个轨迹、内部事件 batch 4：

```text
dataset length: 13,849
event time steps: [1624, 2181, 2451, 2635]
satellite tensor: (4, 42, 56)
task tensor: (4, 90, 6)
max owner after saturation: 3
replan labels: 10
termination observed labels: 75
commitment observed labels: 8
value return range: [0.6417304, 0.6468039]
```

该 smoke 只读取已有轨迹，没有运行 Basilisk，也没有生成反事实分支。

## 单步 CPU preflight

命令：

```bash
/home/hy/miniconda3/envs/aeos/bin/python \
  tools/train_event_v2_warm_start.py \
  --config constellation/new_transformers/config_event_v2_warm_start.py \
  --stage3-checkpoint \
    work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth \
  --output /tmp/event_v2_warm_start_preflight \
  --max-steps 1 \
  --device cpu
```

结果：

```text
total parameters: 93,056,272
trainable parameters: 1,674,507
loss: 0.66654396
gradient norm before clipping: 9.08401108
events: 1
physical seconds: 1
checkpoint: /tmp/event_v2_warm_start_preflight/checkpoint_step_000001.pth
checkpoint size: about 367 MiB
```

loss、梯度和 checkpoint 均有限且可恢复。这个随机单事件样本没有 observed task /
commitment target，因此该步对应的 task distillation 和 commitment loss 为 0；相关 heads
的非零监督与反向传播由人工 batch 单元测试覆盖。正式 V2-0 使用 event batch 8，不根据
本次单步 loss 选择模型。

## 下一阶段入口

正式 V2-0 Slurm 包装：

```text
scripts/train_event_v2_warm_start_slurm.sh
```

它先在已申请 GPU 内运行单步 preflight，再启动 10k warm start；时间上限 4 小时，
输出独立写入 `work_dirs/event_joint_transformer_v2/v2_0_warm_start`，不覆盖 Stage3、
M2/M3 或正式评估目录。

V2-0 完成后仍需先做未见轨迹离线验收，再另写同步 PPO/Event Runtime 实施计划。
