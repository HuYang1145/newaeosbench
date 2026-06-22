# AEOS-Former 输入输出与张量形状流图

这个文件把原来的一张大图拆成三张小图，方便查看和编辑。

符号说明：

- `b`：一批被采样出来的时刻
- `ns`：卫星数
- `nt`：当前时刻的候选任务数
- `d`：Transformer 隐藏维度
- `dt`：时间嵌入维度
- `ds`：传感器类型嵌入维度

核心结论：

- 时间不是 Transformer 的序列轴
- Encoder 把任务当序列，序列长度是 `nt`
- Decoder 把卫星当序列，序列长度是 `ns`
- TimeModel 是卫星-任务两两配对的约束模块，不是 Transformer 序列

## 图 1：Encoder，任务序列

```mermaid
%%{init: {"theme": "base", "flowchart": {"nodeSpacing": 80, "rankSpacing": 70}, "themeVariables": {"fontFamily": "Arial", "fontSize": "21px"}} }%%
flowchart LR
    A["一条轨迹<br/>Trajectory"] --> B["抽取有效时刻<br/>得到 b 个时刻"]

    B --> T0["time_steps<br/>[b]"]
    B --> T1["tasks_sensor_type<br/>[b, nt]"]
    B --> T2["tasks_data<br/>[b, nt, 6]"]
    B --> T3["tasks_mask<br/>[b, nt]"]

    T0 --> TE["时间嵌入<br/>[b, dt]"]
    TE --> TER["复制到每个任务<br/>[b, nt, dt]"]

    T1 --> E1["任务传感器类型嵌入<br/>[b, nt, ds]"]
    T2 --> E2["任务数值特征映射<br/>[b, nt, d_task]"]

    TER --> C["拼接任务 token<br/>[b, nt, dt + ds + d_task]"]
    E1 --> C
    E2 --> C

    C --> P["输入投影<br/>[b, nt, d]"]
    T3 --> M["任务 mask<br/>控制有效任务"]
    P --> SA["任务自注意力<br/>序列长度 = nt"]
    M --> SA
    SA --> H["任务隐藏状态<br/>hidden_states<br/>[b, nt, d]"]

    classDef raw fill:#e8f1ff,stroke:#2563eb,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef time fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef embed fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef mix fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef attn fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef out fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef mask fill:#f5f3ff,stroke:#9333ea,stroke-width:2px,color:#0f172a,font-size:21px;

    class A,B,T0,T1,T2,T3 raw;
    class TE,TER time;
    class E1,E2 embed;
    class C,P mix;
    class SA attn;
    class H out;
    class M mask;
```

## 图 2：TimeModel，约束模块

```mermaid
%%{init: {"theme": "base", "flowchart": {"nodeSpacing": 80, "rankSpacing": 70}, "themeVariables": {"fontFamily": "Arial", "fontSize": "21px"}} }%%
flowchart LR
    T0["time_steps<br/>[b]"] --> T1["扩展到每个卫星-任务对<br/>[b, ns, nt]"]

    S0["constellation_data<br/>卫星特征<br/>[b, ns, 56]"] --> S1["扩展到每个任务<br/>[b, ns, nt, 56]"]
    K0["tasks_data<br/>任务特征<br/>[b, nt, 6]"] --> K1["扩展到每颗卫星<br/>[b, ns, nt, 6]"]

    SM["constellation_mask<br/>[b, ns]"] --> PM["有效配对 mask<br/>[b, ns, nt]"]
    KM["tasks_mask<br/>[b, nt]"] --> PM

    T1 --> CAT["拼接三元组特征<br/>(time, satellite, task)"]
    S1 --> CAT
    K1 --> CAT
    PM --> CAT

    CAT --> MLP["MLP 逐对预测<br/>不是 Transformer 注意力"]

    MLP --> F["feasibility_logits / time_mask<br/>[b, ns, nt]"]
    MLP --> D["pred_duration<br/>[b, ns, nt]"]

    F --> U1["给 Decoder 交叉注意力使用"]
    F --> L1["联合训练<br/>可行性二分类 loss"]
    D --> L2["联合训练<br/>持续时间回归 loss"]

    classDef raw fill:#e8f1ff,stroke:#2563eb,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef expand fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef mask fill:#f5f3ff,stroke:#9333ea,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef pair fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef mlp fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef out fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef use fill:#cffafe,stroke:#0891b2,stroke-width:2px,color:#0f172a,font-size:21px;

    class T0,S0,K0,SM,KM raw;
    class T1,S1,K1 expand;
    class PM mask;
    class CAT pair;
    class MLP mlp;
    class F,D out;
    class U1,L1,L2 use;
```

## 图 3：Decoder，卫星序列与动作输出

```mermaid
%%{init: {"theme": "base", "flowchart": {"nodeSpacing": 80, "rankSpacing": 70}, "themeVariables": {"fontFamily": "Arial", "fontSize": "21px"}} }%%
flowchart LR
    T0["time_steps<br/>[b]"] --> TE["时间嵌入<br/>[b, dt]"]
    TE --> TER["复制到每颗卫星<br/>[b, ns, dt]"]

    S1["constellation_sensor_type<br/>[b, ns]"] --> D1["卫星传感器类型嵌入<br/>[b, ns, ds]"]
    S2["constellation_sensor_enabled<br/>[b, ns]"] --> D2["开关状态嵌入<br/>[b, ns, de]"]
    S3["constellation_data<br/>[b, ns, 56]"] --> D3["卫星数值特征映射<br/>[b, ns, d_sat]"]

    TER --> C["拼接卫星 token<br/>[b, ns, dt + ds + de + d_sat]"]
    D1 --> C
    D2 --> C
    D3 --> C

    C --> P["输入投影<br/>[b, ns, d]"]
    P --> SA["卫星自注意力<br/>序列长度 = ns"]
    SA --> X["卫星隐藏状态<br/>[b, ns, d]"]

    H["来自图 1<br/>任务隐藏状态<br/>[b, nt, d]"] --> CA["交叉注意力<br/>卫星 Query<br/>任务 Key / Value"]
    F["来自图 2<br/>time_mask<br/>[b, ns, nt]"] --> CA
    X --> CA

    CA --> O["解码后状态<br/>[b, ns, d]"]
    O --> N["null_logits<br/>不选任务<br/>[b, ns]"]
    O --> G["task_logits<br/>选择任务<br/>[b, ns, nt]"]

    N --> CAT["拼接动作分数<br/>[b, ns, nt + 1]"]
    G --> CAT
    CAT --> A["动作含义<br/>0 = 空动作<br/>1..nt = 对应任务"]
    CAT --> CE["训练目标<br/>与 actions_task_id 做 CE loss"]

    classDef raw fill:#e8f1ff,stroke:#2563eb,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef time fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef embed fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef mix fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef attn fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef external fill:#cffafe,stroke:#0891b2,stroke-width:2px,color:#0f172a,font-size:21px;
    classDef out fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#0f172a,font-size:21px;

    class T0,S1,S2,S3 raw;
    class TE,TER time;
    class D1,D2,D3 embed;
    class C,P mix;
    class SA,CA attn;
    class H,F external;
    class X,O,N,G,CAT,A,CE out;
```

## 颜色说明

- 蓝色：原始输入
- 黄色：时间嵌入
- 绿色：特征嵌入或特征映射
- 紫色：拼接、投影、配对
- 橙色：注意力或 MLP 计算层
- 红色：输出或损失
- 青色：来自其他图的中间结果

## 简化理解

- 图 1：任务之间互相看，得到任务表示
- 图 2：每个卫星-任务对单独判断是否可行、能持续多久
- 图 3：卫星之间互相看，再去看任务，最后输出每颗卫星选哪个任务

所以它不是“时间序列 Transformer”。它是“任务序列 + 卫星序列 + 约束配对模块”的调度模型。
