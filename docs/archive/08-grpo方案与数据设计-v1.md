# 8. Stage 2 GRPO 训练管线、Reward 与数据集设计草案

> 状态：**讨论与规划中（Draft v1）** · 前置基准：`SFT-v3`（`r=64 checkpoint-68` 已通过 BFCL `core` 17 类验收；`r=16` 对照实验进行中）
> 相关文档：[03-sft数据集调研](03-sft数据集调研.md) · [05-sft训练管线](05-sft训练管线.md) · [SFT-v3 Core 报告](../reports/03-sft/05-sft-v3-2b-ckpt68-core-eval-report.md)

---

## 1. 为什么要做 GRPO（以及 GRPO 只该解决什么问题）

`SFT-v3`（`ckpt-68`）已经把 `Qwen/Qwen3.5-2B` 的 BFCL `core` 全量微平均从 `59.27%` 拉升到 `65.71%`（`Non-Live AST` `60.38% -> 82.25%`、`simple_java` `+45.0 pp`），但全套 checkpoint 演化也暴露出**纯静态 SFT 交叉熵的物理边界**：
1. **单工具调用 vs 拒调的跷跷板**：SFT 第 1 轮后半程到第 2 轮（`ckpt-102..168`）能把 `irrelevance` / `live_irrelevance` 拉到 `89.2% / 85.4%`，但代价是 `simple_python` 发生 70 题零调用误拒（掉 13 个百分点）。
2. **`multi_turn_miss_func` 恢复断点**：缺少动态工具注入后的环境反馈信号，后期模型在 Turn 0 拒答后，Turn 1 工具补齐时依然直接返空停住（`1.0%`）。
3. **API-Bank 回包后正文泄漏 `<tool_call>`（`128/528`）**：在带 `tools` 描述但应纯文本收尾的中间步骤上，SFT 缺少对“误吐 `<tool_call>` 标签”的负反馈惩罚。

**GRPO 的核心任务**不是重头教模型怎么写 JSON（SFT 已经学透了），而是利用**可编程验证的 Reward（RLVR）**，围绕 `SFT-v3` 基座策略（`ckpt-68` 或对比 `r=16` 胜出者）在**“该调 vs 拒调”“多步工具恢复”“参数键值精度”“回包零泄漏”**四个决策边界上做组内优胜劣汰。

---

## 2. 算力、显存与单卡 vs 双卡 A100-40GB 权衡

### 2.1 远端现有环境底座核查
- `runs/envs/train`：`torch 2.10.0+cu129`、`transformers 5.24.0`、`trl 1.18.2`、`peft 0.22.0`（支持 `GRPOTrainer` / `GRPOConfig`，包含 `vllm_mode="server" | "colocate"`、`beta`、`loss_type`、`vllm_importance_sampling_correction`；**当前未安装 `vllm`**）。
- `runs/envs/serve`：独立隔离的 `vllm 0.19.1` 环境。

### 2.2 `Qwen3.5-2B` LoRA GRPO 显存与时间账（A100-40GB）

| 显存 / 耗时开销项 | 关键特征与风险 | 控制手段 |
| --- | --- | --- |
| **模型静态权重 + 优化器** | `2B` BF16 权重仅 ~`4.0 GB`；LoRA（`r=16/64`）+ AdamW 状态 `< 0.5 GB` | **Reference Model 零额外显存**：PEFT LoRA 下直接 `disable_adapter()` 算 ref logprobs，无需再载入第二份 2B 模型 |
| **Logits / 激活显存峰值** | `Qwen3.5` 词表高达 **248K**，GRPO 若一次性将 `(group_size × seq_len × 248K)` 转为 fp32 会急剧吃紧 | GRPO 阶段开启 **`gradient_checkpointing=true`**、控制微批次（如 `per_device_train_batch_size=1` 或 `2`）并限制单条总长 `≤ 4K–6K tokens` |
| **Rollout 生成吞吐（最大耗时项）** | 每条 prompt 采样 `G = 4 ~ 8` 条回答；原生 HF `generate` 极慢（无 Prefix Caching / Continuous Batching），且关闭思考后每条回复仅需 ~60–250 tokens | **Smoke 先用单卡验证；正式跑推荐双卡异步或 vLLM server 模式**（`enable_thinking=false`、`max_completion_tokens=512~768`） |

### 2.3 单卡 vs 双卡 A100-40GB：有没有必要上双卡？

| 维度 | 方案 A：单卡 A100-40GB（适合先跑 **GRPO Smoke**） | 方案 B：双卡 2 × A100-40GB（**强烈推荐用于正式 GRPO**） |
| --- | --- | --- |
| **推荐运行模式** | 1) 最零改动：`use_vllm=False`（HF 原生 `generate`），先跑 **200–300 条 Smoke**；<br>2) 进阶：同卡 `colocate` / 小占比 `vllm-serve`（须单独配齐含 `vllm` 的 `rl` 隔离 venv） | **解耦架构（`vllm_mode="server"`）**：<br>- **GPU 0**：独占运行 vLLM 推理服务（Continuous Batching + Prefix Caching，专注 `G=6~8` 高速采样）；<br>- **GPU 1**：独占运行 `GRPOTrainer`（算 ref logprobs + 反向更新，40GB 显存绰绰有余，步末同步权重到 GPU 0） |
| **显存裕度** | 紧凑：训练与采样 KV Cache 抢同一张 40GB 卡，长 Prompt + 大 `G` 易 OOM | **充裕**：采样与训练显存物理隔离，可轻松放大 `G=8`、`max_prompt_length=4096`、`batch` 更稳 |
| **预期速度对比** | Smoke（250 条 × `G=4` = 1,000 次生成）：HF `generate` 约 **20–40 分钟**；若 1,500 条正式跑约 **3–5 小时** | 比单卡 HF `generate` **快 3–6 倍**：1,500 条 × `G=8`（12,000 次生成）+ 更新预计 **40–75 分钟**即可完成 1 个 epoch |
| **结论建议** | **完全够做 GRPO Smoke！** 先不租双卡，直接在现有单卡把管线、Reward、KL、显存峰值跑通 | **Smoke 跑通准备全量调参 / 多轮 Rollout 时再开双卡**，性价比最高，也避免前期调 BUG 烧双卡机器钱 |

---

## 3. Reward Design 深化方案（基于 `R_final = R_format + R_correct`）

围绕你提出的核心框架：
$$\boxed{R_{\text{final}} = R_{\text{format}} + R_{\text{correct}}, \quad R_{\text{format}} \in [0, 1], \quad R_{\text{correct}} \in [-3, 3]}$$
因此总奖励区间为 **$[-3, +4]$**。为了防止 GRPO 在训练中出现**“格式对但乱拒调拿保底分”“多调用刷分”或“正负样本奖惩不对称导致塌陷”**，需要在公式落地上明确四条关键细则：

### 3.1 `R_format ∈ [0, 1]`（结构合法性 + 防泄漏门控）
按“**是否允许调用工具**”统一判定输出结构（兼容 Qwen3.5 Native FC 解析与原始输出文本检测）：
1. **若模型发出工具调用（有 `<tool_call>` 或解析出 `tool_calls`）**：
   - `tool_calls` 结构完整合法（无截断/残缺标签，能无错解析为 JSON 参数对象，且无多余 `<tool_call>` 泄漏到自然语言正文中）→ **`R_format = 1.0`**；
   - 有调用意图（出现 `<tool_call>` / `<function=`），但 XML 残缺、JSON 解析失败或参数非 dict → **`R_format = 0.0`**。
2. **若模型未调用工具（纯文本回答，如拒调 / 追问 / 文本收尾）**：
   - 正文中**完全不含** `<tool_call>`、`</tool_call>`、`<function=`、`"arguments": {` 等伪调用泄漏，且非空有意义文本 → **`R_format = 1.0`**；
   - 正文中混入伪调用文本泄漏 → **`R_format = 0.0`**（**直接打死 API-Bank 回包泄漏**）。

> [!IMPORTANT]
> **门控熔断规则（Gate Rule，待拍板）**：建议规定当 `R_format == 0` 时，**强制令 `R_correct = -3.0`（或 `-2.0`）**，不再执行内部 Name/Key/Value 拆解；否则模型一旦学会“把半残 XML 塞进正文蒙混名字”，也能蹭到部分 `R_correct` 分。

### 3.2 `R_correct ∈ [-3, 3]` —— 当样本真值「需要调用工具」时（Call 样本）
设金标调用列表为 $G = [g_1, \dots, g_m]$，模型输出调用列表为 $P = [p_1, \dots, p_k]$：

1. **未调用 / 数量不匹配前置惩罚**：
   - 若模型返回空调用（`k = 0`，发生 `zero_call` 漏调）：**直接给 `R_correct = -3.0`**（对称重罚漏调，根治 `simple_python` 零调用退化）；
   - 若工具调用个数不对（如需要并行 2 个只调 1 个，或需要 1 个却多调 `k != m`）：先通过二分图最佳匹配（按单次匹配得分最高对齐 $\min(m, k)$ 对），同时乘以数量惩罚系数 $\alpha_{\text{count}} = \frac{\min(m, k)}{\max(m, k)} \in (0, 1]$，或扣减多余/缺失调用惩罚。
2. **单对调用 $(p_i, g_j)$ 的三维分解与归一化（总分归一化到 $[-3, +3]$）**：
   对每对匹配 $(p, g)$ 拆成三部分，保证**先对工具名，再看键，再看值**（层级门控，避免选错工具却因都有 `city` 参数蹭分）：
   - **① Tool Name Matching（满分 `1.0`）**：
     - 名称精确匹配（允许 BFCL 风格 `_` 与 `.` 兼容或严格一致，按工具库对齐）→ `s_name = 1.0`；若工具名不在候选 `tools` 中（幻觉工具）→ 整个调用直接惩罚 `-3.0`；若选错其他候选工具 → `s_name = 0.0`（且后续 Key/Value 直接置 `0`）。
   - **② Parameter Key Matching（满分 `1.0`）**：
     - 仅当 `s_name == 1.0` 时计算：
       - **特例（无参函数）**：若真值 `keys(g) == ∅` 且模型 `keys(p) == ∅` → `s_key = 1.0`（**避免除以 0**）；若真值 `∅` 但模型幻觉塞参 → 按多余参数数量扣分（如 `0.0`）。
       - **一般情况**：设必填键 $K_{\text{req}}$ 必须全中（缺 1 个必填键即显著扣分或上限封顶 `0.3`），并结合**不可出现 schema 外非法键（禁止幻觉参数）**与总体键匹配（如以真值参数集做 Recall/Precision 调和平均 $F_1(K_p, K_g)$，允许合理缺省可选参数时只扣必填/多余键！）。
   - **③ Parameter Value Matching（满分 `1.0`）**：
     - 仅当 `s_name == 1.0` 且所有对应 `key` 存在时，逐参数核验类型与值（数字 `10 == 10.0`、字符串按 BFCL 规范化去首尾空格/必要大小写容差、列表/嵌套 dict 递归匹配），计算正确参数值占金标参数数比例 `s_val ∈ [0, 1]`（无参且 `s_key == 1.0` 时 `s_val = 1.0`）。
   - **映射到 $[-3, +3]$**：
     - 单次得分 $s_{\text{raw}} = (s_{\text{name}} + s_{\text{key}} + s_{\text{val}}) \in [0, 3]$（已含无参满分 `3.0`、选错名 `0.0`）；
     - 若需**零基线对称化**到 `[-3, 3]`，可线性映射：
       $$R_{\text{correct}} = 2 \cdot \left(\alpha_{\text{count}} \cdot \bar{s}_{\text{raw}}\right) - 3 \quad \in [-3, +3]$$
       验证一下边界点：**全对 = `+3.0`**，**工具名对 + 键对 + 值全错（$2/3$）= `+1.0`**，**只选对工具名（$1/3$）= `-1.0`**，**选错工具或零调用（$0$）= `-3.0`**！物理含义极其干净直觉！

### 3.3 `R_correct ∈ [-3, 3]` —— 当样本真值「不应该调用工具」时（Non-Call 样本）
> **极其关键的设计补全**：如果不把 `irrelevance` / `missing_param` / `post-tool response` 也严格映射到 **`[-3, +3]`**，奖励中心一偏移，GRPO 就会瞬间坍缩成“全调”或“全拒调”。

| Non-Call 子类型 | 模型行为：未调用（`tool_calls == []` 且格式无泄漏） | 模型行为：误调任何工具（`len(tool_calls) > 0`） | 追问/收尾语义附加检查（可选轻量项） |
| --- | --- | --- | --- |
| **`irrelevance`（工具不适用）** | **`+3.0`**（若可附加轻量惩罚：回答不得盲目编造实时事实） | **`-3.0`**（硬惩罚误调） | 防止模型退化成“只输出一个句号”，可加最小有效字数检查（如 `< 5 tokens` 扣 `0.5`） |
| **`missing_param`（缺必填参数需追问）** | **`+2.5 ~ +3.0`**（若回答中点出了缺失的参数名/语义，给 `+3.0`；未点出仅泛泛拒绝给 `+2.0`） | **`-3.0`**（禁止不问就乱编默认值硬调） | 逼迫模型学会精确指明缺哪个参数，而不只是敷衍说“我做不到” |
| **`post-tool response`（回包后文本收尾）** | **`+2.5 ~ +3.0`**（若包含工具返回结果中的关键数值/实体给 `+3.0`；纯无意义敷衍给 `+1.5`） | **`-3.0`**（严厉惩罚重复调用或文本泄漏） | 直接压掉 API-Bank 类回包继续吐工具的行为 |

---

## 4. GRPO 数据集来源、清洗与规模规划

### 4.1 核心选材原则：**「难度筛选 + 对抗配对」，绝不照搬 `sft_v3` 简单题**
- **数据来源（全部源自本地已缓存干净源 + 可控构造，严禁混入 BFCL / API-Bank / τ² 测试题）**：
  1. `xlam-60k`（通过严格类型/去重清洗后，排除 `sft_v3` 已用的 1,450 条训练与验证题，尚余 ~**4.8 万**条可用干净池）；
  2. `toolace`（通过严格 schema 清洗后，排除 `sft_v3` 已用的 2,250 条，尚余 ~**7,800** 条可用池，含多轮、并行、缺参、拒调）；
  3. `apigen-mt-5k`（排除 `τ²` 测试任务实体与 `sft_v3` 已用题后，尚余 ~**1,900** 条轨迹可用）；
  4. `smoltalk`（白名单干净子集 `openhermes / smol-constraints / 限额 metamathqa`）；
  5. **对 `ckpt-68` 做 Pass@8 在线难度筛查**：对候选 Call / Non-Call 题用 `ckpt-68` 温度 `0.7` 采样 8 次，**剔除 8/8 全对的过简单题和 0/8 全错的脏题，优先保留 `1/8 ~ 6/8` 的高信息密度“边缘题”**（GRPO 组内方差最大，梯度最有效）。

### 4.2 两阶段规模与配比方案（Smoke 300 条 vs 正式 1,500 条）

| 能力桶 | GRPO Smoke (`300` 条) | GRPO v1 (`1,500` 条) | 数据构成与构造做法 | 直接瞄准的指标 |
| --- | ---: | ---: | --- | --- |
| **1. 单轮难例 & 1-Tool 正例** | 75 | **380** | 从残余 `xlam`/`toolace` 筛出 `ckpt-68` 易错或带可选/数字/字符串约束的样本；**保证其中一半以上为 `len(tools)==1` 正例** | 防止强化拒调时把 `simple_python` / `live_simple` 拉垮 |
| **2. 成对对抗边界（Call vs Irrelevance）** | 80 | **400** | **同源成对构造（200 对 = 400 条）**：每对用同一用户问法，一条配真工具（判 `+3` 必调），另一条删去/替换为“名字相似但功能不符或万能 HTTP 工具（`requests.get` / `math`）”作为对抗负例（判 `+3` 必拒） | 直接打破 `irrelevance` vs `simple_python` 跷跷板 |
| **3. 缺参追问 & 缺工具恢复** | 75 | **400** | ① **缺参追问**（200）：隐藏输入中 1 个 `required` 字段，判不调且指明缺参；<br>② **缺工具两段恢复**（200，重点！）：若先走单步 Slice，构造「Turn 0 缺失不可调；Turn 1 注入该工具后历史保留前轮拒绝、要求本步立刻调出该工具」的恢复切片（或直接做 2-turn Rollout） | **精准拉起 `multi_turn_miss_func`（1% -> 目标 ≥10%+）** 与 `miss_param` |
| **4. 多轮中间续调 & 回包收尾** | 70 | **320** | 从 `apigen-mt` 与 `toolace` 多轮池切分真实中间状态：<br>① 收到部分 tool 结果后的**下一步调用**（判步骤 AST，160）；<br>② **工具执行完后的纯文本收尾**（保留完整 `<tools>` 定义，判 `tool_calls==[]` + 无 `<tool_call>` 泄漏 + 包含关键返回值，160） | 巩固 `multi_turn_base` 并把 API-Bank `<tool_call>` 泄漏（128/528）打下去 |
| **总计** | **300** | **1,500** | 全部配备确定性规则校验器，零外部 LLM Judge 依赖 | 300 条单卡可快速闭环；1,500 条推荐双卡快速迭代 |

---

## 5. 待讨论与拍板的 7 个关键决策问题

为了后续一步写对代码、少走弯路，以下 7 个工程与算法设计点需要我们一起确认：

1. **硬件与执行策略选择**：
   - 先用现有单张 A100-40GB 跑通 **300 条 GRPO Smoke**（先验证显存峰值、Reward 分布、KL 曲线与无坍缩），还是直接开通 **双卡 2 × A100-40GB** 搭建 `GPU 0 vLLM Server + GPU 1 Trainer` 架构？（以及是否批准新建独立的 `runs/envs/rl` 隔离环境避免污染 `train` / `serve` 现有环境？）
2. **GRPO 起始基座权重选哪一个？**
   - 等当前正在跑的 **`sft-v3 r=16`** 结果出来后，与 **`r=64 checkpoint-68`** 对比（重点看谁在 `simple_python`、`irrelevance`、`multi_turn` 上更健康、保留了更多底座多样性），再择优作为 GRPO 起点（以及思考是否需要把 SFT LoRA **merge 进底座后再在上面挂新 LoRA 跑 GRPO**，还是直接在原 LoRA adapter 上续训）。
3. **Format 门控的严厉程度**：
   - 当 `R_format == 0`（JSON 解析失败或正文泄漏 `<tool_call>`）时，是否直接锁死 `R_correct = -3.0`（硬熔断），还是给 `0.0`（保留 `-3.0` 与 `0.0` 梯度差）？
4. **Call vs Non-Call 的惩罚对称性与“逃逸保护”**：
   - 如果一个难 Call 题大部分采样只拿到 `-1.0`（仅选对名字），而模型如果偷懒不调用（直接拒答）拿 `-3.0`，确实不会故意偷懒；但反过来对于**多轮 `miss_func` 恢复**和 **`simple_python`**，是否需要对“应调未调（`zero_call`）”维持 `-3.0` 强负分，同时对“选错工具名”给 `-3.0` 还是 `-2.0`？
5. **Parameter Value Matching 的容差口径（防过严导致 Reward 稀疏）**：
   - 值匹配是完全照搬 **BFCL AST 规范化等价**（去空格/大小写容差/`10 == 10.0`/允许省略 optional 默认参数），还是要求严格字面匹配？*（强烈建议复用或对齐 BFCL 的类型与可选参数宽松规则，否则合法合理的多样化传参也会被罚分。）*
6. **Multi-turn 的训练形态**：
   - **Phase 1（GRPO Smoke / v1）**：先用**静态多轮切片**（Prompt 里包含前序 `user/assistant/tool` 历史，模型只生成当前一步并单步算 Reward）——实现极快、与现有 `TRL GRPOTrainer` 原生兼容；
   - **Phase 2（如果单步切片对 `multi_turn_base` / `miss_func` 提升还不够）**：再升级成真正的**多轮环境闭环 Rollout**（`model -> sandbox/mock tool -> obs -> model` 多步轨迹收集 + assistant token mask）。
7. **关键 RL 超参起点设定**：
   - 组采样数 `num_generations (G)` 选 `4`（Smoke）→ `8`（双卡）；
   - KL 惩罚系数 `beta`：建议小起步 **`0.01 ~ 0.04`**（防政策偏离 SFT 太远导致语言能力或跨语言 `simple_java` 掉分）；
   - 损失形态 `loss_type`：用 TRL 原生支持的 **`dr_grpo` 或 `dapo`**（消除标准 GRPO 对长度偏置，避免模型为了蹭 format 分写长废话或过短敷衍）。
