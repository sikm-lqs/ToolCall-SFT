# 8. Stage 2 GRPO 训练管线、Reward 与数据集设计（v2）

> 状态：**v2 方案迭代中** · 历史归档：[v1 初版草案](archive/08-grpo方案与数据设计-v1.md)
> 前置基准：`SFT-v3`（`r=64 checkpoint-68` 已通过 BFCL `core` 17 类验收；`r=16` 对比实验进行中）
> 相关文档：[00-服务器环境安装](00-服务器环境安装.md) · [03-sft数据集调研](03-sft数据集调研.md) · [SFT-v3 Core 报告](../reports/03-sft/05-sft-v3-2b-ckpt68-core-eval-report.md)

### v1 -> v2 本次迭代更新要点
1. **明确双卡运行路线与新建隔离环境清单**：补齐 `runs/envs/rl`（或 `runs/envs/rl-serve` + 复用 `runs/envs/train`）需要安装的具体包、国内镜像命令与安装耗时评估；
2. **Reward Design 正式采纳 ToolRL 架构**：采用多重集合 Jaccard（`r_name`）+ 一对一参数键 Jaccard（`r_key`）+ 值匹配（`r_value`）+ `S_max` 线性映射到 `[-3, 3]`，适配 Qwen3.5 关思考模式、默认关闭长度奖励、取消 `R_format==0` 对 `R_correct` 的硬锁死，并补充 Non-Call 边界防零梯度检查；
3. **敲定四项关键工程选型**：值匹配对齐 BFCL 宽松规则、Multi-turn Phase 1 采用静态多轮切片、`G = 4`、`loss_type = "grpo"` + 小 `beta`；
4. **保留待决策项清单**：缩小到起始权重选型、Non-Call `R_correct` 映射（ToolRL 原版 `0` vs 对称扩展）、起始学习率/步数与 SFT merge 策略等 5 项核心拍板点。

---

## 1. 为什么要做 GRPO（以及 GRPO 只该解决什么问题）

`SFT-v3`（`r=64 checkpoint-68`）已将 `Qwen/Qwen3.5-2B` 的 BFCL `core` 全量微平均从 `59.27%` 提升到 `65.71%`（`Non-Live AST` `60.38% -> 82.25%`、`simple_java` `+45.0 pp`），但也明确暴露出静态 SFT 交叉熵的三类边界问题：
1. **单工具调用 vs 拒调的跷跷板**：SFT 后半程（`ckpt-102..168`）能把 `irrelevance` / `live_irrelevance` 拉到 `89.2% / 85.4%`，但代价是 `simple_python` 发生 70 题零调用误拒（掉 13 个百分点）。
2. **`multi_turn_miss_func` 恢复断点**：缺少动态工具注入后的反馈信号，后期模型在 Turn 0 拒答后，Turn 1 工具补齐时直接返空停住（`1.0%`）。
3. **API-Bank 回包后正文泄漏 `<tool_call>`（`128/528`）**：在挂载 `tools` 但应纯文本收尾的步骤上，缺少对误吐 `<tool_call>` 标签的负反馈。

**GRPO 的定位**：利用可编程验证的 Reward（RLVR），在已选定的 SFT 策略基础上，重点校准**“该调 vs 拒调”“缺工具补齐后立刻调用”“参数键值精确匹配”“回包零标签泄漏”**四个决策边界。

---

## 2. 双卡 2 × A100-40GB 硬件架构与新建隔离环境评估

### 2.1 双卡核心收益与拓扑设计
核查当前单卡环境状态：`runs/envs/train` 为 `torch==2.10.0+cu129`、`transformers==5.24.0`、`trl==1.18.2`、`peft==0.22.0`（支持 `GRPOTrainer` / `GRPOConfig` 及 `vllm_mode="server"`，**未装 `vllm`**）；`runs/envs/serve` 为 `torch==2.10.0+cu129`、`transformers==5.24.0`、`vllm==0.19.1`（**未装 `trl`/`peft`**）。

在 **2 × A100-40GB** 上，建议直接采用 **TRL `vllm_mode="server"` 物理隔离双卡拓扑**（无论 Smoke 还是正式 GRPO 均共用同一拓扑，免去后期改架构）：
- **GPU 0（Rollout 推理卡）**：运行 `trl vllm-serve`，以 Continuous Batching + Prefix Caching 专门负责 `G=4` 批量采样生成；每次训练步结束后接收权重增量同步，不跑反向传播、不存优化器状态。
- **GPU 1（训练更新卡）**：运行 `GRPOTrainer`，专心计算 Reference Logprobs（PEFT 下直接 `disable_adapter()` 复用底座，**零第二份 2B 权重拷贝**）和 Policy 前向/反向更新。
- **速度与显存收益**：
  1. **告别单卡显存打架**：40GB 各自独立，GPU 0 KV Cache 充裕，GPU 1 可从容应对 `Qwen3.5` **248K 词表**带来的 logits 显存峰值（建议仍保留 `gradient_checkpointing=true`，控制总长 `≤ 4K–6K tokens`）；
  2. **速度大幅提升**：相比单卡原生 HF `generate`，`300` 条 Smoke × `G=4`（1,200 次生成）可压至 **~8–15 分钟**；`1,500` 条正式 GRPO × `G=4`（6,000 次生成）约 **25–45 分钟** / epoch，试错成本极低。

### 2.2 新建隔离环境需要安装哪些依赖？耗时多久？

`trl vllm-serve` 需要在**推理端环境同时存在 `vllm` 与 `trl` 及其轻量化服务依赖（如 `fastapi`、`uvicorn` 等）**。为了不破坏已经验证稳定的 `runs/envs/train`（SFT）与 `runs/envs/serve`（评测），建议新建独立 RL 隔离环境。共有两种建法：

| 方案 | 做法与需要安装的新包 | 磁盘额外占用 | 预计耗时（国内 TUNA + 上交大 `cu129`） | 适用场景 |
| --- | --- | ---: | ---: | --- |
| **方案 E1（极简增量建双端，最省时）** | 1. **GPU 1 训练端直接复用 `runs/envs/train`**（已具备 `torch 2.10+cu129 / transformers 5.24 / trl 1.18.2 / peft 0.22 / fla`，仅检查 `requests`/`httpx`）；<br>2. **GPU 0 推理端新建 `runs/envs/rl-serve`**：安装 `torch==2.10.0+cu129`、`vllm==0.19.1`、`transformers==5.24.0`、`trl==1.18.2`、`fastapi`、`uvicorn` | ~`5–6 GB` | **约 4–8 分钟**<br>*(注：当前 `/root/.cache/uv` 已清理仅剩 `42 MB`，需重新拉取一次 `cu129` torch + vllm wheel)* | 专注 `vllm_mode="server"`（双卡解耦架构） |
| **方案 E2（新建统一全功能 `runs/envs/rl`）** | 一次性在一个全新 venv 里对齐锁定：<br>`torch==2.10.0+cu129` + `vllm==0.19.1` + `transformers==5.24.0` + `trl==1.18.2` + `peft>=0.22` + `accelerate>=1.0` + `datasets>=4.0` + `flash-linear-attention` + `swanlab` + `safetensors` + `pyyaml` + `fastapi` + `uvicorn` | ~`6–8 GB` | **约 5–10 分钟**<br>*(其中下载约 3.5–4 GB wheel 占 4–8 分钟，uv 解压安装约 1–2 分钟)* | 一套 `runs/envs/rl` 通吃服务端和客户端（未来甚至可切 `colocate`） |

> **工程提醒**：新建 `runs/envs/rl` 前务必先验证 `trl==1.18.2` 与 `vllm==0.19.1` 在 `transformers==5.24.0` 下的依赖冲突情况（先 `uv pip install --dry-run` 预解算）；另外沿用 `SFT-v3 r=16` 踩过的坑，启动 `tmux` 时显式注入 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HUB_OFFLINE=1`（针对本地已有模型/kernel），避免联网检查卡死。

---

## 3. Reward Design 详细方案（仿 ToolRL + 适配 Qwen3.5 无思考模式）

### 3.1 总公式（默认关闭长度奖励）

$$R_{\text{final}} = R_{\text{format}} + R_{\text{correct}}$$

其中 $R_{\text{format}} \in \{0, 1\}$，$R_{\text{correct}} \in [-3, 3]$，总奖励范围为 **$[-3, 4]$**；长度奖励 $R_{\text{length}}$ 默认关闭（置 `0`）。

### 3.2 格式奖励 `R_format ∈ {0, 1}`（关思考版）

由于本项目统一采用 `enable_thinking=false`，**不再检查 `<think>` 推理内容与词数**，允许官方模板自带的空 `<think>\n\n</think>\n\n` 前缀（或零思考前缀），重点校验**结构、标签闭环与无标签泄漏**：

1. **当模型输出发生工具调用时**（输出含 `<tool_call>` 或结构化 `tool_calls` 非空）：
   - `<tool_call>...</tool_call>` 标签完整成对闭合，能无异常解析出合法函数名与 JSON/参数字典（`arguments` 必须为 `dict`）；
   - 不得出现残缺标签、重复未闭合标签或额外乱码尾缀。
   - 满足上述条件则 `R_format = 1`，否则 `R_format = 0`。
2. **当模型输出为纯文本（无工具调用）时**：
   - 正文非空、语义完整，且**严禁包含任何伪调用标签泄漏**（如 `<tool_call>`、`</tool_call>`、`<function=`、`<parameter=`、`"arguments": {`）；
   - 满足上述条件则 `R_format = 1`，一旦正文泄漏伪调用标签则 **`R_format = 0`**。

> **与 v1 草案的区别**：采纳你的意见，当 `R_format == 0` 时**不**直接把 `R_correct` 一刀切锁死为 `-3.0`；仅在「真值需要调用工具，但模型输出无法解析出任何合法调用」时，按下面 ToolRL 特殊情况置 `R_correct = -3.0`。

### 3.3 正确性奖励 `R_correct ∈ [-3, 3]`（Call 样本 · 仿 ToolRL 数学化表达）

设标准答案调用列表为 $G = [g_1, \dots, g_{|G|}]$，模型解析出的预测调用列表为 $P = [p_1, \dots, p_{|P|}]$：

1. **Tool Name Reward（多重集合 Jaccard Similarity）**：
   设 $N_G$ 和 $N_P$ 分别为 $G$ 和 $P$ 的工具名称**多重集合**（考虑同名工具并行调多次的次数，如调 2 次 `get_weather`）：
   $$r_{\text{name}} = \frac{|N_G \cap N_P|}{|N_G \cup N_P|} \in [0, 1]$$
2. **一对一匹配关系 $M$**：
   在同名工具之间建立最优一对一匹配 $M = \{(i, j)\}$（若同一工具出现多次，采用**二分图最大权重匹配 / 贪心最优配对**，优先把键/值重合度最高的预测与真值配对，避免乱序并行调用被错配）。
3. **Parameter Key Reward**：
   $$r_{\text{key}} = \sum_{(i,j) \in M} \frac{|K_i^G \cap K_j^P|}{|K_i^G \cup K_j^P|}$$
   > **工程除零与可选参数补丁**：
   > - **无参函数特例**：当标准答案与预测均无参数（$K_i^G = K_j^P = \emptyset$）时，规定该项 Jaccard 为 **`1.0`**（防止除零，且无参调对可得满分）；
   > - **可选参数对齐 BFCL（Decision 5）**：若真值中某可选参数有默认值（如 `unit=""` 或允许省略）且模型合理省略，不按缺键误杀；但模型多传 schema 中不存在的幻觉键名仍计入分母 $|K_i^G \cup K_j^P|$ 扣分。
4. **Parameter Value Reward（对齐 BFCL 宽松等价规则）**：
   $$r_{\text{value}} = \sum_{(i,j) \in M} \sum_{k \in K_i^G} \mathbb{1}_{\text{BFCL}}\big[P_j(k) = G_i(k)\big]$$
   其中 $\mathbb{1}_{\text{BFCL}}[\cdot]$ 复用/对齐 BFCL 类型与规范化等价规则：整数与等价浮点对齐（`10 == 10.0`）、字符串去除冗余首尾空白与特定标点/大小写容差（按参数类型）、嵌套列表/字典递归比对、真值允许省略的 optional 默认参数不重复扣分。
5. **最大理论分数与 $[-3, 3]$ 线性映射**：
   $$S_{\max} = 1 + |G| + \sum_{i=1}^{|G|} |K_i^G|$$
   $$R_{\text{correct}} = 6 \cdot \frac{r_{\text{name}} + r_{\text{key}} + r_{\text{value}}}{S_{\max}} - 3 \quad \in [-3, 3]$$
6. **ToolRL 特殊情况**：
   - 需要调用工具（$|G| > 0$），但预测结果无法解析或空调用（$|P| = 0$）→ **`R_correct = -3`**；
   - 工具调用完全正确（$r_{\text{name}}=1$，所有对齐项键值全对且无多余调用）→ **`R_correct = +3`**。

### 3.4 关键设计补全：当 Ground Truth **不需要调用工具**时（Non-Call 样本）

> [!WARNING]
> **必须防住 ToolRL 原文写法的「零梯度陷阱」**：ToolRL 原文写了 *"Ground Truth 不需要调用工具：$R_{\text{correct}} = 0$"*。如果 **无论模型调没调都给 $R_{\text{correct}} = 0$**，且模型误调了一个格式合法的工具拿到 $R_{\text{format}} = 1$，那么「误调」总分 $= 1 + 0 = +1$，「正确拒调」总分也是 $1 + 0 = +1$——**两类行为无分差，GRPO 在 `irrelevance` / `missing_param` / `post-tool response` 上将完全学不到任何拒调信号！**

因此，针对 Non-Call 样本（`irrelevance`、`missing_param`、`post-tool response`）给出以下 **两个候选细则**（均保证“正确不调”严格高于“误调”）：

| 子方案 | GT 不需要调用、且模型**正确不调用** | GT 不需要调用、但模型**误调了工具** | GT 不需要调用、模型正文**泄漏 `<tool_call>`** | 特点与取舍 |
| --- | --- | --- | --- | --- |
| **3.4-A（忠实 ToolRL + 格式防线）** | `R_format = 1`, `R_correct = 0` → **总分 `+1`** | 在 Non-Call 任务中把「发出工具调用」视为**违反指令格式**：`R_format = 0`, `R_correct = 0` → **总分 `0`** | `R_format = 0`, `R_correct = 0` → **总分 `0`** | 最贴近 ToolRL 原版定义，正负分差为 `1.0`；风险是相比 Call 样本 `+4 vs -3`（分差 `7.0`），Non-Call 的优势信号偏弱，可能压不住误调。 |
| **3.4-B（Non-Call 对称扩展，推荐对照）** | `R_format = 1`, `R_correct = +3` → **总分 `+4`** *(可选：`missing_param` 点出参数名给 `+3`，未点出给 `+2`)* | `R_format = 1`, **`R_correct = -3`** → **总分 `-2`** | `R_format = 0`, **`R_correct = -3`** → **总分 `-3`** | Call 与 Non-Call **满分同为 `+4`、最低分同为 `-3`**，梯度量级完全对称，直接针对性解决 `irrelevance` 和 `post-tool` 泄漏问题。 |

*(可在 Smoke 阶段用 `300` 条对比一次 `3.4-A` 与 `3.4-B`，实测哪个更容易把 `irrelevance` 和 `simple_python` 同时推高！)*

---

## 4. GRPO 数据集设计与规模（Phase 1：静态多轮切片，少而精）

### 4.1 构建管线：从已有干净原始池 + `ckpt-68` Pass@4/8 筛选
所有数据均来自本地已有开源数据集的剩余干净池（严格排除 `BFCL` / `API-Bank` / `τ²` 测试集及测试实体，并与 `sft_v3` 验证/训练集按三键隔离防漏题）：
- `xlam-60k`（余 ~4.8 万条严格类型通过样本）
- `toolace`（余 ~7,800 条干净样本，含单轮、并行、缺参、拒调、多轮）
- `apigen-mt-5k`（余 ~1,900 条非 `τ²` 实体轨迹）
- `smoltalk`（SHA-256 锁定干净通用问答子集）

**难度筛选策略**：用最终选定的 SFT 基座权重以 `temperature=0.7` 对候选题目各采样 4–8 次，**只保留既有对也有错的“边界题”（如 1/4~3/4 对，或错例占 25%–85%），淘汰 SFT 已 100% 稳拿的简单题**（无组内方差的题对 GRPO 无梯度贡献）。

### 4.2 数据规模与配比表（Smoke 300 条 vs 正式 GRPO v1 1,500 条）

采纳 **Phase 1 静态多轮切片** 方案：所有多轮样本均展开为标准单步决策 Prompt（`messages` 内含完整前序对话与工具回包），由模型生成单步响应并立即用规则 Reward 打分。

| 能力桶 | GRPO Smoke (`300`) | GRPO v1 (`1,500`) | 构造与筛选方法 | 目标改进项 |
| --- | ---: | ---: | --- | --- |
| **1. 高难度单轮 & `1-tool` 正例** | 75 | **380** | 从残余 `xlam`/`toolace` 挑出 SFT 易错题（含可选参数、数值/字符串类型易错、嵌套 dict）；**保证 >50% 为 `len(tools)==1`** | 保住并继续拉升 `simple_python` / `live_simple` |
| **2. 成对对抗边界（Call vs Irrelevance）** | 80 | **400** | **同源成对构造（200 对）**：同一请求配真工具（Call 正例） vs 删去匹配工具或替换为伪相似/万能接口如 `requests_get`/`math`（Non-Call 拒调例） | 打破 `simple_python` vs `irrelevance` 跷跷板 |
| **3. 静态多轮切片 · 缺参追问 & 缺工具恢复** | 75 | **400** | ① **缺参追问**（200）：隐藏输入中 1 个必填参数，Reward 要求不调用且追问；<br>② **缺工具后恢复调用切片**（200）：Prompt 构造「Turn 0 工具缺失时助手已合理提示缺工具 → Turn 1 用户重申/补充且**当前 `tools` 已注入目标工具**」，要求模型**本步必须立刻调用出该工具**（若继续返空直接 `-3`） | **重点突破 `multi_turn_miss_func`（1% -> 目标显著提升）**与 `miss_param` |
| **4. 静态多轮切片 · 续调 & 回包收尾** | 70 | **320** | 从 `apigen-mt` 与 `toolace` 多轮干净轨迹抽取中间切片：<br>① 收到前序工具回包后的**下一步调用**（160）；<br>② **任务已完成后带 `<tools>` 挂载的纯文本收尾**（160，严禁吐 `<tool_call>`） | 稳固 `multi_turn_base`，**直接压低 API-Bank 回包文本泄漏（128/528）** |
| **总计** | **300** | **1,500** | 全量 100% 确定性规则校验器打分 | 双卡下 Smoke ~10 分钟、v1 1 epoch ~30–45 分钟 |

---

## 5. 已拍板事项与剩余待决策问题

### 5.1 本轮已敲定事项
- ✅ **算力模式**：确定以 **双卡 2 × A100-40GB** 先跑 GRPO Smoke，再跑正式 GRPO（采用隔离 RL 环境 + `vllm_mode="server"` 双卡分离拓扑）。
- ✅ **Reward 总结构**：$R_{\text{final}} = R_{\text{format}} + R_{\text{correct}}$（`format ∈ {0,1}`，`correct ∈ [-3, 3]`），采用 ToolRL 多重集合 Jaccard + 一对一 Key Jaccard + Value 匹配 + `S_max` 归一化；不强迫 `<think>`、关闭长度奖励、取消 `R_format=0` 硬锁死。
- ✅ **值匹配口径**：对齐 **BFCL 类型与可选参数宽松等价规则**。
- ✅ **Multi-turn 形式**：Phase 1 先采用 **静态多轮切片**（Prompt 带前序历史，单步生成算 Reward）。
- ✅ **核心 RL 超参**：`num_generations (G) = 4`，KL 惩罚 `beta` 小起步（如 `0.01–0.02`），`loss_type` 先用 **TRL 原生 `grpo`**。

### 5.2 仍需进一步拍板的 4 个小决策
1. **GRPO 起始权重**：待 `sft-v3 r=16` 快评与 `r=64 ckpt-68` 对比出炉后正式选定（并决定：**将选定的 LoRA merge 进 2B 基座后再挂新 LoRA 跑 GRPO**，还是继续在原 SFT LoRA adapter 上续训；*建议 merge 后挂新 LoRA `r=16/32`，这样 `disable_adapter()` 恰好把 SFT 后的模型直接当做 Reference Policy，最符合标准 RLHF KL 约束*）。
2. **Non-Call 样本的 `R_correct` 映射（3.4-A vs 3.4-B）**：GT 不需要调用时，采用 ToolRL 原版 `0`（配合误调 `R_format=0`，分差 `1.0`），还是采用对称扩展 `±3`（正确不调 `+3` / 误调 `-3`，分差 `6.0`）？*建议 Smoke 阶段用少量消融或直接选对称 `3.4-B` 防止被 Call 正分淹没。*
3. **建环境方式选择（方案 E1 vs E2）**：双卡环境是复用现有 `runs/envs/train` + 只新建 `runs/envs/rl-serve`（最轻快），还是新建一个完整的 `runs/envs/rl` 统一管理客户端与服务端？
4. **学习率与训练长度**：GRPO 阶段学习率通常比 SFT 小 1–2 个数量级（建议 `5e-6 ~ 2e-5`、`cosine`、每 `50` 步或 `0.25 epoch` 保存 checkpoint 并按同样 BFCL 9 类 + API-Bank 快评监测是否发生 Reward Hacking）。
