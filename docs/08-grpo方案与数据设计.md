# 8. Stage 2 GRPO 训练管线、Reward 与数据集设计（v3）

> 状态：**v3 方案收敛中** · 历史归档：[v1 初版](archive/08-grpo方案与数据设计-v1.md) · [v2 ToolRL 修订版](archive/08-grpo方案与数据设计-v2.md)
> 前置基准：`SFT-v3`（`r=64 checkpoint-68` 已通过 BFCL `core` 17 类验收；`r=16` 快评进行中）
> 相关文档：[00-服务器环境安装](00-服务器环境安装.md) · [03-sft数据集调研](03-sft数据集调研.md) · [SFT-v3 Core 报告](../reports/03-sft/05-sft-v3-2b-ckpt68-core-eval-report.md)

### v2 -> v3 本次迭代更新要点
1. **明确隔离环境的技术原因**：不仅是避免弄坏现有 SFT（`train`）与评测（`serve`）环境，更因为 `vllm==0.19.1` 与 `trl==1.18.2 / flash-linear-attention` 存在依赖锁冲突与服务端路由依赖；
2. **正式敲定 Reward 3.4-B（Non-Call 对称扩展）**：Non-Call 正确不调得 `R_correct = +3`（总分 `+4`），误调得 `-3`（总分 `-2`），正文泄漏标签得 `R_format = 0, R_correct = -3`（总分 `-3`），与 Call 区间完全对称；
3. **重构 GRPO 数据集来源与比例（不排除 SFT ID，按 `40% / 20% / 40%` 精简高质量构建）**：以清洗去重后的 `xlam-60k + toolace` 为核心来源（Smoke 300 条 / 正式 1,500 条），并单列「格式 / Template / System Prompt 一致性」历史防坑硬契约。

---

## 1. 为什么要做 GRPO（以及 GRPO 只该解决什么问题）

`SFT-v3`（`r=64 checkpoint-68`）已将 `Qwen/Qwen3.5-2B` 的 BFCL `core` 全量微平均从 `59.27%` 提升到 `65.71%`（`Non-Live AST` `60.38% -> 82.25%`、`simple_java` `+45.0 pp`），但也明确暴露出静态 SFT 交叉熵的三类边界问题：
1. **单工具调用 vs 拒调的跷跷板**：SFT 后半程（`ckpt-102..168`）能把 `irrelevance` / `live_irrelevance` 拉到 `89.2% / 85.4%`，但代价是 `simple_python` 发生 70 题零调用误拒（掉 13 个百分点）。
2. **`multi_turn_miss_func` 恢复断点**：缺少动态工具注入后的反馈信号，后期模型在 Turn 0 拒答后，Turn 1 工具补齐时直接返空停住（`1.0%`）。
3. **API-Bank 回包后正文泄漏 `<tool_call>`（`128/528`）**：在挂载 `tools` 但应纯文本收尾的步骤上，缺少对误吐 `<tool_call>` 标签的负反馈。

**GRPO 的定位**：利用可编程验证的 Reward（RLVR），在已选定的 SFT 策略基础上，校准**“该调 vs 拒调”“缺工具补齐后立刻调用”“参数键值精确匹配”“回包零标签泄漏”**四个决策边界。

---

## 2. 双卡 2 × A100-40GB 拓扑与新建隔离环境的原因

### 2.1 为什么要新建隔离环境？（不只是防干扰，还有三重硬原因）
核查远端环境现状：`runs/envs/train` 已装 `torch==2.10.0+cu129`、`transformers==5.24.0`、`trl==1.18.2`、`peft==0.22.0`、`flash-linear-attention`（**无 `vllm`**）；`runs/envs/serve` 已装 `torch==2.10.0+cu129`、`transformers==5.24.0`、`vllm==0.19.1`（**无 `trl` / `peft`**）。

新建独立环境主要有三个目的：
1. **保护已验证通过的 SFT 与评测环境（最直接目的）**：实测发现 **`trl==1.18.2` 的 `vllm_mode="server"` 硬性要求 `vllm >= 0.23.0`（支持 `--weight-transfer-config '{"backend": "nccl"}'` 与 `VLLM_SERVER_DEV_MODE=1`）**，且训练侧 `VLLMClient` 也会导入 `vllm.distributed.weight_transfer`；而 `vllm==0.30.0` 依赖 **`torch==2.13.0+cu129`**，与现有 `runs/envs/train`（`torch==2.10.0+cu129`）及 `runs/envs/serve`（`vllm==0.19.1`）版本锁均不兼容。因此绝不能在现有环境中覆盖升级，必须新建隔离 RL 环境。
2. **双卡统一隔离环境 `runs/envs/rl`（兼容 `E1/E2`）**：在一个全新的 `runs/envs/rl` 内对齐锁定 `torch==2.13.0+cu129 + vllm==0.30.0 + transformers==5.24.0 + trl==1.18.2 + peft==0.22.0 + flash-linear-attention==0.6.0 + fastapi + uvicorn + swanlab`，同时支撑 GPU 0 的 `trl vllm-serve` 与 GPU 1 的 `GRPOTrainer`（同版本 PyTorch/NCCL 通讯零错配），现有 `train` 与 `serve` 零改动。
3. **支持按需回滚与复现**：遵循项目一贯的隔离规范（`.venv` / `serve` / `train` / `tools/*` 各自独立），RL 环境即使后续升级 TRL/vLLM 参数出错，删掉重建也不会牵连其他管线。

### 2.2 双卡拓扑与建环境耗时估算

在 **2 × A100-40GB** 上采用 **TRL `vllm_mode="server"` 物理隔离双卡拓扑**：
- **GPU 0（Rollout 推理卡）**：运行 `trl vllm-serve`，以 Continuous Batching + Prefix Caching 专门负责 `G=4` 批量采样生成；每次训练步结束后接收权重同步。
- **GPU 1（训练更新卡）**：运行 `GRPOTrainer`，专心计算 Reference Logprobs（PEFT LoRA 下直接 `disable_adapter()` 复用底座，**零第二份 2B 权重拷贝**）与反向传播更新（开启 `gradient_checkpointing=true` 防 248K 词表 logits 峰值爆显存）。
- **预期速度**：`300` 条 Smoke × `G=4`（1,200 次生成）约 **~8–15 分钟**；`1,500` 条正式 GRPO × `G=4`（6,000 次生成）约 **25–45 分钟** / epoch。

| 建环境方案 | 安装内容与做法 | 额外磁盘 | 预计耗时（国内 TUNA + 上交大 `cu129`） |
| --- | --- | ---: | ---: |
| **方案 E1（推荐，最轻快）** | **GPU 1 复用 `runs/envs/train`（零改动）**；仅新建 **GPU 0 的 `runs/envs/rl-serve`**：装 `torch==2.10.0+cu129`、`vllm==0.19.1`、`transformers==5.24.0`、`trl==1.18.2`、`fastapi`、`uvicorn` | ~`5–6 GB` | **约 4–8 分钟**<br>*(因 `/root/.cache/uv` 已清理，需重下 `torch` + `vllm` wheel)* |
| **方案 E2（新建统一 `runs/envs/rl`）** | 新建单一环境装齐 `torch==2.10.0+cu129` + `vllm==0.19.1` + `transformers==5.24.0` + `trl==1.18.2` + `peft>=0.22` + `accelerate>=1.0` + `datasets>=4.0` + `flash-linear-attention` + `swanlab` + `fastapi` + `uvicorn` | ~`6–8 GB` | **约 5–10 分钟** |

---

## 3. Reward Design 详细方案（仿 ToolRL + 3.4-B 对称 Non-Call 扩展）

### 3.1 总公式（默认关闭长度奖励）

$$R_{\text{final}} = R_{\text{format}} + R_{\text{correct}}$$

其中 $R_{\text{format}} \in \{0, 1\}$，$R_{\text{correct}} \in [-3, 3]$，总奖励范围为 **$[-3, 4]$**；长度奖励 $R_{\text{length}}$ 默认关闭（置 `0`）。

### 3.2 格式奖励 `R_format ∈ {0, 1}`（关思考模式适配）

本项目统一 `enable_thinking=false`，**不检查 `<think>` 推理内容与词数**，允许官方模板自带的空 `<think>\n\n</think>\n\n` 前缀，严格校验结构闭合与无标签泄漏：
1. **模型输出包含工具调用时**：`<tool_call>...</tool_call>` 结构完整成对闭合，能无异常解析出合法函数名与 `dict` 类型 JSON 参数，且无残缺尾缀 → `R_format = 1`；否则 `R_format = 0`。
2. **模型输出为纯文本（无工具调用）时**：正文非空且**完全不含** `<tool_call>`、`</tool_call>`、`<function=`、`<parameter=`、`"arguments": {` 等伪调用标签泄漏 → `R_format = 1`；否则 **`R_format = 0`**。

*(说明：当 `R_format == 0` 时，不直接强制锁死 `R_correct = -3.0`，而是按下方规则各自独立计分；仅在 GT 需要调用但预测完全无法解析任何调用时，置 `R_correct = -3.0`。)*

### 3.3 正确性奖励 `R_correct ∈ [-3, 3]` —— Call 样本（GT 需要调用工具，$|G| > 0$）

设金标调用列表 $G = [g_1, \dots, g_{|G|}]$，预测解析列表 $P = [p_1, \dots, p_{|P|}]$：
1. **Tool Name Reward（多重集合 Jaccard Similarity）**：
   $$r_{\text{name}} = \frac{|N_G \cap N_P|}{|N_G \cup N_P|} \in [0, 1]$$
   其中 $N_G, N_P$ 为考虑并行重复调用次数的工具名称多重集合。
2. **一对一匹配关系 $M$**：
   同名工具之间建立最优一对一匹配 $M = \{(i, j)\}$（同名多调用时优先匹配键值重合度最高的组合）。
3. **Parameter Key Reward**：
   $$r_{\text{key}} = \sum_{(i,j) \in M} \frac{|K_i^G \cap K_j^P|}{|K_i^G \cup K_j^P|}$$
   - **无参补丁**：若 $K_i^G = K_j^P = \emptyset$，该项得 `1.0`（免除零且无参全对得满分）；
   - **可选参数兼容（已定）**：对齐 BFCL 规则，不强制惩罚合理省略的 optional 默认参数，但严罚模型编造 schema 外的幻觉键名。
4. **Parameter Value Reward（对齐 BFCL 宽松等价规则，已定）**：
   $$r_{\text{value}} = \sum_{(i,j) \in M} \sum_{k \in K_i^G} \mathbb{1}_{\text{BFCL}}\big[P_j(k) = G_i(k)\big]$$
   数值等价对齐（`10 == 10.0`）、字符串去除冗余首尾空白与 BFCL 规范化容差、嵌套列表/字典递归匹配。
5. **归一化映射与特殊情况**：
   $$S_{\max} = 1 + |G| + \sum_{i=1}^{|G|} |K_i^G|, \qquad R_{\text{correct}} = 6 \cdot \frac{r_{\text{name}} + r_{\text{key}} + r_{\text{value}}}{S_{\max}} - 3 \in [-3, 3]$$
   - 需要调用工具，但预测无法解析或空调用（$|P| = 0$）→ **`R_correct = -3`**；
   - 工具调用完全正确 → **`R_correct = +3`**（总分 `+4`）。

### 3.4 正确性奖励 `R_correct ∈ [-3, 3]` —— Non-Call 样本（已定：**3.4-B 对称扩展**）

针对 **Ground Truth 不需要调用工具** 的样本（`irrelevance`、`missing_param`、`post-tool response`），采用 **3.4-B 对称映射**，保证与 Call 样本奖励量级完全对齐、杜绝偏好漂移：

| Non-Call 场景 | 模型行为：正确不调用（`tool_calls == []` 且无泄漏） | 模型行为：误调任意工具（`len(tool_calls) > 0`） | 模型行为：纯文本正文泄漏 `<tool_call>` |
| --- | --- | --- | --- |
| **`irrelevance`（工具不适用拒答）** | `R_format = 1`, `R_correct = +3` → **总分 `+4`** *(若回复 `< 5 tokens` 空洞敷衍可微扣 `0.5`)* | `R_format = 1` *(或 `0`)*, **`R_correct = -3`** → **总分 `-2`** | `R_format = 0`, **`R_correct = -3`** → **总分 `-3`** |
| **`missing_param`（缺参追问）** | `R_format = 1`, `R_correct = +3`（若准确点出缺失的 `required` 参数名/语义给 `+3`；泛泛拒绝给 `+2`）→ **总分 `+3 ~ +4`** | **`R_correct = -3`** → **总分 `-2`** | `R_format = 0`, **`R_correct = -3`** → **总分 `-3`** |
| **`post-tool response`（回包收尾）** | `R_format = 1`, `R_correct = +3`（正文覆盖工具回包核心数值/实体给 `+3`；纯套话给 `+1.5`）→ **总分 `+2.5 ~ +4`** | **`R_correct = -3`** → **总分 `-2`** | `R_format = 0`, **`R_correct = -3`** → **总分 `-3`** |

---

## 4. GRPO 数据集设计（高质量 `xLAM-60k + ToolACE` · `40% / 20% / 40%` 配比）

### 4.1 来源与筛选策略：不强制排除 SFT 历史样本，质量与难度优先
采纳最新共识：
- **核心数据源**：以 **`xlam-60k`** 与 **`toolace`** 经严格清洗与去重后的高质量全池为主体（不再强行排除已在 `sft_v3/train.jsonl` 出现过的题目，因为同一批高质量题目进入 RL 阶段同样有效；**但依然强制保持与 `sft_v3/val.jsonl` 及 BFCL / API-Bank / `τ²` 测试集 100% 零重合**）。
- **高质量筛选与防全对空跑**：
  1. 复用 `sft_v3` 全部严格校验（`strict_arg_types=True`、`reject_duplicate_tools=True`、ToolACE 双向严格边界清洗、零正文伪调用泄漏、`max_prompt_tokens <= 4096`）；
  2. **可选 Pass@4 过滤（强烈推荐）**：用最终选定的 SFT 起始权重（`temperature=0.7`）在候选池跑一轮快速推理，**过滤掉 4/4 全对且无任何瑕疵的“零方差简单题”**，优先把有限的 300 / 1,500 个名额留给 SFT 还会出错、摇摆或边界模糊的高价值题目（也可保留少量 ~15% 锚点题防遗忘）。

### 4.2 能力桶配比方案（Smoke 300 条 vs 正式 1,500 条）

严格遵循 **单轮调用 `40%` : 多轮 `20%` : 并行+拒答+缺参+其他 `40%`**：

| 能力大类（占比） | 细分能力子桶 | GRPO Smoke (`300`) | GRPO v1 (`1,500`) | 样本来源与构造要点 |
| --- | --- | ---: | ---: | --- |
| **1. 单轮工具调用 (`40%`)** | `single_turn_call` (`xlam_single` + `toolace_single`) | **120** | **600** | 多候选单选 + **`len(tools)==1` 单工具正例（建议占该桶 ≥40%）**，重点挑含可选参数、数值/字符串/嵌套参数约束的高质量题，稳住 `simple_python` 与 `live_simple/multiple` |
| **2. Multi-turn (`20%`)**<br>*(Phase 1 静态多轮切片)* | `multi_turn_step`（续调 + `miss_func` 恢复切片 + 回包收尾） | **60** | **300** | 全部来自 `toolace_multi_turn`（必要时可少量补 `apigen-mt` 非 `τ²` 实体切片）：<br>- **多轮续调切片**（`120`）：前序含 `tool` 回包的下一步调用；<br>- **`miss_func` 工具恢复切片**（`90`）：Turn 0 工具缺失拒答 → Turn 1 注入缺失工具后要求本步立刻调出；<br>- **回包纯文本收尾切片**（`90`）：保留 `<tools>` 挂载，工具刚返回完整结果后要求纯文本总结、零 `<tool_call>` 泄漏 |
| **3. 并行 / 拒答 / 缺参 (`40%`)** | `parallel_call` (`15%`) | **45** | **225** | `xlam_parallel` + `toolace_parallel`（含单工具多参并调 & 多工具并调，适配 `r_name` 多重集合与 `M` 一对一匹配） |
| | `irrelevance_refusal` (`15%`) | **45** | **225** | ToolACE 双向严格拒调 + 基于正例**替换/抽离匹配工具**构造的 `1-tool` 对抗拒调负例（防 `requests_get`/`math` 伪匹配误调） |
| | `missing_param` (`10%`) | **30** | **150** | ToolACE 严格缺参样本 + 隐藏 1 个 `required` 字段构造的缺参追问（要求不调用且指出缺失参数名） |
| **合计 (`100%`)** | **全量可控校验** | **300** | **1,500** | — |

### 4.3 历史踩坑防重犯硬契约（Format / Template / System Prompt 一致性）

吸取 `SFT-v1..v3` 曾经踩过的全部坑，GRPO 数据构建与训练管线在落盘与运行前必须满足以下四条**自动化硬断言**：
1. **官方 Template 100% 字节一致（禁止两套模板打架）**：
   - GRPO 送入 `vLLM` 与 `GRPOTrainer` 的每一个 Prompt 必须通过**与线上评测完全同一份 `Qwen3.5-2B` 官方 `chat_template.jinja`** + `enable_thinking=false` 渲染；
   - 渲染末尾必须精确匹配 `<|im_start|>assistant\n<think>\n\n</think>\n\n`，并用正则与 tokenizer 反解检查，确保训练时的生成前缀和 BFCL / API-Bank 评测零偏差。
2. **System Prompt 洁净统一（彻底清除 ToolACE 格式冲突话术）**：
   - 严禁把原始数据集里自带的旧格式提示（如 ToolACE 里的 `[FuncName(param=val)]` 示例、过时 JSON 拼接指令、或误导性“必须调工具”约束）带入 `system`；
   - 统一清洗后使用标准的 Native FC `system prompt`，确保模型只通过官方 `<tools>` JSON Schema 与 `<tool_call>` 协议进行工具交互。
3. **Strict Schema & 类型防脏门控**：
   - 全量经过 `validate_record(..., strict_arg_types=True, reject_duplicate_tools=True)`，杜绝 `"math.pi / 2"` 数值冲突、重复同名工具、空字段及 `think` 伪调用；
   - Multi-turn 切片里的历史 `assistant.tool_calls` 与 `tool` 消息必须严格保持 OpenAI 结构化格式交由模板统一渲染，禁止把原始 XML 文本硬拼进历史对话里。
4. **污染隔离与可追溯台账**：
   - 强制与 `sft_v3/val.jsonl`、`BFCL V4`、`API-Bank`、`τ²-Bench` 保持三键零重合；
   - 数据目录随附 `manifest.json`（SHA-256）、`audit_ledger.jsonl` 与渲染抽样 `.txt` 人工核验样本（单轮、多轮恢复、拒调、缺参、并行各 1 份），预检不通过即阻断训练。

---

## 5. 剩余待决策清单（仅剩 3 个拍板项）

| 决策项 | 当前进展 / 建议选项 |
| --- | --- |
| **1. GRPO 起始权重** | ⏳ 待 `sft-v3 r=16` 快评出炉后与 `r=64 ckpt-68` 横向对比再拍板；确定后建议**先 `merge_lora` 进 2B 底座再挂新 LoRA（如 `r=16`）跑 GRPO**，这样 `disable_adapter()` 天然充当 SFT Reference Policy，完全贴合 KL 惩罚定义。 |
| **2. 建隔离环境选型** | ⏳ 确认采用 **方案 E1（GPU 1 复用 `runs/envs/train`，仅新建 GPU 0 `runs/envs/rl-serve`）**，还是 **方案 E2（新建统一 `runs/envs/rl`）**？ |
| **3. 学习率 & Rollout 长度** | ⏳ 建议初始：`lr = 1e-5`（cosine）、`beta = 0.02`、`G = 4`、`loss_type = "grpo"`、`max_prompt_length = 4096`、`max_completion_tokens = 512`（关思考下足够覆盖并行调用和回复，大幅省生成耗时）。 |
