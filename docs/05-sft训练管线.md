# 5. SFT 训练管线设计

本文档定义 `Qwen/Qwen3.5-2B` 与 `Qwen/Qwen3.5-0.8B` 第一阶段监督微调（SFT）的工程与实验方案，承接[基线评测](02-基线评测.md)、[数据集调研](03-sft数据集调研.md)与[数据归档与样本图鉴](04-数据归档与样本图鉴.md)。

为兼顾**研发迭代速度、数据纯净度与清晰的数据消融实验**，SFT 按 **V0（链路烟囱）→ `SFT-v1`（仅 `xlam-60k + toolace` 精选）→ `SFT-v2`（按需增量补 `apigen-mt-5k` 与通用保持）** 递进实施，为后续 **OPD** 与 **Agentic RL（GRPO）** 打底。

---

## 1. 基线问题映射：SFT 要解决什么、怎么分步打

结合 64K 正式基线（[01-baseline-report.md](../reports/01-baseline/01-baseline-report.md)）暴露出的真实错因与本地样本盘点（[01-sft-data-analysis.md](../reports/02-data-audit/01-sft-data-analysis.md)）：

| 基线表现与核心病灶 | 2B / 0.8B 基线水平 | 分步解决策略 |
| --- | --- | --- |
| **并行工具调用**（`parallel` / `parallel_multiple`） | 50.5% / 71.0%（2B）<br>31.5% / 52.5%（0.8B） | **`SFT-v1` 主攻**：`xlam-60k` 含 31,539 条多调用 + `toolace` 含 4,144 条并行消息，优先采样强化 |
| **参数忠实度与嵌套 Schema**（`value_error:string`、`dict_key`、`multiple`） | BFCL 参数错 287 / 337 次；0.8B `multiple` 63.5% | **`SFT-v1` 主攻**：两套干净源合并清洗，筛选标准 JSON Schema 与多候选调用样本 |
| **拒答边界与回复轮标签泄漏**（`live_irrelevance` 误调用、API-Bank 回复轮吐 `<tool_call>`） | 2B `live_irrelevance` 错 328 次；API-Bank 回复轮泄漏 106/79 次 | **`SFT-v1` 优先覆盖 + `SFT-v2` 兜底**：`SFT-v1` 全留 `toolace` 1,941 条 0 调用与 3,740 条自然语言回复；若 `irrelevance` 仍退化则在 `SFT-v2` 混入纯无工具对话 |
| **多轮状态推进与交互**（`multi_turn_base` / `long_context`） | 29.5% / 28.0%（2B）<br>10.5% / 10.0%（0.8B） | **`SFT-v1` 初步注入 → `SFT-v2` 补齐**：`toolace` 仅有 797 条含 `tool` 返回轮（上采样 `2×–3×`）；长程多轮上限由 `SFT-v2`（`apigen-mt-5k` 21,955 次交互）+ OPD/GRPO 拉升 |
| **缺工具恢复与缺参追问**（`miss_func` / `miss_param`） | 2.5% / 15.0%（2B）<br>2.0% / 8.0%（0.8B） | **留给后续阶段主攻**：SFT 打底后由 OPD 与可验证多轮沙盒 GRPO 解决 |
| **非 Python 语法题**（`simple_java` / `simple_javascript`） | 14–29% | **战略降级**：不单独造 Java/JS 语法数据，避免为冷门字面量规则稀释通用 JSON Schema 能力 |

---

## 2. 管线总体架构与目录规划

沿用仓库隔离原则，数据处理与训练逻辑分别落在 `src/toolcall/data/` 与 `src/toolcall/training/`：

```text
configs/
├── data/                   SFT-v0 / SFT-v1 / SFT-v2 配比与清洗配置 YAML
└── train/                  SFT 超参配置（LoRA / Full SFT）
src/toolcall/
├── data/                   原始解析 → 统一 Schema → 清洗去重 → 统计/导出
└── training/               Tokenizer 对齐检查、TRL SFTTrainer 训练入口、LoRA 合并
data/
├── raw/                    已归档原始快照（不改原文件）
├── eval-inspection/        评测样例索引（严禁训练脚本读取）
└── processed/              sft_v1/（含 train/val/smoke_v0、manifest、补丁模板与抽检样例）及后续 sft_v2/
runs/train/                 checkpoint、merged 权重、训练日志与 SwanLab 记录
tests/                      格式转换、Chat Template 与 Loss Mask 单元测试
```

---

## 3. 数据清洗与统一标准化流程（`data/`）

**核心原则：绝不把原始异构字符串直接拼进训练，必须经过“反序列化 → Schema 标准化 → 脏数据过滤 → 去重”四道工序。**

### 3.1 统一中间格式（OpenAI-style）

所有样本清洗后统一存为包含 `messages`、`tools` 与 `meta` 的 JSONL：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Beijing?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "type": "function",
          "function": {"name": "get_weather", "arguments": {"city": "Beijing"}}
        }
      ]
    },
    {"role": "tool", "name": "get_weather", "content": "{\"temp\": 25}"},
    {"role": "assistant", "content": "The temperature in Beijing is 25°C."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "meta": {
    "source": "toolace",
    "task_type": "multi_turn",
    "num_turns": 2,
    "num_tool_calls": 1,
    "has_tool_response": true,
    "is_parallel": false,
    "is_refusal": false
  }
}
```

### 3.2 `SFT-v1` 主力数据源实测格式与清洗细则

结合本地 `data/raw/` 实测归档（[01-sft-data-analysis.md](../reports/02-data-audit/01-sft-data-analysis.md)）逐项处理：

| 数据集 | 实测规模与原始结构 | 关键清洗与防坑规则 |
| --- | --- | --- |
| **`xlam-60k`** | 60,000 行（28,461 单调用 + 31,539 多调用，0 空调用）；`id + query + tools(JSON str) + answers(JSON str)` | 1. `json.loads(tools/answers)`；将 `parameters` 扁平字典转为标准 JSON Schema（`type: "object", properties, required`），映射 `int/str/float/bool/list`。<br>2. 仅构造 `user -> assistant(tool_calls)`，严禁伪造 `tool` 返回轮。 |
| **`toolace`** | 11,300 行（含 `tool` 返回 797 条、`>=2` 轮 530 条、并行 4,144 条、0 调用 1,941 条）；**仅 `conversations[{from, value}]` 一列，`system` 为列表首项，无独立 `tools` 列**；assistant 为 `[函数名(参数=值)]` | 1. **抽出工具并剥离旧指令**：从 `conversations[0]` 提取函数 JSON 列表放入 `tools`，并剥离 `Put it in the format of [func1(...)]` 文本指令，防止与 Qwen3.5 模板冲突。<br>2. **Schema 类型修正**：将 `parameters.type` 里的 `"dict"` 统一改为 `"object"`。<br>3. **含空格函数名处理**：`toolace` 部分函数名含空格（如 `"Market Trends API"`），统一规范化（空格转下划线）并同步替换 `tool_calls` 调用名；无法无损解析的异常行直接丢弃。 |

### 3.3 `SFT-v2` 备选扩展源清洗要点（暂不进 V1，代码可预留或二期接入）

- **`apigen-mt-5k`（5,000 行，21,955 次交互）**：`json.loads(tools)`；映射 `human/gpt/function_call/observation`，合并同轮多个 `function_call` 为并行 `tool_calls`；上线前核查与 `τ²` Retail/Airline 任务去重。
- **`smoltalk` 通用子集 / `magpie-pro-300k`**：仅取纯无工具通用对话（排除 `smoltalk` 的 `all` 与 `apigen-80k`）。
- **`glaive-v2`（112,960 行）**：仅在需要大量补强拒答/追问时再按状态机严格过滤接入，前期不引入其噪声。

### 3.4 公共过滤与去重标准

1. **Schema 合法性校验**：`tool_calls` 函数名必须存在于 `tools` 中，`arguments` 必须为可解析 JSON Object，且满足 `required` 字段约束。
2. **序列配对校验**：`tool` 消息必须紧跟在含 `tool_calls` 的 `assistant` 之后；结尾不得截断在 `user` 或无后续回复的 `tool` 轮。
3. **长度过滤**：用 `Qwen3.5` tokenizer 统计渲染后总 token 数，过滤超过训练 `max_length`（如 `> 8192`）的样本，防止静默截断。
4. **去重**：按归一化首轮 query + tools 签名去重。

---

## 4. Qwen3.5 模板契约与 Loss Masking 硬校验（开训前必过）

**必须在 `tests/` 中编写自动化断言，启动 GPU 训练前 100% 通过：**

1. **推理与训练模板 100% 同源**：
   - 调用 `runs/envs/train` 中随 `Qwen/Qwen3.5-2B` 权重提供的 `tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=False)`。
   - 打印多轮与并行样本的渲染结果，确认 `<tool_call>` / `<tool_response>` 标签格式与评测端 vLLM `--tool-call-parser qwen3_coder` 解析契约完全一致，且 `enable_thinking=False` 渲染一致。
2. **`assistant_only_loss=True` 与 `{% generation %}` 边界校验**：
   - 实测发现 `Qwen/Qwen3.5-2B` 原生 `chat_template.jinja` 不含 `{% generation %}` 标记，直接传 `assistant_only_loss=True` 会导致 mask 全 0。
   - 需通过 `patch_qwen35_chat_template()` 生成字节等价的训练模板（`data/processed/sft_v1/qwen35_chat_template_train.jinja`），并经 `load_tokenized_sft_dataset()` 预生成 `input_ids + labels`（同时规避 PyArrow 嵌套 Dict 键合并污染），断言：
     - **参与 loss（`labels != -100`）**：仅包含 `assistant` 的 `<tool_call>` 与自然语言回复（含轮末 `<|im_end|>`）。
     - **严格 mask（`labels == -100`）**：`system`（含 tools 声明）、`user` 输入、`tool` 返回（`<tool_response>`）以及 `<|im_start|>assistant\n<think>\n\n</think>\n\n` 前缀全部屏蔽。

> `SFT-v1` 的本地完整构建统计与详细实验方案已落地于 [06. SFT-v1 实验方案](06-sft-v1实验方案.md)。

---

## 5. `SFT-v1` 与演进数据配比方案

### 5.1 规模分步：先 V0 验证链路，再跑 `SFT-v1`

1. **V0 烟囱验证版（~3,000 条，仅从 `xlam + toolace` 抽样）**
   - **目的**：10 分钟内跑通 `数据转换 → SFTTrainer → LoRA 合并 → vLLM → BFCL 8 类子集` 全闭环，排查模板/Mask/Parser Bug。
   - **抽样**：`xlam` 1,500（并行 1,000 + 单调用 500）+ `toolace` 1,500（优先包含全部类型：多轮 + 并行 + 0 调用）。
2. **`SFT-v1` 正式精选版（仅 `xlam-60k + toolace`，约 26,000–27,000 条）**
   - **核心设计理由**：**切忌把 60k `xlam` 全量直接与 11.3k `toolace` 拼接**，否则 85% 的单轮必调样本会彻底淹没 `toolace` 的 797 条多轮反馈与 1,941 条拒答样本！

| 来源与子桶 | `SFT-v1` 采样策略 | 建议条数（占比） | 核心作用 |
| --- | --- | --- | --- |
| **`toolace` · 多轮反馈子集** | 清洗后保留全部 **797 条**含 `tool` 返回样本，做 **`2×–3×` 上采样** | ~1,600–2,400（~8%） | 放大工具返回理解与多轮衔接信号 |
| **`toolace` · 0 调用边界子集** | 清洗后全保留 **1,941 条**无调用样本（拒答 / 缺参追问 / 纯对话） | ~1,900（~7%） | 守住 `irrelevance` 拒答底线与回复轮格式边界 |
| **`toolace` · 常规单轮/并行** | 清洗后其余全部单轮及并行样本（含 4,144 并行消息主力） | ~7,500（~28%） | 补充多样化嵌套 Schema、复杂类型与中文调用 |
| **`xlam-60k` · 并行/多调用** | 从 **31,539 条**多调用中去重抽样 | ~10,000（~38%） | 重点拉升 `parallel`（50.5%/31.5%）与 `parallel_multiple` |
| **`xlam-60k` · 多候选单调用** | 从 **28,461 条**单调用中优先抽选多候选工具样本 | ~5,000（~19%） | 巩固 `multiple` / `live_multiple` 选择准确率 |
| **合计（`SFT-v1`）** | 全部 Shuffle 混合 | **~26,000–27,000** | 干净、无脏噪点，单卡 A100 快速出消融结论 |

> **2B vs 0.8B 调优微调**：`0.8B` 单轮底子比 `2B` 低 15–19 个点，可在保持 `toolace` 全量不变的前提下将 `xlam` 并行/单调用适当放宽至 `18,000` 条；`2B` 则严格按上表控单轮总量，防 `irrelevance` 退化。

3. **`SFT-v2` 增量消融预案（视 `SFT-v1` 评测结果决定是否启动）**
   - 若 `SFT-v1` 跑完后：`parallel` 明显提升，但 `multi_turn_*` 提升受限或 `irrelevance` 出现下降 → 直接接入本地已备好的 **`apigen-mt-5k`（5k 多轮轨迹）** + **`smoltalk` 通用子集（~5k 无工具对话）**，形成 `Baseline vs SFT-v1 vs SFT-v2` 的定量数据消融对照。

### 5.2 难度分层策略

- **SFT 内部不做“先简单 epoch、后困难 epoch”的硬切换**，统一将单轮、并行、拒答与多轮样本 **Shuffle 混合训练**，避免后期难样本覆盖前期单轮与拒答能力。
- **跨阶段课程递进**：`SFT-v1/v2（混合打底）` → `OPD（难多轮 On-Policy 蒸馏）` → `Agentic RL / GRPO（可验证多轮沙盒专攻 miss_param / miss_func / 错误恢复）`。

---

## 6. 训练参数与评测放行规范（`training/`）

基于单张 **A100-SXM4-40GB**、`runs/envs/train`（`torch==2.10.0+cu129`、`transformers>=5.0`、`trl>=0.27`、`flash-linear-attention`）：

### 6.1 默认超参矩阵

| 参数项 | LoRA SFT（V0 & `SFT-v1` 首选） | Full SFT（定型后对照实验） | 说明 |
| --- | --- | --- | --- |
| **模型精度与内核** | `bf16=True` + `flash-linear-attention` | 同左 | 加速 Qwen3.5 Gated DeltaNet，不开启量化 |
| **LoRA 配置** | `r=64`, `lora_alpha=128`, `lora_dropout=0.05`, `target_modules="all-linear"` | — | 抗遗忘且产物轻（~200MB）；`SFT-v1` 优先 LoRA 跑通并挑最佳 ckpt |
| **学习率与调度** | `1.5e-4`（`cosine`, `warmup_ratio=0.05`） | `1.5e-5`（`cosine`, `warmup_ratio=0.05`） | 小模型全参严控在 `1e-5 ~ 2e-5`，防止震荡与通用能力退化 |
| **序列上限与打包** | `max_length=8192`, `packing=True` | 同左 | 若 `packing=True` 与 `assistant_only_loss=True` 存在边界兼容问题，优先保 `assistant_only_loss` 精确性 |
| **Batch Size** | 不开 packing：`bs=2, accum=16`（有效 32）；开 `packing=True`（8192 打包）：建议 **`bs=1, accum=8` + `gradient_checkpointing=True`**（~65K tok/step，防 248k 大词表 logits 爆 40GB 显存） | 同左（开 `gradient_checkpointing=True`） | `packing=True` 下若用太大有效 token 会导致每 epoch step 过少；设 `save_total_limit=4` 保存全部半轮 ckpt |
| **训练轮数与选优** | `num_train_epochs=2`（每 `0.5` epoch 存 ckpt） | `num_train_epochs=2` | 因 V1 未混入纯无工具通用对话，严控 `<= 2` epochs 防过拟合 |

### 6.2 评测与晋级门禁

1. **快速筛选（每 `0.5` epoch checkpoint）**：LoRA merge 后用 vLLM 64K + Prefix Caching 启动，跑 BFCL **8 类高性价比子集**（1,840 题，几分钟出分）：
   ```bash
   .venv/bin/toolcall-eval bfcl-v4 --categories \
     simple_python multiple parallel parallel_multiple irrelevance \
     multi_turn_base multi_turn_miss_func multi_turn_miss_param \
     --variant sft-v1-ckpt --model "$model" --base-url "$base_url" \
     --concurrency 128 --timeout 300 --gpu-index 0
   ```
2. **正式验收与决策**：对最佳 checkpoint 补跑 **`BFCL --profile core`（4,441 题）+ `API-Bank --profile full`（1,281 步）**：
   - 核对 **`parallel` / `multiple` / `simple_python`** 是否大幅超越基线；
   - 核对 **`irrelevance` / `live_irrelevance`** 与 **`response_leaked_tool_call`** 是否守住或改善；
   - 依据 `multi_turn_*` 涨幅决定直接进 **OPD / GRPO**，或先补一轮 **`SFT-v2`（+`apigen-mt-5k` + 通用对话）** 消融对照。
