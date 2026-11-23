# 3. SFT 数据集调研与构建规范

本文档整理工具调用（Tool Call / Agentic）后训练阶段的**评测集与训练集边界**、**五大训练数据集的 Schema / 体积 / 国内下载方式**，以及面向单张 **A100 40GB（Qwen3.5-0.8B / 2B）** 的数据集构建与分阶段配比方案。

---

## 1. 评测集与训练集边界（防污染原则）

| 类型 | 数据集 | 定位与覆盖能力 | 训练隔离原则 |
| --- | --- | --- | --- |
| **主评测** | **BFCL V4** | Non-Live / Live / Multi-turn / Memory / Web Search（22 类）；测工具选择、参数 AST、并行、拒答、多轮纠错 | **严禁混入训练** |
| **主评测** | **τ²-Bench** | Retail（114）/ Airline（50）/ Telecom（114）；测真实多轮对话、业务 Policy 遵循、数据库终局状态 | **严禁混入训练** |
| **辅助评测** | **API-Bank** | 314 段对话（753 调用 + 528 回复）；测检索与调用回归（注：原生 FC 下 Oracle 上限非 100%，不作主榜） | **314 条评测集严禁混入训练** |
| **可选评测** | **ToolSandbox / ACEBench** | 状态隐式依赖、异常恢复 / 原子参数类型、相似 API、异常请求 | 如启用，仅作评测 |
| **训练集** | **xLAM-60k** | 单轮主力：工具选择、参数填充、并行调用 | SFT 主力 |
| **训练集** | **APIGen-MT-5k** | 多轮 Agent 轨迹：追问、连续调用、Policy 遵循 | SFT 多轮主力（**注意域重叠**） |
| **训练集** | **ToolACE** | 复杂参数、并行、缺参/拒答、**中英双语** | SFT 质量与中文补充 |
| **训练集** | **Glaive V2** | 大规模对话调用、无工具/拒答边界样本 | 清洗后按需抽样补充 |
| **训练集** | **SmolTalk / Magpie 精选** | 通用指令遵循与自然语言对话能力（防过度调工具退化） | 抽取 10%–20% 混合 |

> **关键污染提醒（APIGen-MT-5k vs τ²-Bench）：** `APIGen-MT-5k` 基于 τ-bench 的 **Retail** 与 **Airline** 环境与工具规则合成轨迹。虽然任务实例不同，但在 `τ²-Bench` 上评测时，`Retail / Airline` 属于**同域（In-domain）**迁移，而 **`Telecom` 与 `BFCL V4` 才是严格域外（Out-of-Domain, OOD）泛化指标**。构建数据时务必与测试任务去重，并在汇报结果时区分同域与域外表现。

---

## 2. 候选 SFT 数据集详解（渠道 / 规模 / Schema）

### 2.1 总览对比

| 数据集 | 推荐国内渠道 | 样本数 | 原始体积 | License | 核心作用与短板 |
| --- | --- | --- | --- | --- | --- |
| `Salesforce/xlam-function-calling-60k` | ModelScope | 60,000 | ~98 MB | CC BY 4.0 | 单轮/并行高质量主力；**无 tool response 轮、无拒答** |
| `AI-ModelScope/APIGen-MT-5k` | ModelScope | 5,000 | ~128 MB | CC BY-NC 4.0 | 完整多轮人机+工具交互轨迹；**单条 token 很长，非商用限制** |
| `Team-ACE/ToolACE` | HF-Mirror | 11,300 | ~37 MB | Apache 2.0 | 复杂嵌套参数、缺参/拒答、含中文；**工具定义混在 system 中** |
| `glaiveai/glaive-function-calling-v2` | ModelScope / HF-Mirror | 112,960 | ~271 MB | Apache 2.0 | 规模大、拒答/无工具场景多；**纯文本拼接、噪声多需强清洗** |
| `Magpie-Align/Magpie-Pro-300K-Filtered` | ModelScope | 300,000 | ~563 MB | Apache 2.0 | 通用指令对齐，体积小易抽样 |
| `HuggingFaceTB/smoltalk` | ModelScope | ~1.1M | ~4.15 GB | 混合（以 Apache 2.0 为主） | 综合通用 SFT；**内含 `apigen-80k` 与 xLAM 重叠，禁止全量直接用** |

> 若通用数据采用 `Magpie-Pro`（或只取 `SmolTalk` 的通用子集），全部原始训练数据下载仅约 **1.1 GB**，对服务器 196 GB 磁盘毫无压力。

### 2.2 各数据集原始 Schema 与清洗转换要点

#### ① xLAM-Function-Calling-60k
- **原始字段**：`id (int64)`, `query (str)`, `tools (JSON str)`, `answers (JSON str)`。
- **解析与转换**：`tools` 和 `answers` 需先 `json.loads()`。注意 `parameters` 中类型为简写结构，需转为标准 OpenAI JSON Schema（`{"type": "object", "properties": ..., "required": [...]}`）；`answers` 为调用列表，支持并行调用。
- **注意**：仅有 `user -> assistant(tool_calls)`，**切勿伪造 `tool` 返回消息**。

#### ② APIGen-MT-5k
- **原始字段**：`system (str)`, `tools (JSON str)`, `conversations (list[{from, value}])`。
- **角色映射**：`human -> user`、`gpt -> assistant`、`function_call -> assistant.tool_calls`、`observation -> tool`。
- **注意**：包含大量多轮状态推进，单条 token 数远高于 xLAM；需检查截断长度并校验 `function_call` JSON 合法性。

#### ③ ToolACE
- **原始字段**：`system (str)`, `conversations (list[{from, value}])`（`from` 为 `user` / `assistant` / `tool`）。
- **清洗要点**：**无独立 `tools` 字段**，候选工具列表以描述文本嵌在 `system` 中；assistant 调用可能采用 `[FuncName(arg=val)]` 文本格式。需要解析出结构化 `tools` 和 `tool_calls`（或丢弃无法无损解析的异常样本）。

#### ④ Glaive Function Calling V2
- **原始字段**：`system (str)`, `chat (str)`（整段对话用 `USER:`、`ASSISTANT:`、`<functioncall>`、`FUNCTION RESPONSE:`、`<|endoftext|>` 拼接）。
- **清洗要点**：噪点与格式错误较多，必须用状态机/正则拆分为多轮消息，过滤 JSON 不可解析或函数名不在 `system` 声明中的样本；**重点保留其中的拒答（irrelevant）与缺参追问（clarification）子集**。

#### ⑤ 通用保持数据（SmolTalk 子集 / Magpie-Pro）
- **Magpie-Pro-300K-Filtered**：`uuid`, `conversations`（`human` / `gpt`）。
- **SmolTalk**：`messages`（`role` / `content`）, `source`。**务必按 `source` 过滤**，仅保留 `smol-magpie-ultra`、`everyday-conversations`、`smol-constraints`、`smol-rewrite`、`smol-summarize` 等通用对话，**排除 `apigen-80k`**（避免与 xLAM 重复并打乱配比）。

---

## 3. 国内高速下载命令

在项目根目录（已配置 `HF_ENDPOINT=https://hf-mirror.com`）运行：

```bash
mkdir -p data/raw

# 1. ModelScope 下载主力数据集
.venv/bin/modelscope download --dataset Salesforce/xlam-function-calling-60k --local_dir data/raw/xlam-60k
.venv/bin/modelscope download --dataset AI-ModelScope/APIGen-MT-5k --local_dir data/raw/apigen-mt-5k
.venv/bin/modelscope download --dataset Magpie-Align/Magpie-Pro-300K-Filtered --local_dir data/raw/magpie-pro-300k
# 可选：若优先用 SmolTalk 子集替代 Magpie，再下载 smoltalk
# .venv/bin/modelscope download --dataset HuggingFaceTB/smoltalk --local_dir data/raw/smoltalk

# 2. Glaive V2（先试 ModelScope 镜像；如缺文件则改用下方 HF-Mirror）
.venv/bin/modelscope download --dataset AI-ModelScope/glaive-function-calling-v2 --local_dir data/raw/glaive-v2

# 3. ToolACE（37 MB，直接走 HF-Mirror）
tool-ace的可能链接（modelscope）：modelscope download --dataset mlabonne/ToolACE

HF_ENDPOINT=https://hf-mirror.com runs/envs/train/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Team-ACE/ToolACE", repo_type="dataset", local_dir="data/raw/toolace")
PY
```

---

## 4. 统一训练 Schema 与标准构建流程

### 4.1 统一中间格式（对齐 Qwen3.5 Native FC + TRL）

所有原始数据集清洗后统一转换为如下 JSONL 结构，交给 `Qwen3.5` tokenizer 的 `apply_chat_template(..., tools=sample["tools"])` 渲染，并用 `TRL SFTConfig(assistant_only_loss=True)` 自动只对 assistant 输出（含 `tool_calls` 与最终回复）计算损失：

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
          "function": {
            "name": "get_weather",
            "arguments": {"city": "Beijing"}
          }
        }
      ]
    },
    {"role": "tool", "name": "get_weather", "content": "{\"temperature\": 25}"},
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
    "source": "apigen_mt",
    "task_type": "multi_turn",
    "num_turns": 2,
    "num_tool_calls": 1,
    "has_tool_response": true,
    "is_parallel": false
  }
}
```

> **落地前必做校验**：用 `Qwen3.5-2B` 的 `AutoTokenizer.apply_chat_template` 实测一条样本，确认 `arguments` 传 `dict` 还是 JSON `str`、确认渲染出的 `<tool_call>` 标签与 vLLM `--tool-call-parser qwen3_coder` 解析格式完全一致，且 `assistant_only_loss=True` 严格 mask 掉了 `system`、`user` 与 `tool` 返回内容。

### 4.2 标准数据工程流水线

1. **格式解析与结构化校验**：解析各源数据集，校验 `tools` JSON Schema 合法性、`tool_calls` 参数类型匹配、`tool` 消息前后顺序配对；丢弃解析失败或空回复样本。
2. **去重与污染过滤**：按首轮 query / 对话哈希去重；检查 `APIGen-MT` 与 `τ²` / `BFCL` 测试集无逐题重合。
3. **长度统计与截断过滤**：用 Qwen3.5 tokenizer 统计每条样本总 token 数与 **assistant 有效监督 token 数**；过滤超长异常样本（如 SFT 阶段设定 `max_seq_length=4096` 或 `8192`）。
4. **按“能力桶 + Token 量”配比**：**不按样本条数直接相加**（1 条 `APIGen-MT` 的 token 量 ≈ 5–10 条 `xLAM`），按目标能力桶统计实际 token 占比。

---

## 5. 面向 A100 40GB 的分阶段配比建议

针对单卡 A100 40GB 与 `0.8B / 2B` 小模型，**不要一上来全量灌入 50 万条数据**——小模型参数容量有限，过多嘈杂单轮样本会稀释多轮轨迹信号并拖慢迭代，建议按两步推进：

### 阶段 A：V0 最小全链路跑通版（~5,000 条）

- **目的**：在 10~20 分钟内跑通 `数据构造 → SFTTrainer → 保存权重 → vLLM 启动 → BFCL / τ² 评测` 全流程，验证模板、Loss Mask 与解析器零 Bug。
- **建议抽样**：
  - `xLAM-60k`：1,500 条（含单轮 + 并行）
  - `ToolACE`：1,000 条（含复杂参数 + 中文 + 拒答）
  - `APIGen-MT-5k`：1,000 条（多轮轨迹）
  - `Glaive V2`：500 条（精选拒答 / 缺参追问）
  - `Magpie-Pro`（或 `SmolTalk` 通用子集）：1,000 条

### 阶段 B：V1 正式 SFT 基线版（~40,000–50,000 条精选）

- **目的**：为后续 **OPD** 与 **GRPO** 提供格式稳健、不退化通用对话、且具备基础多轮和拒答能力的强 SFT 起点（2B 模型在 A100 上 BF16 训练 2 epochs 约 1–2 小时级别，便于消融）。
- **建议能力配比（按样本与 Token 平衡控制）**：

| 能力桶 | 数据来源 | 建议样本数 | 占比（按条数参考） | 核心作用 |
| --- | --- | --- | --- | --- |
| 基础与并行调用 | `xLAM-60k`（去重精选） | 15,000–20,000 | ~38% | 稳固 Schema 理解、参数抽取与并行调用 |
| 复杂参数与中文调用 | `ToolACE`（清洗全量） | 8,000–10,000 | ~20% | 增强嵌套 Schema、缺参判断与中文能力 |
| 多轮 Agent 轨迹 | `APIGen-MT-5k`（全量） | 5,000 | ~12%（**Token 占比 ~30%**） | 学习追问、工具反馈理解、多步执行与 Policy |
| 拒答与无工具边界 | `Glaive V2`（专挑拒答/追问子集） | 5,000 | ~11% | 抑制幻觉调工具，提升 `irrelevance` 表现 |
| 通用指令与对话保持 | `Magpie-Pro` 或 `SmolTalk` 通用子集 | 7,000–8,000 | ~17% | 保持自然语言表达与指令理解，防止过拟合工具 |

> **后续演进衔接**：当 V1 SFT 模型在 BFCL Single-turn 和格式合法率上基本达标后，**不再继续堆砌 SFT 样本量**，而是转向用 **OPD（同族大模型 Teacher 蒸馏）** 和 **可验证多轮沙盒 GRPO** 主攻 `BFCL Multi-turn` 与 `τ²-Bench`（特别是 `Telecom` 域外泛化与错误自恢复）。
