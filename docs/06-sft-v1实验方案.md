# 6. `SFT-v1` 数据集构建记录与实验方案

本文档完整记录第一版监督微调数据集 **`SFT-v1`**（仅基于 `xlam-60k` + `toolace`）的本地工程构建过程、格式标准化与深坑排查、本地 `Qwen3.5-2B` Tokenizer / Loss Mask 实测验证，以及后续在单卡 **A100-SXM4-40GB** 上执行的 `V0 → SFT-v1` 实验方案。

---

## 1. 工程决策：本地构建混合 vs 远程 GPU 机清洗

**结论：在本地完成全部解析、标准化、去重、配比混合与 Tokenizer 强校验，再将代码进 Git、将冻结的数据产物同步到远程 GPU 服务器直接开训，是最合理的工程实践。**

| 维度 | 本地清洗混合 + 冻结产物同步（采用） | 远程 GPU 服务器现场下载与清洗 |
| --- | --- | --- |
| **算力成本** | 数据清洗与全量 7 万条分词校验纯 CPU 完成，**零占用付费 A100 租期** | 占用 GPU 实例时长做正则/JSON 清洗与排错 |
| **可复现性** | 产出固定 `manifest.json`（含源文件与输出文件逐一 SHA-256），与本地 `Qwen3.5-2B` tokenizer 严格绑定，杜绝 ModelScope `master` 漂移 | 若重新拉取上游，不保证与本地审计完全一致 |
| **本地闭环优势** | 本地已完整下载原始快照与 `Qwen/Qwen3.5-2B` tokenizer / `trl==1.18.1`，可直接完成渲染核对、Loss Mask 单元测试与样本抽检 | 需要等到远程搭好环境才能暴露模板/Schema Bug |
| **Git 仓库洁净度** | 代码（`src/toolcall/data/`）、配置（`configs/data/sft_v1.yaml`）、测试（`tests/`）进 Git；`data/processed/sft_v1/`（85 MB，压缩后 ~12 MB）通过 `rsync`/`scp` 传至服务器并用 SHA-256 校验，不污染 Git 历史 | 若混入大 JSONL 进 Git 会拖慢仓库克隆 |

---

## 2. 原始数据集 Shape 实测、清洗规则与 Schema 标准化

所有构建脚本位于 `src/toolcall/data/`（配置文件 [configs/data/sft_v1.yaml](../configs/data/sft_v1.yaml)），入口命令为 `.venv/bin/toolcall-build-data`（本地亦可用 `PYTHONPATH=src runs/tools/bfcl/bin/python -m toolcall.data.build_sft_v1`）。

### 2.1 两大原始数据源 Shape 实测对比

| 数据源 | 本地文件与 SHA-256（前 12 位） | 原始行数与物理 Shape | 原始格式特征与异构点 |
| --- | --- | --- | --- |
| **`xlam-60k`** | `data/raw/xlam-60k/xlam_function_calling_60k.json`<br>（92 MB，`4ef5c6f0dc55`） | **60,000 行**<br>4 列：`id`、`query`、`tools`(JSON str)、`answers`(JSON str) | 1. 无 `system` 与 `tool` 返回轮，全为单轮 `user -> assistant`。<br>2. `parameters` 为**扁平字典** `{param: {type, description, default?}}`，**没有 `properties` 与 `required` 数组**。<br>3. `type` 含非标准 Python 类型标注，如 `"str, optional"`、`"int, optional"`、`"List[int]"`、`"Dict"`、`"Callable[[float], float]"`。 |
| **`toolace`** | `data/raw/toolace/data/train-00000-of-00001.parquet`<br>（11 MB，`d86417edb8a4`） | **11,300 行**<br>仅 1 列：`conversations[{from, value}]` | 1. **无独立 `tools` 列**：工具列表以 JSON 数组混排在 `conversations[0]`（`system`）文本中，且附带 `[func1(params=val), ...]` 旧格式指令与 `NO other text MUST be included.`。<br>2. **518 行异构非 JSON 模板**：使用 XML `<tool_name>` 或 `{Get Order:("id"/'...')}` 语法。<br>3. `parameters.type` 使用 `"dict"` / `"int"` / `"float"` 而非标准 JSON Schema 类型。<br>4. 部分函数名含**空格、括号或短横线**（如 `"User Feed (Video Posts) V2"`），部分参数名为 **Python 保留字**（如 `from="2025-01-01"`）。<br>5. `tool` 消息为 JSON 数组 `[{"name": ..., "results": ...}, ...]`，并行调用时单条 `tool` 消息可包含多个工具结果。 |

### 2.2 清洗、反序列化与标准化管线（四道工序）

1. **反序列化与解析（`src/toolcall/data/parsers.py`）**：
   - **`xlam-60k`**：`json.loads(tools/answers)`；过滤缺失参数或无法解析的行；仅保留原始 `user -> assistant(tool_calls)`，**不伪造 `tool` 返回轮**。
   - **`toolace`**：
     - 以 `"Here is a list of functions in JSON format that you can invoke:"` 为锚点，使用 `json.JSONDecoder().raw_decode` 精准切出标准 JSON 工具列表（自动跳过 518 行非 JSON 异构提示格式）。
     - 从 `system` 中彻底剥离旧版 `[func1(...)]` 格式指令与 `"You should only return the function call in tools call sections."`，仅保留拒答/追问行为约束及可选语言提示（防止与 Qwen3.5 的 `<tool_call>` XML 系统提示冲突）。
     - 用**已知工具名最长前缀匹配 + 引号/括号深度感知分割器**解析 `[func(...)]` 调用，完美兼容带空格/括号的函数名（如 `User Feed (Video Posts) V2(...)`）和 Python 关键字参数名（如 `from="..."`）——全量 `9,964` 条 ToolACE 调用语句 **100% 无损解析**。
     - 将每条 `tool` 消息中的 `[{"name": ..., "results": ...}]` 拆分为对应的标准 `{"role": "tool", "name": ..., "content": ...}` 消息序列。
2. **Schema 与函数名标准化（`src/toolcall/data/schema.py`）**：
   - **xLAM 扁平参数转标准 JSON Schema**：将 `type` 中的 `, optional` 或存在 `default` 的字段判为可选，其余写入 `required: [...]`；映射 `str->string`、`int->integer`、`float->number`、`bool->boolean`、`List[T]->{"type":"array","items":...}`、`Dict->object`。
   - **ToolACE 递归 JSON Schema 修正**：仅沿着 `properties`、`items`、`additionalProperties`、`anyOf/oneOf/allOf` 递归把 `"dict"->"object"`、`"int"->"integer"`、`"float"->"number"`（特别保护名叫 `"type"` 的业务参数不被误改）。
   - **函数名清洗与冲突消解**：非 `[A-Za-z0-9_.-]+` 字符统一替换为 `_` 并同步映射到 `tools`、`tool_calls` 与 `tool.name`（实测碰撞次数为 `0`）。
3. **结构强校验与去重**：
   - 校验每条样本：所有被调用函数必须在 `tools` 中存在；`arguments` 必须为 Python `dict`（满足 Qwen3.5 模板 `|items` 要求）；所有 `required` 参数必须出现；`tool` 消息必须紧跟含 `tool_calls` 的 `assistant`；对话必须以 `assistant` 结束。
   - 按 `user` 轮文本列表 + 排序工具名生成 SHA-256 签名跨源去重（优先保留 `toolace_multi_turn > toolace_no_call > toolace_single_or_parallel > xlam`）。
4. **分词长度门控（`max_seq_length = 8192`）**：
   - 使用本地 `Qwen/Qwen3.5-2B` tokenizer 渲染并统计全长与 Assistant 可训练 token 数，超过 `8192` 的样本直接过滤（实测最大长度为 `5028`，**`0` 条截断丢弃**）。

### 2.3 值得深挖的 4 个核心工程坑点（本轮实测暴露并解决）

> [!IMPORTANT]
> 以下四个问题均在本次本地实测中真实触发；若未提前排查，直接开训会导致**零梯度 / 错位训练 / 样本损坏**：

1. **Qwen3.5 原生 `chat_template.jinja` 缺少 `{% generation %}` 标记**
   - **实测现象**：直接对本地 `Qwen/Qwen3.5-2B` 调用 `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)` 会报警 `chat template does not contain {% generation %} keyword`，返回的 `assistant_masks` **全为 0**；在 TRL `SFTConfig(assistant_only_loss=True)` 下会直接触发 `RuntimeError` 或全 `-100` 零梯度。
   - **解法**：编写 `patch_qwen35_chat_template()`（[src/toolcall/data/template.py](../src/toolcall/data/template.py)），在原生模板的 `assistant` 分支输出体与 `<|im_end|>` 外包裹 `{%- generation %}...{%- endgeneration %}`，将 `<|im_start|>assistant\n<think>\n\n</think>\n\n` 与末尾换行保留在 mask 外。
   - **验证**：导出为 [qwen35_chat_template_train.jinja](../data/processed/sft_v1/qwen35_chat_template_train.jinja)，与官方原模板在每类样本上的渲染文本 **100% 逐字节一致**。
2. **多用户轮对话的 `<think>` 模板非对称性**
   - **源码事实**：`Qwen3.5` 官方模板通过 `ns.last_query_index` 定位最后一个真实 `user` 问句；只有 `loop.index0 > ns.last_query_index` 的 `assistant` 轮会插入 `<think>\n\n</think>\n\n`，此前历史轮次无 `<think>` 标签。
   - **影响**：ToolACE 多轮中约有 `509` 条含 2–5 轮人类交互。统一在 `{%- if loop.index0 > ns.last_query_index %}` 之后再开启 `{%- generation %}`，确保两种轮次都只监督真实模型回复与 `<|im_end|>`。
3. **HuggingFace `datasets.load_dataset("json")` 的 PyArrow Struct 键合并污染**
   - **实测现象**：由于不同样本的 `tool_calls[*].function.arguments` 和 `tools[*].function.parameters.properties` 键集合完全不同，直接调用 `load_dataset("json")` 会让 PyArrow 推断出包含全库所有参数名的巨型 Struct，向每条样本的 `arguments` 注入大量 `None` 键，导致 Qwen3.5 模板把 `<parameter=无关字段>None</parameter>` 渲染进训练序列。
   - **解法**：磁盘保留干净可读的标准 OpenAI JSONL；训练加载统一走 [load_tokenized_sft_dataset()](../src/toolcall/data/loader.py)，用标准 `json.loads` 逐行读取并预先转为扁平整数列表列 `Dataset({"input_ids": ..., "labels": ...})`，已在单测中验证与 TRL `pack_dataset` + `DataCollatorForLanguageModeling(padding_free=True)` 完全兼容。
4. **防泄漏切分顺序**
   - 验证集 `val.jsonl`（500 条）必须在 `toolace_multi_turn` `3×` 上采样**之前**按分层比例切出；经 SHA-256 签名比对，训练集与验证集交集严格为 `0`。

---

## 3. `SFT-v1` 最终构建统计与配比逻辑

### 3.1 清洗过滤与分桶漏斗（实测数据，详见 `manifest.json`）

- **`toolace`**：原始 `11,300` → 丢弃非 JSON 模板 `518`、空 assistant 回复 `5`、去重 `3` → 净可用 **`10,774`**（多轮 `792` / 无调用 `1,538` / 单轮或并行 `8,444`）。
- **`xlam-60k`**：原始 `60,000` → 丢弃缺失 `required` 参数 `275`、去重 `385` → 净可用 **`59,340`**（多调用 `31,269` / 单调用 `28,071`）。

### 3.2 目录产物与最终配比（`data/processed/sft_v1/`）

| 产物文件 | 条数 / 大小 | 构成说明 |
| --- | --- | --- |
| **`train.jsonl`** | **27,058 条**（75 MB） | 见下表 5 桶混合（已 Shuffle，`seed=42`） |
| **`val.jsonl`** | **500 条**（1.4 MB） | 分层无泄漏验证集（多轮 40 / 0 调用 40 / ToolACE 140 / xLAM 并行 180 / xLAM 单选 100） |
| **`smoke_v0.jsonl`** | **3,000 条**（8.4 MB） | V0 烟囱测试子集（ToolACE 1,500 + xLAM 1,500，无重复采样） |
| **`qwen35_chat_template_train.jinja`** | 7.7 KB | 含 `{% generation %}` 标记的训练专用等价模板 |
| **`manifest.json`** | 6.6 KB | 源/目标 SHA-256、清洗统计、分桶配比与 token 分位分布 |
| **`samples/*.txt`** | 5 个样本（32 KB） | 5 类桶的 `[MASK]` vs `[TRAIN]` 可视化渲染核对样例 |

**`train.jsonl` 最终配比与为什么这样配：**

| 子桶 | 净池可用 | `SFT-v1` 实际条数（占比） | 配比理由（紧扣基线短板） |
| --- | --- | --- | --- |
| `toolace_multi_turn` | 752（去 val 后） | **2,256（8.3%）**（`3×` 上采样） | 含 `tool` 返回轮的最稀缺高价值信号；`3×` 上采样避免被单轮淹没，同时乘 `2 epoch = 6 pass` 不至于过度背诵 |
| `toolace_no_call` | 1,498（去 val 后） | **1,498（5.5%）**（全留） | 守住 `irrelevance` / `live_irrelevance` 拒答与缺参追问底线，抑制 API-Bank 回复轮泄漏 `<tool_call>` |
| `toolace_single_or_parallel` | 8,304（去 val 后） | **8,304（30.7%）**（全留） | 提供复杂嵌套 `object/array` Schema、多约束参数与自然语言收尾 |
| `xlam_parallel` | 31,089（去 val 后） | **10,000（37.0%）**（随机抽样） | 主攻基线最弱的单轮能力之一 `parallel`（2B `50.5%` / 0.8B `31.5%`），但限幅 10k 防单轮必调压垮拒答率 |
| `xlam_single` | 27,971（去 val 后） | **5,000（18.5%）**（优先多候选工具） | 优先抽取 `len(tools) >= 2` 样本巩固 `multiple`，与 ToolACE 并行形成 `52.3%`（14,159 条）并行、80.4% 多候选覆盖 |

**Token 长度实测分布（`Qwen/Qwen3.5-2B` tokenizer）：**
- `train.jsonl` 总长度：`min=122` / `p50=758` / `p95=1,545` / `p99=2,193` / `max=5,028` / `mean=843.6` tokens。
- `train.jsonl` 可训练 Assistant 长度：`min=7` / `p50=74` / `p95=353` / `p99=814` / `max=2,079` / `mean=116.6` tokens（约占总 token 的 **13.8%**，再次证明精确 `assistant_only_loss` 对防止 86% prompt token 稀释梯度的必要性）。

---

## 4. `SFT-v1` 实验方案（2B 与 0.8B）

### 4.1 实验目的与预期验证假设

1. **主假设 H1（格式与并行修复）**：仅靠纯净的 `xlam-60k + toolace`（27k 精选）微调，即可显著拉升 `Qwen3.5-2B` 与 `0.8B` 的 `parallel`、`parallel_multiple`、`multiple` 和 `simple_python` 准确率，减少 JSON/类型/值忠实度错误。
2. **主假设 H2（边界保护）**：保留 1,498 条 `no_call` 样本 + 2 epoch 控轮数 + LoRA `r=64`，应能在提升调用率的同时守住 `irrelevance` / `live_irrelevance`，并降低 API-Bank `response_leaked_tool_call`（基线 106/79 次）。
3. **消融基点 H3（多轮天花板探针）**：定量观察仅靠 752 条去重多轮（`3×`）能把 `multi_turn_base` / `long_context` 拉高多少，以此客观决定下一步是直接进 **OPD / GRPO**，还是先做 **`SFT-v2`（+ `apigen-mt-5k` + 通用无工具对话）** 消融对照。

### 4.2 同步数据到远程服务器的标准操作

本地已通过 `tests/test_sft_v1_data.py`（5/5）与零泄漏验证。将代码推上 Git 后，按以下方式无损同步冻结数据至远程 A100 服务器：

```bash
# 本地打包（约 12 MB）并传到远程仓库同级目录
tar -I 'gzip -9' -cf sft_v1.tar.gz -C data/processed sft_v1
scp sft_v1.tar.gz <remote>:<repo>/data/processed/

# 远程解压并核验 SHA-256（4 个核心文件应全部 OK）
cd <repo>/data/processed && tar -xzf sft_v1.tar.gz && rm sft_v1.tar.gz
python3 - <<'PY'
import hashlib, json
from pathlib import Path
base = Path("sft_v1")
manifest = json.loads((base / "manifest.json").read_text())
for fname, info in manifest["outputs"].items():
    digest = hashlib.sha256((base / fname).read_bytes()).hexdigest()
    assert digest == info["sha256"], f"SHA mismatch on {fname}"
    print("OK:", fname, digest[:12])
PY
```

### 4.3 实验序列与参数设置（A100-SXM4-40GB，`runs/envs/train`）

| 实验编号 | 目标模型 | 数据文件 | 核心超参 | 预期耗时与目的 |
| --- | --- | --- | --- | --- |
| **`E0-smoke`** | `Qwen3.5-2B` | `smoke_v0.jsonl`（3k） | LoRA `r=64, alpha=128`, `lr=1.5e-4`, `bs=1, accum=8`（eff 8×8192≈65K tok/step）, `grad_ckpt=True`, `max_len=8192`, `packing=True`, `bf16`, `1 epoch` | **<10 分钟**打通训练、SwanLab 上报、LoRA merge、vLLM 启动与 8 类 BFCL 评测全链路 |
| **`E1-2B-lora`** | `Qwen3.5-2B` | `train.jsonl`（27k）+ `val.jsonl`（500） | 同上，`num_train_epochs=2`, `cosine`, `warmup_steps=0.05`, `weight_decay=0.01`, `max_grad_norm=1.0`, `eval_steps=0.25 / save_steps=0.25`（精确每 `0.5` epoch）, `save_total_limit=4` | **`SFT-v1` 主力实验**；在 `0.5 / 1.0 / 1.5 / 2.0 epoch` 挑选最佳 ckpt |
| **`E2-0.8B-lora`** | `Qwen3.5-0.8B` | 同上（或先跑同配比对照） | 同上，优先复现同一配比做跨尺寸对比；若 0.8B 欠拟合可设对照实验将 `lr` 调至 `2e-4` 或 `xlam` 扩至 18k | 检验小尺寸 0.8B 的 SFT 增益上限与后期接 OPD 的基座质量 |
| **`E3-2B-full`** *(可选对照)* | `Qwen3.5-2B` | 同上 | Full SFT，`lr=1.5e-5`, `gradient_checkpointing=True`, `epochs=2`, `save_total_limit=2` | 验证在 27k 纯工具数据下 LoRA 是否已接近全参上限，同时对比 `irrelevance` 保持能力 |

> **Packing 与大词表显存校准**：`Qwen3.5` 词表高达 `248,320`；若 `packing=True` 且 `max_length=8192` 下设 `bs=2`，单卡微批次 logits 需占用 `16384 × 248320 × 4B ≈ 16.3 GB`，极易触及 40 GB 上限；同时每步吃太多 token 会把 27k 数据（~3,000 packed 序列）压缩到几十步。因此配置固定为 **`per_device_train_batch_size=1, gradient_accumulation_steps=8, gradient_checkpointing=True`**（有效 ~65K tokens/step，约 ~375 steps/epoch、~750 steps/2 epochs），兼顾显存安全与充分迭代。

### 4.4 训练期观测指标与异常判断（SwanLab + Trainer Logs）

| 指标 | 正常健康范围 | 异常信号与排查方向 |
| --- | --- | --- |
| **`train/loss`** | 起步 ~`0.8–1.8`（纯 assistant token），平滑单调下降至 `0.25–0.55` | 若起步 >`3.0`：模板格式与基座不一致（检查 `enable_thinking` 与 `<tools>` 换行）；若掉到 <`0.05`：过拟合或序列截断 |
| **`eval/loss`**（`val.jsonl`） | 随训练前 `1.0–1.5 epoch` 持续下降并趋稳 | 若 `1.0 epoch` 后明显抬头而 `train/loss` 继续骤降：说明 `2 epoch` 过拟合，直接回退 `1.0` 或 `1.5 epoch` ckpt |
| **`train/mean_token_accuracy`** | 从基线逐步升至 `0.85–0.93` | 若直接 `>0.97` 且 loss 极低：怀疑 mask 失效，把大段模板死文本当成训练目标 |
| **`train/grad_norm`** | 平稳在 `0.2–1.5`，偶有小毛刺 | 连续尖峰 `>5` 或 `NaN`：检查学习率、数据中是否存在极短（仅 1–2 个有效 label）异常 batch |
| **`num_tokens` / `labels != -100` 占比** | 单 batch 内有效监督 token 占比约 **12%–18%** | 若占比突变为 ~`100%`：说明 `assistant_masks` 失效退化为全序列 loss；若为 `0%`：batch 被全 mask |
| **GPU 显存与吞吐** | LoRA BF16 (`max_len=8192`, `bs=2`) 预计峰值显存 <`24 GB` / 40 GB | 若 `flash-linear-attention` 未生效会导致 step 时间慢数倍，开训前务必跑环境自检 |

### 4.5 Loss Mask 四重硬校验方案（开训前必过）

1. **静态单元测试（已落盘）**：运行 `tests/test_sft_v1_data.py`（使用本地 `runs/tools/bfcl/bin/python -m unittest discover -s tests -v`），验证 xLAM/ToolACE 规范化、Arrow 加载不串键、`enable_thinking=False` 字节一致性与多轮 mask 边界。
2. **人工可读样本检查（已落盘）**：抽查 `data/processed/sft_v1/samples/*.txt`，确认只有 `<tool_call>...</tool_call><|im_end|>` 与助理文本回复处于 `[TRAIN (labels != -100)]`，其余全部 `[MASK (labels == -100)]`。
3. **训练器入口 Batch 实测断言（开训前 0 step 必须执行）**：
   从 `SFTTrainer.get_train_dataloader()` 取前 2 个真实 batch（含 `packing=True` 后的 `DataCollatorForLanguageModeling(padding_free=True)` 输出），执行：
   ```python
   # 1. 确认存在大量 -100 屏蔽与 10%~25% 有效监督 token
   valid = batch["labels"] != -100
   assert 0.05 < valid.float().mean().item() < 0.40
   # 2. 按连续 valid 区间解码，断言绝不包含系统/用户/tool_response 边界标记
   ```
4. **`E0-smoke` 推理回归测试**：训练完 3,000 条后 merge 模型并用 vLLM `--tool-call-parser qwen3_coder` 启动，确认输出格式可被 `qwen3_coder` 正常解析成 `tool_calls` 而不会泄漏 raw `<tool_call>` 到 `content`。

### 4.6 训后评测与晋级决策流程

1. **Step 1：LoRA Merge 与推理启动**
   - 用 `PeftModel.from_pretrained(...).merge_and_unload()` 合并到纯文本模型，再放回官方 `Qwen3_5ForConditionalGeneration` 外层导出 `runs/train/<exp>/merged-<ckpt>/`；保留原始推理模板和处理器文件，服务使用 `--language-model-only`。
   - 用与基线**完全一致**的 vLLM 配置启动（64K 上下文、`--tool-call-parser qwen3_coder`、`--enable-prefix-caching`、`chat_template_kwargs: {enable_thinking: false}`）。
2. **Step 2：Checkpoint 快筛（8 类高性价比 BFCL 子集，1,840 题，每次数分钟）**
   ```bash
   .venv/bin/toolcall-eval bfcl-v4 --categories \
     simple_python multiple parallel parallel_multiple irrelevance \
     multi_turn_base multi_turn_miss_func multi_turn_miss_param \
     --variant sft-v1-ckpt --model "$merged_model" --base-url http://127.0.0.1:18003/v1 \
     --concurrency 128 --timeout 300 --gpu-index 0
   ```
   - 比较 `0.5 / 1.0 / 1.5 / 2.0 epoch` 四个 checkpoint 在 8 类上的得分，**兼顾工具调用提升与 `irrelevance` 不掉点**选出最佳 checkpoint。
3. **Step 3：最佳 Checkpoint 正式全量验收**
   - 补跑完整 **`BFCL --profile core`（4,441 题）** 与 **`API-Bank --profile full`（1,281 步）**，并与 [01-baseline-report.md](../reports/01-baseline/01-baseline-report.md)（2B `core_weighted=43.48%` / `API-Bank=66.53%`；0.8B `34.30%` / `57.24%`）逐类对照。
4. **Step 4：下一阶段分叉决策**
   - **情况 A（符合预期）**：`parallel` / `parallel_multiple` / `multiple` / `simple_python` 明显大幅高于基线，`irrelevance` 与 `response_leaked_tool_call` 持平或改善 → 冻结当前模型为 **`SFT-v1` 成品基座**，直接进入 **OPD + Agentic GRPO** 攻克 `multi_turn_*`（尤其是 `miss_func` 2.5% / `miss_param` 15%）。
   - **情况 B（暴露短板）**：若发现 `irrelevance` / `live_irrelevance` 退化超过 `3%`，或 `multi_turn_base` 几乎停滞 → 立即启动 **`SFT-v2` 消融实验**（保持训练框架不变，仅在数据中增补本地已备好的 `apigen-mt-5k` + `smoltalk` 通用无工具子集），直接对比 `SFT-v1` vs `SFT-v2` 数据收益。
