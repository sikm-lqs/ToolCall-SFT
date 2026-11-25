# `SFT-v1r1` 数据集构造报告

> 日期：2026-11-24（Asia/Shanghai） · 对应配置：[configs/data/sft_v1r1.yaml](../../configs/data/sft_v1r1.yaml) · 构建产物：`data/processed/sft_v1r1/`

## 1. 重构背景与核心问题修复

在 `SFT-v1` 的 3k 烟囱实验（[01-e0-smoke-run.md](../03-sft/01-e0-smoke-run.md)）中，2B 模型虽然在 `parallel`（50.50% → 89.00%）和 `simple_python`（70.50% → 92.50%）取得显著提升，但在 `irrelevance` 上从 **78.75%（189/240）暴跌至 25.00%（60/240）**，180 道错题全部为 `irrelevance_error:decoder_success`。

对原始 `SFT-v1` 数据做细粒度审计后，定位到四项构造缺陷；本次 **`SFT-v1r1` 在完全保留原 `data/processed/sft_v1/` 不动的前提下**，独立重构造 `data/processed/sft_v1r1/`：

| 原 `SFT-v1` 缺陷 | 实测表现 | `SFT-v1r1` 修复方案 |
| --- | --- | --- |
| **1. ToolACE `no_call` 解析漏网** | 文本中含 Markdown 代码块 / 裸 JSON 调用或 `[No Need to Ask]` 的样本未进 `[...]` 解析分支，被归入 `toolace_no_call` 并进入 loss | 在 [parsers.py](../../src/toolcall/data/parsers.py) 新增 `classify_toolace_no_call()`，隔离丢弃 `no_call_suspect_leak`（86 条）与 `no_call_ambiguous`（65 条） |
| **2. 三种“不调用”行为混为一桶** | `toolace_no_call`（1,538 条）混杂了空工具列表（248 条）、缺参追问（551 条）与工具不适用拒答（588 条），导致真实 `irrelevance` 占比在 27k 中不足 2% | 丢弃无选择意义的 `no_call_empty_tools`（248 条）；将剩余干净样本显式拆分为 **`toolace_missing_param`** 与 **`toolace_irrelevant_refusal`** |
| **3. System Prompt 与回复风格错位** | ToolACE 全量样本保留了原生指令 `"If none of the function can be used, point it out..."`，而 xLAM 与 BFCL 评测均不带此提示，且 BFCL 期望在无合适工具时直接正常回答问题 | 在 [parsers.py](../../src/toolcall/data/parsers.py) 新增 `strip_toolace_policy_prompt()` 剥离通用策略模板，仅保留真实日期/时间锚点；全部样本统一对齐 Qwen3.5 原生 `<tools>` + `<IMPORTANT>` 提示分布 |
| **4. 缺少“直接回答型”难负例供给** | 仅靠 ToolACE 只有 588 条外部 API 拒答，缺乏类似 BFCL `irrelevance_0`（给出不相关工具，用户问常识/数学题时直接作答）的样本 | 采用**做法 A**：新增 [smoltalk.py](../../src/toolcall/data/smoltalk.py)，从已归档 `smoltalk` 抽取纯单/多轮问答，按 TF-IDF 词面重叠挂接 1–4 个高干扰无关 API 工具，构造 **`smoltalk_irrelevant_tools`** 桶 |

---

## 2. 核心工程实现明细

### 2.1 修改与新增文件清单

- **[src/toolcall/data/parsers.py](../../src/toolcall/data/parsers.py)**：
  - 新增 `strip_toolace_policy_prompt()`，剥离 ToolACE 通用函数调用策略句，保留 `Today is ...` / `The current time is ...` 等真实时间上下文。
  - 新增 `classify_toolace_no_call()`，将零调用样本拆分为 `toolace_irrelevant_refusal` 与 `toolace_missing_param`，并隔离 `no_call_empty_tools`、`no_call_suspect_leak` 与 `no_call_ambiguous`。
  - `load_toolace()` 通过可选参数 `strip_policy_prompt` 与 `split_no_call_subtypes`（默认 `False`）接入上述逻辑，保持原 `build_sft_v1.py` 完全兼容。
- **[src/toolcall/data/smoltalk.py](../../src/toolcall/data/smoltalk.py)**：
  - `DistractorToolIndex`：从清洗后的 `xlam-60k + toolace` 汇集 **18,393** 个去重工具 schema，剔除名称含 `math / calculator / algebra / triangle / area / translate / summarize / web_search / python / exec` 等通用计算或文本处理词的工具（避免产生假负例），用倒排索引 + IDF 词面重叠检索高干扰外部 API 工具（并在工具名与问句重合度过高时二次过滤）。
  - `load_smoltalk_irrelevant_records()`：从 `metamathqa-50k`、`openhermes-100k`、`smol-constraints`、`everyday-conversations` 筛选无 `system`、无 `<tool_call>` 泄漏且总字符 `<= 3500` 的干净对话，随机挂载 `1–4` 个（权重 `(1, 1, 2, 3, 4)`）高干扰无关工具，保留原始正常回答作为 assistant 监督目标。
- **[src/toolcall/data/build_sft_v1r1.py](../../src/toolcall/data/build_sft_v1r1.py)** & **[configs/data/sft_v1r1.yaml](../../configs/data/sft_v1r1.yaml)**：
  - 复用 `build_sft_v1.py` 的去重、长度过滤与 SHA-256 落盘工具，完成 `SFT-v1r1` 构建（CLI 入口同步注册于 [pyproject.toml](../../pyproject.toml) 的 `toolcall-build-data-v1r1`）。
- **[tests/test_sft_v1r1_data.py](../../tests/test_sft_v1r1_data.py)**：
  - 覆盖策略句剥离、ToolACE 负例拆桶与泄漏隔离、SmolTalk 干扰工具过滤及 Qwen3.5 多轮 Loss Mask 校验。

### 2.2 数据清洗漏斗统计（详见 [manifest.json](../../data/processed/sft_v1r1/manifest.json)）

| 数据源 | 原始/候选规模 | 过滤/隔离明细 | 去重丢弃 | 净可用池 |
| --- | ---: | --- | ---: | ---: |
| **`toolace`** | 11,300 | 非 JSON 模板 518、空回复 5、空工具 248、疑似调用泄漏 86、歧义 65 | 3 | **10,375**（多轮 792 / 单与并行 8,444 /  irrelevance 588 / 缺参 551） |
| **`xlam-60k`** | 60,000 | 缺失 `required` 参数 275 | 385 | **59,340**（并行 31,269 / 单选 28,071） |
| **`smoltalk_irrelevant`** | 池上限 4,800 | 跳过含 `system`/异常轮/超长（`openhermes` 39,464、`smol-constraints` 372、`everyday` 251、`metamathqa` 8） | 0 | **4,800**（`metamathqa` 1,600、`openhermes` 1,600、`smol-constraints` 800、`everyday` 800） |

> 全部 **74,515** 条干净样本经本地 `Qwen/Qwen3.5-2B` tokenizer 与 `qwen35_chat_template_train.jinja` 校验，`max_seq_length=8192` 下 **0 条超长丢弃、0 条 mask 错误**。

---

## 3. `SFT-v1r1` 最终产物与行为配比

输出目录：`data/processed/sft_v1r1/`

| 产物文件 | 条数 | SHA-256（前 12 位） | 说明 |
| --- | ---: | --- | --- |
| **`train.jsonl`** | **25,015** | `d07544794aba` | 全量训练集（`seed=42` 打乱） |
| **`smoke_v1r1.jsonl`** | **3,000** | `dd01d6e652ca` | 与正式集**行为比例严格对齐**的 3k 烟囱集（无重复采样） |
| **`val.jsonl`** | **600** | `0f280342de86` | 在任何上采样/抽样前分层切出的零泄漏验证集 |
| **`qwen35_chat_template_train.jinja`** | — | `602a5dbfb1f2` | 含 `{% generation %}` 标记的 Qwen3.5 等价模板 |
| **`samples/*.txt`** | 7 份 | — | 7 个子桶的 `[MASK]` vs `[TRAIN]` 可视化核对样本 |

### 分桶行数与可训练 Assistant Token 占比对比

| 能力子桶 | `train.jsonl` 行数(占比) | `train` Asst Token 占比 | `smoke_v1r1` 行数(占比) | `smoke` Asst Token 占比 | `val.jsonl` 行数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `xlam_parallel` | 6,500 (25.98%) | 17.34% | 780 (26.00%) | 17.33% | 130 |
| `xlam_single` | 3,500 (13.99%) | 4.91% | 420 (14.00%) | 4.92% | 80 |
| `toolace_single_or_parallel` | 7,500 (29.98%) | 23.22% | 900 (30.00%) | 23.66% | 150 |
| `toolace_multi_turn` | 2,256 (9.02%)* | 30.09% | 270 (9.00%) | 29.75% | 40 |
| `toolace_missing_param` | 511 (2.04%) | 0.91% | 60 (2.00%) | 1.00% | 40 |
| `toolace_irrelevant_refusal` | 548 (2.19%) | 0.93% | 70 (2.33%) | 0.96% | 40 |
| `smoltalk_irrelevant_tools` | 4,200 (16.79%) | 22.61% | 500 (16.67%) | 22.37% | 120 |
| **合计** | **25,015** | **100.00%** | **3,000** | **100.00%** | **600** |

*\* 注：`toolace_multi_turn` 在 `train.jsonl` 中由去 val 后剩余的 752 条做 3× 上采样得到；`smoke_v1r1.jsonl` 为无重复抽样。*

关键分布特征（`train` vs `smoke_v1r1`）：
- **纯工具不相关负例（`toolace_irrelevant_refusal + smoltalk_irrelevant_tools`）**：`train` 为 **4,748 条（18.98%）** / **23.54% assistant token**；`smoke` 为 **570 条（19.00%）** / **23.33% assistant token**。同时覆盖 **Type 1（无合适工具时直接正常问答）** 与 **Type 2（外部动作无法完成时明确拒答且不输出 `<tool_call>`）**。
- **总非调用边界（含缺参追问）**：`train` **21.02%**、`smoke` **21.00%**（均保证 `num_candidate_tools >= 1`）。
- **并行与多候选能力保留**：`train` 含 **10,287 条（41.12%）** 并行调用、**19,405 条（77.57%）** 多候选工具样本；`smoke` 分别对应 **41.23%** 与 **77.87%**。
- **Token 长度分布**：`train.jsonl` 总长度 `min=314 / p50=773 / p95=1,621 / max=4,956 / mean=870.1` tokens；Assistant 可训练长度 `mean=137.6` tokens（有效监督占比 **15.8%**）。

---

## 4. 校验结果与下一步操作建议

### 4.1 本地已完成校验
1. **自动化单测**：`runs/tools/bfcl/bin/python -m unittest discover -s tests -p "test_sft_v1*.py" -v` **9/9 通过**（涵盖 `test_sft_v1_data.py` 与 `test_sft_v1r1_data.py`）。
2. **Manifest 与 SHA-256 核验**：调用 `toolcall.training.preflight.verify_data_manifest` 核验 `train.jsonl`、`val.jsonl`、`smoke_v1r1.jsonl`、`qwen35_chat_template_train.jinja` 全部通过。
3. **零泄漏与洁净度核验**：
   - `dedup_signature(val) ∩ dedup_signature(train) = 0`，`dedup_signature(val) ∩ dedup_signature(smoke) = 0`。
   - `train`、`val`、`smoke` 三份文件中，ToolACE 策略句 `"If none of the function can be used"` 残留数 **全部为 `0`**；`num_candidate_tools == 0` 样本数 **全部为 `0`**。
4. **渲染样例人工抽检**：已核对 [smoltalk_irrelevant_tools.txt](../../data/processed/sft_v1r1/samples/smoltalk_irrelevant_tools.txt)、[toolace_irrelevant_refusal.txt](../../data/processed/sft_v1r1/samples/toolace_irrelevant_refusal.txt)、[toolace_missing_param.txt](../../data/processed/sft_v1r1/samples/toolace_missing_param.txt)，确认全部按 Qwen3.5 原生工具格式渲染，且 `[TRAIN]` 严格只覆盖 assistant 回复与 `<|im_end|>`。

### 4.2 下一步执行建议（`E0-r1` 烟囱对照）
1. **同步代码与冻结数据包到 `rl-server`**：
   - 使用 `bash scripts/remote/sync_code.sh` 同步代码；
   - 将 `data/processed/sft_v1r1` 打包压缩后 `rsync` 至 `rl-server:/root/ToolCall-SFT/data/processed/` 并校验 SHA-256。
2. **保持超参不变跑 `E0-r1`**：
   - 沿用 [e0_smoke_2b.yaml](../../configs/train/e0_smoke_2b.yaml) 的全部超参（`Qwen/Qwen3.5-2B`、LoRA `r=64`、`lr=1.5e-4`、`1 epoch`），仅将数据切换为 `data/processed/sft_v1r1/smoke_v1r1.jsonl` 与 `val.jsonl`。
3. **扩充快筛放行线**：
   - 在原 BFCL 8 类（1,840 题）基础上加入 `live_irrelevance`（884 题）；
   - 放行标准：`irrelevance >= 76%`、`live_irrelevance >= 66%`（与基线差距不超过 3 个百分点），同时保住 `parallel >= 85%`、`simple_python >= 88%` 增益，过线后再启动 25k `E1-2B-lora`。
