# `SFT-v2-curated` 高质量精简数据集构造报告

> 日期：2026-11-24（Asia/Shanghai） · 数据集名称：**`sft_v2_curated`** · 配置文件：[configs/data/sft_v2_curated.yaml](../../configs/data/sft_v2_curated.yaml) · 产物目录：`data/processed/sft_v2_curated/`

---

## 1. 构造动机：从 2 万条“过拟合跷跷板”转向 5,500 条高密度精选

基于 `Qwen/Qwen3.5-2B` 在 `sft_v1r1_clean`（20,947 条）上的正式训练对比发现：
1. **单轮正例已严重过剩**：2,574 条的 Clean Smoke 仅训 34 步（7 分钟）即达到 `simple_python 89.5%` / `parallel 88.5%`；而 2 万条全量数据中 1.75 万条均为单轮调用，不仅拉长训练至近 2 小时，更在第 1 个 epoch 就开始出现验证损失反升（`eval_loss: 0.2284 → 0.2455`）。
2. **负例偏科触发过度拒调**：`ckpt-274` 虽然将 `irrelevance` / `live_irrelevance` 拉高到 `85.00%` / `79.30%`，却导致 `simple_python` 骤降 9.25 个百分点（`90.75% → 81.50%`），逐题审计确认新增的 **38 条失败全部输出 `[]`（误拒调）**——根因是 `sft_v1r1_clean` 正式集里 `smoltalk` 负例 70%（94/134）集中于 `metamathqa-50k` 数学应用题，模型过拟合后误伤了 `simple_python` 数值计算类工具。
3. **多轮轨迹严重不足且形态单一**：此前仅靠 ToolACE 的 752 条（3× 上采样）无法支撑长程多轮状态推进与工具回包后收敛，`multi_turn_base` 跌至 `15.00%`。

因此 **`sft_v2_curated`** 按照“**宁缺毋滥、控制总规模（5,500 条）、补齐高质量多轮、均衡拒调题材、三键防泄漏**”原则重构，在加速训练 3–4 倍的同时消除正负例跷跷板。

---

## 2. 清洗管线与“宁缺毋滥”质量门控

构建脚本位于 [build_sft_v2_curated.py](../../src/toolcall/data/build_sft_v2_curated.py)（CLI 入口 `toolcall-build-data-v2`），对四大原始数据源执行了以下严格过滤（精确漏斗见 [manifest.json](../../data/processed/sft_v2_curated/manifest.json)）：

### 2.1 四大数据源清洗明细

| 数据源 | 原始规模 | 丢弃 / 隔离明细（宁缺毋滥） | 去重/长度淘汰 | 净可用池 |
| --- | ---: | --- | ---: | ---: |
| **`apigen-mt-5k`**<br>（[parsers.py](../../src/toolcall/data/parsers.py) `load_apigen_mt_5k`） | 5,000 | • 含 `think` 伪工具调用 **426**<br>• `bad_function_call` **199**<br>• 非自然语言结尾 `missing_final_nl` **59**<br>• 连贯角色错位 `consecutive_gpt` **38**<br>• 无真实工具交互 **23**、缺必填参数 **4**<br>• 并从 `tools` 剥离全部未使用的 `think` schema；兼容双层转义 JSON 字符串 `arguments` | 超 `7500` token 截断门控淘汰 **1,295**、mask 异常 **1**（保证与 8K packing 高效适配） | **2,955**<br>（均含真实 `observation` 与自然语言收尾） |
| **`toolace`**<br>（[parsers.py](../../src/toolcall/data/parsers.py) `load_toolace`） | 11,300 | • 非 JSON 模板 **518**、空回复 **5**<br>• 隔离 `no_call_empty_tools` **248**、疑似代码块/文本调用泄漏 `no_call_suspect_leak` **86**、歧义 **65**<br>• 隔离多轮文本含泄漏噪声 **1**<br>• 全量剥离 ToolACE 原生策略提示句（仅保留日期时间锚点） | 跨源去重 **3** | **10,374**<br>（多轮 791 / 并行 4,065 / 单选 4,379 / 拒答 588 / 缺参 551） |
| **`xlam-60k`** | 60,000 | • 缺失 `required` 参数 **275**<br>• 强制 `xlam_single` 满足 `len(tools) >= 2` | 去重 **311** | **51,404**<br>（并行 31,269 / 多候选单选 20,135） |
| **`smoltalk_irrelevant`**<br>（[smoltalk.py](../../src/toolcall/data/smoltalk.py)） | 预核准 **159** + 新采样 **212** | • 优先复用 `sft_v1r1_clean` 人工审核清单（压低 `metamathqa` 至 55 train / 10 val）<br>• 新增部分启用 `DistractorToolIndex` **主题冲突守卫**（从 17,441 个去重安全 API 工具中剔除 `math/calc/weather/currency/stock/time/translate/sql` 等冲突工具）+ `_LIVE_QUERY_RE` 实时问句过滤 + 最短回复长度门控（`>=12` 字符），共过滤 **11** 条 | 去重 **0** | **360**<br>（train 300 + val 60） |

### 2.2 零泄漏与防污染核验

- **Train / Val 三键硬隔离**：先分层切分 `val.jsonl`（500 条），再基于 `normalized_user_turns`、`dedup_signature` 和 `(source, source_id)` 三个键从 `train` 池彻底剔除重叠组，实测 `train_val_group_overlap = 0`。
- **评测集隔离审计**：将 `apigen-mt-5k` 首轮与 `runs/upstream/tau2-sparse/data/tau2/domains/{retail,airline}/tasks.json`（403 条任务指令）以及 `data/eval-inspection/{bfcl-v4,api-bank,tau2-bench}` 全部用户问句做归一化比对，实测 `eval_user_turn_overlap = 0`。
- **关键不变量核验**（运行 `verify_data_manifest` 与全库扫描确认）：
  - `train.jsonl` 与 `val.jsonl` 中含 `think` 工具或调用数 = **`0`**；
  - 含 ToolACE 通用策略句残留数 = **`0`**；
  - `num_candidate_tools == 0` 样本数 = **`0`**；
  - 超过 `8192` 截断样本数 = **`0`**（`train.jsonl` 最大 `7,498` tokens）。

---

## 3. 能力桶配比与 Token 梯度分布

产物路径：`data/processed/sft_v2_curated/`

| 文件 | 条数 | 总 Token（均值 / 最大） | Assistant 监督 Token（均值 / 占比） | SHA-256（前 12 位） |
| --- | ---: | ---: | ---: | --- |
| **`train.jsonl`** | **5,500** | 7,899,727（1,436.3 / 7,498） | 1,113,922（202.5 / **14.10%**） | `4e84896b860e` |
| **`val.jsonl`** | **500** | 725,180（1,450.4 / 7,417） | 91,400（182.8 / **12.60%**） | `a9a919bd0e79` |
| **`qwen35_chat_template_train.jinja`** | — | — | — | `602a5dbfb1f2` |

### 3.1 `train.jsonl`（5,500 条）9 大能力桶精确配比

| 能力大类 | 子桶名 | 行数 (占比) | Assistant Token (占比) | 来源与设计意图 |
| --- | --- | ---: | ---: | --- |
| **多轮状态与工具交互**<br>（合计 **1,350 条 / 24.55%**<br>梯度占比 **67.09%**） | `toolace_multi_turn` | 750 (13.64%) | 343,380 (30.83%) | ToolACE 通用 API 多轮（无上采样，保留 100% 独特轨迹） |
| | `apigen_multi_turn` | 600 (10.91%) | 403,960 (36.26%) | APIGen-MT 净池均衡抽样（**Airline 300 + Retail 300**），补齐长程约束、策略核验、读 `observation` 后自然收尾 |
| **单轮并行调用**<br>（合计 **1,500 条 / 27.27%**） | `xlam_parallel` | 850 (15.45%) | 77,598 (6.97%) | xLAM 多函数/同函数并行调用 |
| | `toolace_parallel` | 650 (11.82%) | 104,338 (9.37%) | ToolACE 复杂嵌套 schema 并行调用 |
| **单轮多候选单选**<br>（合计 **1,500 条 / 27.27%**） | `xlam_single` | 750 (13.64%) | 36,263 (3.26%) | xLAM 多候选（`len(tools) >= 2`）单调用，涵盖科学/计算/API 工具 |
| | `toolace_single` | 750 (13.64%) | 44,532 (4.00%) | ToolACE 复杂嵌套单调用 |
| **工具不适用拒调**<br>（合计 **750 条 / 13.64%**<br>梯度占比 **7.23%**） | `toolace_irrelevant_refusal` | 450 (8.18%) | 26,326 (2.36%) | 外部动作无法用已有工具完成时的清晰拒答（Type 2） |
| | `smoltalk_irrelevant_tools` | 300 (5.45%) | 54,285 (4.87%) | 挂接 1–4 个无关工具时直接正常作答（Type 1）：`openhermes-100k` **145** + `everyday-conversations` **60**（多轮闲聊）+ `metamathqa-50k` **55**（严格限流）+ `smol-constraints` **40** |
| **缺参追问** | `toolace_missing_param` | 400 (7.27%) | 23,240 (2.09%) | 工具适用但缺必填参数时的精准追问 |
| **总计** | **9 个子桶** | **5,500 (100%)** | **1,113,922 (100%)** | 正例 79.09% / 边界负例 20.91%（含 1,554 条并行、4,474 条多候选工具） |

> 注：`val.jsonl`（500 条）覆盖全部 9 个子桶（`apigen_multi_turn` 60 / `toolace_multi_turn` 40 / `xlam_parallel` 70 / `toolace_parallel` 50 / `xlam_single` 60 / `toolace_single` 60 / `toolace_irrelevant_refusal` 50 / `smoltalk_irrelevant_tools` 60 / `toolace_missing_param` 50）。

---

## 4. 质量抽检与样例 Case

### 4.1 自动化测试与人工抽检结果
1. **单元测试**：运行 `runs/tools/bfcl/bin/python -m unittest discover -s tests -p "test_sft_v*.py" -v`，**14/14 全部通过**（含新增的 [test_sft_v2_curated_data.py](../../tests/test_sft_v2_curated_data.py)）。
2. **`smoltalk_irrelevant_tools` 全量/抽样复查**：
   - `train.jsonl` 300 条中，**95 条**直接来自已人工审核的 `v1r1_human_approved` 清单；其余 **205 条**经 `v2_strict_guarded` 守卫生成。
   - 对全部 `247` 条新生成记录（train 205 + val 42）执行自动化冲突扫描，**0 条**出现主题冲突（weather/currency/stock/time/translate/math 撞车）；抽检前 20 条样例均确认为高质量高干扰负例（如 Java API 设计问答挂载 `JavaDBDocManager.createDatabaseSchema`、拉普拉斯变换挂载 `Standard_Deviation` 等，模型均直接给出正确解答而未误调工具）。
3. **渲染检查**：[samples/](../../data/processed/sft_v2_curated/samples/) 下已保存全部 9 个子桶的 `[MASK]` vs `[TRAIN]` 渲染文本，确认所有多轮与单轮样本仅 assistant 回复体与 `<|im_end|>` 参与梯度计算。

### 4.2 代表性样例 Case（均来自 `samples/*.txt` 实测渲染）

1. **`apigen_multi_turn`**（[apigen_multi_turn.txt](../../data/processed/sft_v2_curated/samples/apigen_multi_turn.txt)，总长 `5,472` tokens / 监督 `787` tokens）：
   - 挂载 13 个航空客服工具（已剔除 `think`）与 `# Airline Agent Policy`；
   - 第 1 轮先向用户追问 user ID、reservation ID 和取消原因；第 2 轮拿到信息后调用 `get_reservation_details(reservation_id="683OBU")`；读取 `<tool_response>` 后依政策核验并生成最终回复收尾。
2. **`smoltalk_irrelevant_tools`（Type 1 直接正常作答）**（[smoltalk_irrelevant_tools.txt](../../data/processed/sft_v2_curated/samples/smoltalk_irrelevant_tools.txt)，`875` tokens / 监督 `151` tokens）：
   - 挂载无关工具 `weightmanagement.meal_planner`、`legalDocumentAnalysis`；
   - 4 轮日常对话（冰箱异常噪音排查）全程保持自然回答，无任何 `<tool_call>` 幻觉或多余“缺少工具”套话。
3. **`toolace_irrelevant_refusal`（Type 2 外部 API 拒答）**（[toolace_irrelevant_refusal.txt](../../data/processed/sft_v2_curated/samples/toolace_irrelevant_refusal.txt)）：
   - 挂载 3 个新闻/票房查询工具，用户要求抓指定 `detik.com` URL 网页内容，无 ToolACE 策略模板，模型直接说明现有工具不支持抓取 URL。
4. **`toolace_missing_param`（缺参追问）**（[toolace_missing_param.txt](../../data/processed/sft_v2_curated/samples/toolace_missing_param.txt)）：
   - 挂载 `verifyHash(data, hash_algorithm, hash_value)`，用户只给了 `"Hello World"`，模型准确追问缺失的 `hash_algorithm` 和 `hash_value`。
5. **`toolace_parallel` / `xlam_single` / `toolace_multi_turn`**：
   - 完整渲染样例见 [toolace_parallel.txt](../../data/processed/sft_v2_curated/samples/toolace_parallel.txt)、[xlam_single.txt](../../data/processed/sft_v2_curated/samples/xlam_single.txt)、[toolace_multi_turn.txt](../../data/processed/sft_v2_curated/samples/toolace_multi_turn.txt)。

---

## 5. 训练参考参数与预期

- 7.90M 总 token 在 8K `packing=True` 下约打包为 **~970 条 packed 序列**；按 `bs=1, accum=8`（有效 `8` 序列/step），**1 epoch ≈ 121 步、2 epochs ≈ 242 步**（A100 上预计 **~45 分钟**即可完成 2 epochs 全程，每 `0.25` 即 ~60 步保存一个 checkpoint）。
- 后续同步至 `rl-server:/root/ToolCall-SFT/data/processed/sft_v2_curated/` 后，建议对 `0.5 / 1.0 / 1.5 / 2.0 epoch` 四个 checkpoint 跑统一 **BFCL 9 类 + API-Bank full** 对比验收。
