# Qwen3.5-2B `SFT-v3` 全 Checkpoint 快评与全历史横向综合分析

> 日期：2026-11-24（Asia/Shanghai；Run ID 中时间戳为 UTC）
> 训练配置：[sft_v3_2b_lora.yaml](../../configs/train/sft_v3_2b_lora.yaml) · 数据文档：[05-sft-v3-dataset-report.md](../02-data-audit/05-sft-v3-dataset-report.md)

---

## 1. `SFT-v3` 训练概况与 Loss 轨迹

本次微调基于严格清洗后的 **`sft_v3`** 数据集（`4,590` 条训练、`440` 条验证，零跨切分/零历史验证/零 `τ²` 测试实体重合），在单张 A100-40GB 上从 `Qwen/Qwen3.5-2B` 底座以 LoRA（`r=64, alpha=128, dropout=0.05, all-linear`）训练 **2 个 Epoch**：

- **打包与加速配置**：`8K BFD packing` + `padding_free=true` + `gradient_checkpointing=false`，`bs=1 × accum=8`（有效 batch = 8 条 packed 序列 ≈ 65K tokens/step）。
- **实际运行表现**：4,590 条训练样本打包为 **670 条**序列，共 **168 步**（84 步/epoch）；关闭梯度检查点后总训练耗时 **1,608.0 秒（26.8 分钟）**，PyTorch 峰值显存 **`33.192 GiB`**，零 OOM；预检 loss-mask 与 GPU 前向检查均通过（SwanLab 训练运行：[`tgm380ca`](https://swanlab.cn/@sikm/toolcall-sft/runs/tgm380ca)）。

| Checkpoint | Epoch | Train Loss (邻近步) | Val `eval_loss` | Val Token Acc | Val Entropy | Adapter SHA-256（前 12 位） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **`checkpoint-34`** | 0.4060 | 0.3667 (`@30`) | 0.35825 | 89.31% | 0.3636 | `cdfaaa0d7573` |
| **`checkpoint-68`** | 0.8119 | 0.4679 (`@65`) | **0.34161（最低）** | 89.73% | 0.3420 | `7b749bfb675b` |
| **`checkpoint-102`** | 1.2149 | 0.3357 (`@100`) | 0.34587 | 89.82% | 0.2793 | `27c1586f34d4` |
| **`checkpoint-136`** | 1.6209 | 0.2857 (`@135`) | 0.34444 | 90.02% | 0.2590 | `b7cc2e37f1eb` |
| **`checkpoint-168`** | 2.0000 | 0.2721 (`@165`) | 0.34274 | 90.02% | 0.2604 | `dc33b9ca0421` |

**训练曲线信号**：`eval_loss` 在 **`checkpoint-68`（0.81 epoch）** 达到全局最低 `0.34161`；此后进入第 2 轮，训练 loss 持续下行至 `0.2721`，但 `eval_loss` 再未低于第 68 步，提示第 1.0–2.0 轮存在明显的静态边界拟合增强倾向。

---

## 2. 全项目已快评权重横向对比（同口径对齐）

> **评测口径一致性核验**：`sft-v3` 全部 5 个 checkpoint 均在相同的 vLLM 0.19.1 官方封装（`--language-model-only --max-model-len 65536 --tool-call-parser qwen3_coder`，`temperature=0`、关思考、`max_tokens=1024`）下通过「结构化调用 + 回包收尾」双探针后串行评测；BFCL 9 类均完成 **2,724 题**（0 请求失败、0 上下文拒绝，仅 `ckpt-68/102/136/168` 分别有 8/5/5/6 次命中 1024 输出上限），API-Bank 均完成 **1,281 步**（753 调用 + 528 回复，0 失败、0 拒绝、0 截断）。
>
> **历史对照口径提示**：`2B 基座` BFCL 九类分项与微平均摘自 [基线报告](../01-baseline/01-baseline-report.md) 的 `core` 17 类完整跑批（非单独 9 类负载）；`旧 E0` 仅测 8 类（不含 `live_irrelevance`，未测 API-Bank）；`Clean Smoke` 与 `v1r1-clean` 源自上一版清洗数据，`v3` 与 `v1r1` 验证集彼此严格隔离。

### 2.1 BFCL 同题快筛对比（9 类 2,724 题，与预设四条行为门槛）

| 权重 / 阶段 | Epoch | 九类微平均 | `irrelevance` (≥76%) | `live_irrelevance` (≥66%) | `simple_python` (≥88%) | `multiple` | `parallel` (≥85%) | `parallel_multiple` | `multi_turn_base` | `multi_turn_miss_func` | `multi_turn_miss_param` | 四条门槛通过 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **`Qwen3.5-2B 基座`** | — | 58.00% | 78.75% | 68.67% | 70.50% | 82.50% | 50.50% | 71.00% | 29.50% | 2.50% (5/200) | 15.00% | 2/4 |
| **`旧 E0 (v0 smoke)`** (`ckpt-39`) | 1.00 | *(8类 57.55%)* | 25.00% | 未测 | **92.50%** | **94.50%** | 89.00% | 88.00% | 23.00% | **7.50%** (15/200) | 12.50% | *(未全测)* |
| **`Clean Smoke (v1r1)`** (`ckpt-34`) | 1.00 | 67.22% | 82.08% | 75.00% | 89.50% | 93.00% | 88.50% | 88.00% | 24.50% | 0.00% (0/200) | 12.50% | **4/4** |
| **`v1r1-clean` 正式** (`ckpt-137`) | 0.50 | 62.85% | 69.58% | 64.59% | 90.75% | 91.00% | **89.50%** | **89.00%** | 19.50% | 2.00% (4/200) | 14.50% | 2/4 |
| **`v1r1-clean` 正式** (`ckpt-274`) | 1.00 | 66.52% | 85.00% | 79.30% | 81.50% | 90.00% | 89.00% | 85.00% | 15.00% | 1.00% (2/200) | 10.50% | 3/4 |
| **`v3` `ckpt-34`** | 0.41 | 59.69% | 58.75% | 55.20% | 89.50% | 93.00% | 86.50% | 88.00% | 29.50% | **6.50% (13/200)** | 16.00% | 2/4 |
| **`v3` `ckpt-68`** ★ | 0.81 | 64.72% | **77.92%** | **67.53%** | **89.25%** | 92.50% | **86.50%** | 86.00% | 27.50% | 1.00% (2/200) | 17.50% | **4/4** |
| **`v3` `ckpt-102`** | 1.21 | 68.54% | 88.75% | 81.56% | 80.75% | 91.00% | 85.50% | 84.50% | 26.50% | 0.50% (1/200) | 17.00% | 3/4 |
| **`v3` `ckpt-136`** | 1.62 | **68.94%** | **89.17%** | 85.07% | 76.00% | 88.00% | 81.50% | 84.50% | **30.50%** | 1.00% (2/200) | **18.50%** | 2/4 |
| **`v3` `ckpt-168`** | 2.00 | 68.91% | **89.17%** | **85.41%** | 76.25% | 88.00% | 82.50% | 84.00% | 29.00% | 1.00% (2/200) | 17.00% | 2/4 |

### 2.2 API-Bank Full 对比（1,281 步：753 调用 + 528 回复）

| 权重 / 阶段 | 调用准确率 (753) | L1 (389) | L2 (119) | L3 (245) | 应调轮未调用 (`missing_tool_call`) | 选错工具 (`wrong_tool`) | 回包后正文泄漏 `<tool_call>` (528) | 回复 ROUGE-L F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Qwen3.5-2B 基座`** | 66.53% (501) | 79.18% | 47.90% | 55.51% | 28 | — | 106/528 | 0.3660 |
| **`Clean Smoke (v1r1)`** (`ckpt-34`) | 63.75% (480) | 75.84% | 46.22% | 53.06% | 57 | — | 126/528 | 0.3617 |
| **`v1r1-clean` 正式** (`ckpt-137`) | 66.67% (502) | 77.12% | **51.26%** | 57.55% | 40 | — | 111/528 | 0.3622 |
| **`v1r1-clean` 正式** (`ckpt-274`) | 63.61% (479) | 76.86% | 48.74% | 49.80% | 93 | — | **83/528** | **0.3764** |
| **`v3` `ckpt-34`** | **68.26% (514)** | 79.43% | 49.58% | 59.59% | **16** | 32 | 133/528 | 0.3516 |
| **`v3` `ckpt-68`** ★ | **67.99% (512)** | **80.21%** | 49.58% | 57.55% | 28 | 25 | 128/528 | 0.3581 |
| **`v3` `ckpt-102`** | **68.13% (513)** | 78.92% | 49.58% | **60.00%** | 57 | **20** | 126/528 | 0.3561 |
| **`v3` `ckpt-136`** | **67.86% (511)** | 79.69% | 47.90% | 58.78% | 53 | 21 | 124/528 | 0.3673 |
| **`v3` `ckpt-168`** | 66.80% (503) | 78.15% | 47.90% | 57.96% | 62 | 21 | 124/528 | 0.3664 |

---

## 3. 客观诊断与五大核心发现

### 发现 1：`SFT-v3` 彻底扭转了 `v1r1` 的多轮崩塌与 API-Bank 退化
对比 `v1r1-clean` 正式训练与 `sft-v3`：
- **BFCL 多轮全面回暖并超越基座**：`v1r1` 训练中 `multi_turn_base` 从基座 `29.50%` 一路跌至 `19.50%`（`ckpt-137`）和 `15.00%`（`ckpt-274`），`multi_turn_miss_param` 也跌至 `10.50%`。而在 **`sft-v3`** 全程 5 个 checkpoint 中：
  - `multi_turn_base` 稳定保持在 **`26.50%–30.50%`**（`ckpt-136` 达 `30.50%` 超越基座）；
  - `multi_turn_miss_param` 全程 **`16.00%–18.50%`**，**五个权重全部高于基座（15.00%）和全部 v1r1 实验（10.50%–14.50%）**。
- **API-Bank 全程全面压过基座与所有历史权重**：`sft-v3` 五个 checkpoint 的 API-Bank 调用准确率区间为 **`66.80%–68.26%`**，均高于 `2B 基座`（66.53%）、`Clean Smoke`（63.75%）和 `v1r1`（63.61%–66.67%）。其中 `ckpt-68` L1 达 **`80.21%`**，`ckpt-102` L3 达 **`60.00%`**（基座 L3 为 `55.51%`）。

### 发现 2：`v3 ckpt-68` 是目前综合性能最均衡、唯一通过正式全量 4/4 门槛的候选权重
- **全轮唯一验证集最优 + 四线全过**：`checkpoint-68`（0.81 epoch）既是全轮最低 `eval_loss = 0.34161`，也是除小数据集 Smoke 外**唯一一个正式训练同时满足四条预设行为线**的权重：`irrelevance 77.92%`（≥76%）、`live_irrelevance 67.53%`（≥66%）、`parallel 86.50%`（≥85%）、`simple_python 89.25%`（≥88%）。
- 与 `Clean Smoke`（同样 4/4 通过）相比，`v3 ckpt-68` 在 **`multi_turn_base`（+3.0 pp）、`multi_turn_miss_param`（+5.0 pp）、`multi_turn_miss_func`（+1.0 pp）以及 `API-Bank`（+4.24 pp，67.99% vs 63.75%）** 上全面领先，没有以牺牲多轮和 API-Bank 为代价换取四线通过。

### 发现 3：超过 1 个 Epoch 后出现剧烈的「过保守拒调漂移（Over-Refusal / Zero-Call Drift）」，根因已定位至单工具训练配比偏斜
从 `ckpt-34 -> 68 -> 102 -> 136 -> 168`，出现了与 `v1r1`（`137 -> 274`）完全同构、但幅度更清晰的**「拒调飙升 vs 单轮零调用漏调」跷跷板**：
- `irrelevance` / `live_irrelevance` 分别单调攀升至 **`89.17% / 85.41%`**（所有错误 100% 对应结构化 `tool_calls`：`ckpt-168` 为 26/26 与 129/129）；
- 但 `simple_python` 却从 **`89.50%` / `89.25%`** 断崖跌至 **`80.75%`（`102`）-> `76.00%`（`136`）-> `76.25%`（`168`）**！逐题核验 `model-requests.jsonl` 与错误日志，**几乎全部退化都来自 `simple_function_checker:wrong_count` 且 `finish_reason="stop"`（模型零调用，直接拒答或文本作答）**：

| Checkpoint | `simple_python` 准确率 | `wrong_count` 错误数 | 其中**零调用 (`finish_reason=stop`)** | API-Bank `missing_tool_call` |
| --- | ---: | ---: | ---: | ---: |
| `ckpt-34` | 89.50% | 1 | **1** | 16 |
| `ckpt-68` | 89.25% | 3 | **3** | 28 |
| `ckpt-102` | 80.75% | 48 | **48** | 57 |
| `ckpt-136` | 76.00% | 71 | **70** | 53 |
| `ckpt-168` | 76.25% | 70 | **70** | 62 |

**为什么 `ckpt-102..168` 会把正常的 `simple_python` 题当成拒调/直接答？**
1. 对比 `ckpt-68` 通过而 `ckpt-168` 零调用退化的 **53 道** `simple_python` 题（如 `simple_python_32` 用 `calculate_final_speed` 算自由落体末速度、`simple_python_70` 算车辆碳排放、`simple_python_94` 调用 `update_user_info`、`simple_python_112` 算抽扑克牌概率），可见模型在物理/数学题或存在默认可选参数（如初速度、重力加速度）的单工具请求上选择了「直接自己作答或拒调」。
2. **数据层实证根因**：统计 `data/processed/sft_v3/train.jsonl` 中所有 `len(tools) == 1` 的样本（共 **898** 条，占比 19.6%）发现：
   - **非调用边界样本高达 370 条（占全部 1-tool 样本的 41.2%）**：`toolace_missing_param`（167）+ `toolace_irrelevant_refusal`（108）+ `smoltalk_irrelevant_tools`（95）；
   - **1-tool 单轮单调用正例仅 234 条（26.1%）**：由于 `xlam_single`（700 条）显式要求 `len(tools) >= 2`，导致 `xlam_single` 中 `len(tools) == 1` 的正例为 **0**，仅剩 `toolace_single` 贡献 234 条（另 294 条为 1-tool 多并行调用）。
   - **当训练超过 1 epoch 后，模型反复强化 `1-tool + 数学/科普问法/缺省参数 -> 不调工具` 的统计捷径**，从而在 BFCL `simple_python`（全部为 `len(tools) == 1`）上发生系统性漏调！这也警告我们：**不能仅看「BFCL 九类微平均」选模型**——`ckpt-136`（68.94%）与 `ckpt-168`（68.91%）微平均最高，纯粹是因为 `irrelevance + live_irrelevance` 合计占了 1,124/2,724 题（41.3% 权重），掩盖了 `simple_python` 暴跌 13 个百分点的严重退化！

### 发现 4：`multi_turn_miss_func` 的先升后降证明「缺工具后恢复调用」是 SFT-v3 仍需补齐的环节
- `ckpt-34`（0.41 epoch）在 `multi_turn_miss_func` 拿到了 **`6.50%`（13/200）**，超越基座（2.50%）和 Clean Smoke（0.00%），其 `force_terminated` 也从 `v1r1` 的 117–118 降到 **88**，Turn 1 返空 `empty_turn_model_response` 仅 **2 题**。
- 但从 `ckpt-68` 开始，随着模型学会 Turn 0 拒答缺工具，`miss_func` 的 Turn 1 `empty_turn_model_response` 骤增至 **59 题（`ckpt-68`）→ 72 题（`ckpt-168`）**（所有 `empty_turn_model_response` 合计达 105–130 题），准确率跌回 `0.50%–1.00%`。
- 逐案检查 `multi_turn_miss_func_6` 可确认：Turn 0 缺失目标工具，到 Turn 1 BFCL 环境把缺失工具注入候选列表后，模型本应立即调用新出现的工具，却直接返回空调用列表 `[[]]` 停住。这正印证了审计阶段的预判——**没有显式构造「工具缺失拒答 → 下一轮工具补齐后立刻恢复调用」的多轮对比样本，仅靠单轮拒调与正常多轮，不足以让模型在后期强拒调态下完成工具恢复调用。**

### 发现 5：API-Bank 回复轮 `<tool_call>` 文本泄漏高度集中于 L1/L2，且呈平台期
- `sft-v3` 五个 checkpoint 在 API-Bank 528 个回复样本上的 `<tool_call>` 正文泄漏数为 **`124–133/528`**（结构化 `tool_calls` 均为 `0`），分级别拆解如下：
  - **L3（50 条）**：`ckpt-34` 仅 3 条，`ckpt-68..168` 仅 **0–1 条**（基本彻底收敛，ROUGE-L 与完整轨迹总结均正常）；
  - **L1（375 条）**：稳定在 **73–76 条**（约 19.5%）；
  - **L2（103 条）**：稳定在 **50–54 条**（约 48.5%）。
- **原因**：`api_bank.py` 对 L1/L2 回复测试步构造的是多事件对话中途切片，请求里依然挂载了完整 `tools` 定义并设置 `tool_choice="none"`（vLLM 禁用自动 tool-call parser，但 chat template 仍渲染 `<tools>`）；当模型在对话中间看到刚执行过的 `tool` 回包或前序未完任务时，会以原始 `<tool_call>` XML 文本形式继续输出下一步调用（基座自身也有 `106/528` 泄漏）。`v1r1 ckpt-274` 之所以能降到 `83/528`，很大程度是以牺牲调用积极性（`missing_tool_call=93`、`simple_python=81.5%`）为代价；而 `sft_v3` 并未专门包含「带 `<tools>` 挂载且刚刚收到工具结果、应当立即自然语言收尾」的强约束样本。

---

## 4. 综合思考与下一步行动建议

1. **当前最优可用权重：`sft-v3` `checkpoint-68`（不选 `136/168`）**
   - 切忌被 `136/168` 的 68.9% 九类微平均误导：`136/168` 是靠 1,124 道拒调分拉高的“拒调偏科模型”，在必调单工具任务上已经出现 17.5% 的零调用拒答；
   - **`ckpt-68` 是唯一验证集 loss 最低（`0.34161`）、BFCL 四条线 4/4 全过、`multi_turn_base`（27.5%）与 `miss_param`（17.5%）稳固、且 API-Bank 达到 67.99%（L1 80.21%）的均衡节点**。
   - **立即可做**：用 `ckpt-68` 跑一次 **BFCL `core`（17 类完整集）** 正式评测，重点观察未参加快筛的 8 类（`live_simple`、`live_multiple`、`multi_turn_long_context`、`simple_java`、`simple_javascript` 等）表现，明确区分「快筛内 9 类」与「独立 8 类」泛化结论。

2. **如果继续做下一版 SFT 数据迭代（`sft_v3r1` / `sft_v4`），只需精准补 3 件事**：
   - **平衡 1-tool 分布**：放宽 `xlam_single` 的 `len(tools) >= 2` 限制，补充 300–500 条高质量 `len(tools) == 1` 正例（含带可选默认参数的科学/物理/工程函数调用），把 1-tool 里的边界负例占比从 41.2% 压回 ~20% 以内；
   - **补齐「缺失工具恢复」配对样本**：利用 BFCL 多轮同构构造或可控合成，生成「Turn 0 工具不在场拒答 → Turn 1 注入目标工具后立刻调用」的配对样本（同时对比加/不加工具的正反两面）；
   - **控制训练 epoch 在 `~0.8–1.0 epoch`**：在 4.5k–5.5k 规模下，1 轮左右的 cosine 衰减足以学透调用格式与边界，2 轮重复遍历同一批静态负例收益为负。

3. **更深层的架构判断：为什么这里恰恰应该进入 Stage 2（OPD / GRPO RL）？**
   - 本轮 5 个 checkpoint 的完整轨迹清楚地揭示了 **纯 SFT 的边界**：SFT 只有 token 级交叉熵监督，当同时存在「调用」与「拒调」两类样本时，优化过程实质上是在全局调节“调用 logit 阈值”——调低阈值就过积极（`ckpt-34` 拒调 58%/55%、但 `miss_func` 6.5% 且 `simple_python` 89.5%）；调高阈值就过保守（`ckpt-168` 拒调 89%/85%、但 `simple_python` 掉 13 分、`miss_func` Turn 1 卡死）。
   - 这类**决策边界精确对齐问题**，正是 **GRPO / On-Policy Distillation（OPD）** 的强项：以 **`v3 ckpt-68`** 为 SFT 基座策略，在 RL 阶段同时用「可执行/AST 准确奖励（惩罚漏调与错参）」+「不可用工具拒调奖励（惩罚误调）」，让模型按语义匹配而非按工具数量/表面句式决定调与不调，从而打破当前 SFT 的跷跷板上限。

---

## 5. 复现索引与产物路径

- **远端训练产物**：`/root/ToolCall-SFT/runs/train/sft-v3-2b-lora-20261124T143953Z`（SwanLab [`tgm380ca`](https://swanlab.cn/@sikm/toolcall-sft/runs/tgm380ca)）
- **本地已拉回训练元数据**：[runs/train/sft-v3-2b-lora-20261124T143953Z/](../../runs/train/sft-v3-2b-lora-20261124T143953Z/train_summary.json)
- **本地快评结果与 SwanLab ID**：
  - `ckpt-34`（BFCL [`p9i17trh`](https://swanlab.cn/@sikm/toolcall-sft/runs/p9i17trh) · API-Bank [`kpks5bzj`](https://swanlab.cn/@sikm/toolcall-sft/runs/kpks5bzj)）：[quick-ckpt-34](../../runs/eval/sft-v3-2b-quick-20261124T151222Z/quick-ckpt-34/bfcl/summary.json)
  - `ckpt-68`（BFCL [`4rox8cq0`](https://swanlab.cn/@sikm/toolcall-sft/runs/4rox8cq0) · API-Bank [`y8t0hilw`](https://swanlab.cn/@sikm/toolcall-sft/runs/y8t0hilw)）：[quick-ckpt-68](../../runs/eval/sft-v3-2b-quick-20261124T152018Z/quick-ckpt-68/bfcl/summary.json)
  - `ckpt-102`（BFCL [`mpkor4jq`](https://swanlab.cn/@sikm/toolcall-sft/runs/mpkor4jq) · API-Bank [`3rg3jnq8`](https://swanlab.cn/@sikm/toolcall-sft/runs/3rg3jnq8)）：[quick-ckpt-102](../../runs/eval/sft-v3-2b-quick-20261124T152018Z/quick-ckpt-102/bfcl/summary.json)
  - `ckpt-136`（BFCL [`shihqygd`](https://swanlab.cn/@sikm/toolcall-sft/runs/shihqygd) · API-Bank [`u9eatmls`](https://swanlab.cn/@sikm/toolcall-sft/runs/u9eatmls)）：[quick-ckpt-136](../../runs/eval/sft-v3-2b-quick-20261124T152018Z/quick-ckpt-136/bfcl/summary.json)
  - `ckpt-168`（BFCL [`6rf8hj8x`](https://swanlab.cn/@sikm/toolcall-sft/runs/6rf8hj8x) · API-Bank [`7ooaer57`](https://swanlab.cn/@sikm/toolcall-sft/runs/7ooaer57)）：[quick-ckpt-168](../../runs/eval/sft-v3-2b-quick-20261124T151222Z/quick-ckpt-168/bfcl/summary.json)
