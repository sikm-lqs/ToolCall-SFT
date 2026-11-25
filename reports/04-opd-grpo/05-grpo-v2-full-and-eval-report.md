# `GRPO-v2` 全量 `colocate` 训练与双卡并行全量评测分析报告（BFCL Core 4,441 题 + API-Bank Full 1,281 步）

> 日期：2026-11-25（Asia/Shanghai；Run ID 中时间戳为 UTC）
> 初始化基座：`sft-v3-2b-lora` `merged-checkpoint-68`（BFCL Core 加权 `47.02%`，全量微平均 `65.71%`）
> 训练配置：[grpo_v2_2b_colocate.yaml](../../configs/train/grpo_v2_2b_colocate.yaml) · 双卡并发评测脚本：[run_core_eval_dual.sh](../../scripts/remote/run_core_eval_dual.sh)
> 前序报告：[03-grpo-baseline-rollout-mining-report.md](03-grpo-baseline-rollout-mining-report.md) · [04-grpo-v2-smoke-colocate-report.md](04-grpo-v2-smoke-colocate-report.md) · [SFT-v3 ckpt-68 全量报告](../03-sft/05-sft-v3-2b-ckpt68-core-eval-report.md)

---

## 1. 关键问题直答与总体结论

1. **全量 `grpo_v2` 会保存多少个权重？**
   - 配置设定 `max_steps` 对应 1 个 epoch（`1,500` 条训练样本 / 有效每步 `8` 个 prompt = **`187` 步**），`save_steps: 47`、`save_total_limit: 5`。
   - 实际落盘 **5 个 LoRA Adapter 目录**（远端 `/root/ToolCall-SFT/runs/train/grpo-v2-2b-colocate-20261125T115813Z/`）：
     - `checkpoint-47`、`checkpoint-94`、`checkpoint-141`、`checkpoint-187`
     - `final_adapter`（与 `checkpoint-187` 对应的终态导出目录）

2. **这些权重是否都在 `grpo_v2/val.jsonl` 上评测？如何选优？**
   - **是，全部都测**。配置开启 `eval_on_start: true` 与 `eval_steps: 47`，分别在 `step = 0`（起点基线）、`47`、`94`、`141`、`187` 共 **5 个评估点**完整跑完 `grpo_v2/val.jsonl`（`190` 条双层验证集：140 条困难/中档难度样本 + 50 条含 `4/4` 的锚点）。
   - 验证集 `eval_reward` 单调递增，按最高验证奖励选出 **Top 2 权重**：
     - **Top 1**：`checkpoint-187`（`eval_reward = 3.3720`）→ 部署于 **GPU 1**
     - **Top 2**：`checkpoint-141`（`eval_reward = 3.3314`）→ 部署于 **GPU 0**
   - 两张 A100-40G 通过 [run_core_eval_dual.sh](../../scripts/remote/run_core_eval_dual.sh) 合并 LoRA 并独立启动双路 `vllm serve`（`:18003` 与 `:18004`），通过结构化调用与回包收尾探针后并行完成 **BFCL V4 Core（17 类 / 4,441 题）+ API-Bank Full（1,281 步）** 全量评测，总壁钟约 **21 分钟**（UTC `12:54:49Z` ~ `13:15:58Z`）。

3. **总体发现（正负迁移剧烈分化）：**
   - **显著突破**：`grpo_v2` 在 **无关拒调**（`irrelevance` `+15.83 pp` 达 `93.33%`、`live_irrelevance` `+17.19 pp` 达 `84.95%`）、**多轮长历史**（`multi_turn_base` `+4.00 pp` 达 `31.50%`、`multi_turn_long_context` `+6.50 pp` 达 `30.50%`、`multi_turn_miss_func` 翻倍至 `2.00%`）、以及 **API-Bank 回复质量**（ROUGE-L `0.3582 -> 0.3835`，正文 `<tool_call>` 泄漏由 `128/528` 降至 **`82/528`**，下降 **`35.9%`**）上均显著优于 `SFT-v3 ckpt-68`。
   - **严重回退**：在单轮与并行工具调用场景出现严重的**“过度拒调塌方（Over-Refusal Collapse）”**——模型未丢掉语法或参数填充能力，而是把大量本该发起调用的样本输出成了纯文本拒答/解释（BFCL 解析为空列表 `[]`）：`simple_java` 从 `59.00%` 跌至 `7.00%~9.00%`（`83~85/100` 空调）、`parallel` 从 `86.50%` 跌至 `38.00%~38.50%`（`106~108/200` 空调）、`simple_python` 从 `89.00%` 跌至 `59.75%~60.25%`（`132~134/400` 空调）。受此拖累，BFCL Core 加权分从 `47.02%` 回落至 **`43.43%`**（`ckpt-187`）/ **`42.65%`**（`ckpt-141`），未达到上代 `SFT-68` 上线门槛。

---

## 2. 全量 `grpo_v2` `colocate` 训练与内部验证表现

### 2.1 资源、耗时与工程稳定性

- **Run ID / SwanLab**：`grpo-v2-2b-colocate-20261125T115813Z`（[SwanLab `7te8a410`](https://swanlab.cn/@sikm/toolcall-sft/runs/7te8a410)）
- **并行模式**：2× A100-40G DDP + vLLM `colocate`（`gpu_memory_utilization = 0.38`，`max_model_length = 6144`，`enable_sleep_mode = true`），LoRA（r=32, α=64），`G = 8`、`T = 0.9`、`top_p = 0.95`、`top_k = 512`、`lr = 1e-5`、`beta = 0.02`。
- **实际运行统计**（源自 [train_summary.json](../../runs/train/grpo-v2-2b-colocate-20261125T115813Z/train_summary.json) 与 `trainer_state.json`）：
  - **总耗时**：`3,223.49 s`（**53 分 43 秒**），平均单步（含 rollout 与更新）`11.75 s`（区间 `9.52 s ~ 14.99 s`），含 5 次全量 `val` 评估（单次 `179.5 s ~ 192.9 s`）；
  - **单卡峰值显存**：**`20.176 GiB`**（A100 40GB 仅半卡出头，全程零 OOM / 零抢占 / 零 NCCL 异常）；
  - **训练 37 次采样日志均值**（`step 5..185`，每 5 步记 1 次）：
    - `reward = 3.0744`（`2.3219 ~ 3.3916`），`reward_std = 1.4528`（`0.9242 ~ 2.0497`）
    - `frac_reward_zero_std = 0.4169`（`0.1750 ~ 0.6000`），约 `58.3%` 的组持续产生有效对比梯度
    - `reward/format_mean = 0.9975`，`reward/leak_rate = 0.0012`
    - `reward/call_correct_mean = 1.9916`，`reward/non_call_correct_mean = 2.2897`
    - `kl = 0.0048`（最高仅 `0.0113`），`entropy = 0.1784`，`grad_norm = 0.2813`（`0.1527 ~ 0.5792`）
    - `completions/mean_length = 76.55` tokens，`sampling/importance_sampling_ratio/mean = 0.9996`（`0.9782 ~ 1.0249`）

### 2.2 内部 `grpo_v2/val.jsonl`（190 题）5 个评估点全轨迹

| Eval Step | Epoch | `eval_reward` | `eval_reward_std` | `eval_frac_zero_std` | `format_mean` | `call_correct` | `non_call_correct` | `leak_rate` | `kl` | `entropy` | `mean_len` | `eval_runtime` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` *(基线)* | `0.0000` | `2.6227` | `1.3305` | `0.4632` | `0.9961` | `2.1388` | **`+0.2513`** | `0.0039` | `0.000000` | `0.1297` | `69.66` | `179.50 s` |
| `47` | `0.2513` | `3.2474` | `0.8258` | `0.5789` | `1.0000` | `2.1034` | **`+2.5532`** | `0.0000` | `0.005648` | `0.1781` | `70.96` | `180.99 s` |
| `94` | `0.5027` | `3.3138` | `0.7631` | `0.6053` | `0.9987` | **`2.1938`** | `+2.5173` | `0.0013` | `0.007368` | `0.1664` | `73.37` | `192.88 s` |
| **`141` (Top 2)** | `0.7540` | `3.3314` | `0.7570` | `0.6211` | `0.9974` | `2.1909` | `+2.5918` | `0.0000` | `0.007901` | `0.1757` | `72.67` | `189.24 s` |
| **`187` (Top 1)** | `1.0000` | **`3.3720`** | `0.6831` | `0.6632` | `0.9987` | `2.1825` | **`+2.7194`** | `0.0000` | `0.005103` | `0.1697` | `70.33` | `183.23 s` |

### 2.3 训练集典型节点轨迹抽样（`trainer_state.json`）

| Train Step | `loss` | `reward` | `reward_std` | `frac_zero_std` | `call_correct` | `non_call_correct` | `kl` | `entropy` | `grad_norm` | `lr` | `mean_len` | `step_time` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5` | `0.00974` | `2.3219` | `2.0497` | `0.1750` | `1.7856` | `0.6625` | `0.000037` | `0.1126` | `0.2991` | `4.00e-6` | `81.11` | `12.34 s` |
| `45` | `0.01150` | `2.8702` | `1.3969` | `0.3500` | `1.9280` | `1.4531` | `0.003231` | `0.1276` | `0.2126` | `9.12e-6` | `76.72` | `11.61 s` |
| `95` | `-0.01344` | `3.1947` | `1.3029` | `0.5000` | `1.9280` | `2.5125` | `0.008642` | `0.2328` | `0.1677` | `5.40e-6` | `87.52` | `11.43 s` |
| `140` | `-0.01756` | `3.3916` | `0.9242` | `0.5250` | `2.2211` | `2.9219` | `0.003600` | `0.1395` | `0.1615` | `1.71e-6` | `74.70` | `13.44 s` |
| `185` | `-0.00797` | `2.8552` | `1.5361` | `0.4750` | `1.4351` | `2.7969` | `0.004514` | `0.2243` | `0.2737` | `7.09e-9` | `80.19` | `12.41 s` |

> **关键预警信号（事后回看极为清晰）**：在内部 `val.jsonl` 从 `step=0` 涨到 `step=187` 的 `+0.7493` 总奖励提升中，**`call_correct_mean` 仅从 `2.1388` 微移至 `2.1825`（`+0.0437`），而 `non_call_correct_mean` 从 `+0.2513` 狂飙到 `+2.7194`（`+2.4681`）**——也就是说，训练总奖励的增长几乎 100% 由“学会在 `non_call` 上拒调”贡献。

---

## 3. 双卡并行全量正式评测：BFCL Core + API-Bank Full

评测范围、服务端参数（`vllm 0.19.1`、`--max-model-len 65536 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser qwen3_coder`、`enable_thinking=false`、`temperature=0`、`max_tokens=1024`）与历史基线完全一致。

- **`GRPO-v2 checkpoint-141`**（GPU 0）：BFCL Core [`41lnhnpf`](https://swanlab.cn/@sikm/toolcall-sft/runs/41lnhnpf) · API-Bank Full [`xyhxlae4`](https://swanlab.cn/@sikm/toolcall-sft/runs/xyhxlae4)
- **`GRPO-v2 checkpoint-187`**（GPU 1）：BFCL Core [`6pbzlu7r`](https://swanlab.cn/@sikm/toolcall-sft/runs/6pbzlu7r) · API-Bank Full [`nqoamerq`](https://swanlab.cn/@sikm/toolcall-sft/runs/nqoamerq)

### 3.1 五模型全轨迹总表对比（新增 `checkpoint-47` 早期节点验证）

为验证过度拒调是否随训练步数逐步恶化，使用 [run_core_eval_dp2.sh](../../scripts/remote/run_core_eval_dp2.sh)（双独立单卡 `vllm serve` 按类别分片并发，**BFCL Core 生成压缩至 `543.8 s` / ~9 分钟**）对首保存权重 **`checkpoint-47`** 也跑完了全量 BFCL Core + API-Bank Full（BFCL [`zwc4uglo`](https://swanlab.cn/@sikm/toolcall-sft/runs/zwc4uglo) · API-Bank [`2fh802p5`](https://swanlab.cn/@sikm/toolcall-sft/runs/2fh802p5)）：

| 模型 | BFCL Core 加权分 | BFCL Core 全量微平均 (4,441) | Non-Live AST (4组均) | Live Acc (AST微均) | Multi-Turn Acc | Relevance / Irrelevance | API-Bank 调用准确率 (753) | API-Bank ROUGE-L F1 | API-Bank 回包泄漏 (528) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-2B 基座` | 43.48% | 59.27% (2,632) | 60.38% | 70.61% | 18.75% | **87.50%** / 73.71% | 66.53% (501) | 0.3660 | 106 |
| `SFT-v3 ckpt-68` *(RL起点)* | **47.02%** | **65.71% (2,918)** | **82.25%** | **75.06%** | 17.38% | 81.25% / 72.63% | **67.73% (510)** | 0.3582 | 128 |
| `GRPO-v2 ckpt-47` *(GPU0+1分片)* | 42.00% | 57.44% (2,551) | 44.46% | 63.95% | 18.12% | 75.00% / **89.22%** | 64.14% (483) | **0.3875** | **77** |
| `GRPO-v2 ckpt-141` | 42.65% | 58.03% (2,577) | 46.40% | 64.47% | 18.62% | 75.00% / 89.14% | 64.01% (482) | 0.3835 | 82 |
| `GRPO-v2 ckpt-187` | 43.43% | 58.55% (2,600) | 47.31% | 65.36% | **19.88%** | 75.00% / 88.33% | 64.14% (483) | 0.3791 | 86 |

> **关键动力学反转结论（`47 -> 141 -> 187`）**：过度拒调并未因取更早期的 `checkpoint-47` 而缓解，反而 **`ckpt-47` 比 `ckpt-141 / 187` 略微更差**（`Non-Live AST` `44.46% -> 46.40% -> 47.31%`；`simple_python` 空调 `139 -> 134 -> 132`；`parallel` 空调 `116 -> 106 -> 108`；API-Bank `missing_tool_call` `58 -> 56 -> 55`）。结合第 2.2 节轨迹可知：前 `47` 步因 `non_call` 的 `-3 vs +3` 极端方差梯度把拒调阈值一把拉偏（`non_call` 从 `+0.25 -> +2.55`，`call` 从 `2.14 -> 2.10`）；随着 `non_call` 组饱和（`zero_std` 升至 `0.66`），后半程 `47 -> 187` 反而在持续缓慢恢复调用能力并推高多轮表现（`18.12% -> 19.88%`）。因此仅靠“早停”无法解决该问题，必须通过 `GRPO-v3` 数据配比或 LoRA 缩放纠偏。

### 3.2 BFCL Core 17 类逐类成绩与“空列表 `[]`（拒调）次数”透视

下表同时列出 `SFT-68` → `GRPO-141` / `GRPO-187` 的**准确率**以及模型**未输出任何调用（`result == []` 或 `""`）的次数**，这是理解本次变化的核心证据：

| BFCL 类别 | 题数 | `SFT-68` 准确率 | `GRPO-141` 准确率 | `GRPO-187` 准确率 (`Δ vs 68`) | `68` 空输出数 | `141` 空输出数 | `187` 空输出数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `simple_python` | 400 | **89.00%** (356) | 59.75% (239) | 60.25% (241) *(−28.75)* | 3 | 134 | **132** |
| `simple_java` | 100 | **59.00%** (59) | 7.00% (7) | 9.00% (9) *(−50.00)* | 0 | 85 | **83** |
| `simple_javascript` | 50 | **44.00%** (22) | 16.00% (8) | 14.00% (7) *(−30.00)* | 1 | 34 | **32** |
| `multiple` | 200 | **92.50%** (185) | 66.00% (132) | 66.00% (132) *(−26.50)* | 1 | 56 | **57** |
| `parallel` | 200 | **86.50%** (173) | 38.00% (76) | 38.50% (77) *(−48.00)* | 0 | 106 | **108** |
| `parallel_multiple` | 200 | **86.00%** (172) | 54.00% (108) | 57.00% (114) *(−29.00)* | 0 | 56 | **53** |
| **`irrelevance`** *(拒调类)* | 240 | 77.50% (186) | **93.33% (224)** | 92.50% (222) *(**+15.00**)* | 186 | 224 | 222 |
| `live_simple` | 258 | **78.68%** (203) | 54.65% (141) | 55.81% (144) *(−22.87)* | 6 | 54 | **54** |
| `live_multiple` | 1,053 | **74.07%** (780) | 67.33% (709) | 68.19% (718) *(−5.89)* | 8 | 80 | **72** |
| `live_parallel` | 16 | **68.75%** (11) | 62.50% (10) | 62.50% (10) *(−6.25)* | 0 | 2 | 2 |
| `live_parallel_multiple` | 24 | **83.33%** (20) | 45.83% (11) | 45.83% (11) *(−37.50)* | 0 | 4 | 4 |
| **`live_irrelevance`** *(拒调类)* | 884 | 67.76% (599) | **84.95% (751)** | 84.16% (744) *(**+16.40**)* | 599 | 751 | 744 |
| `live_relevance` | 16 | **81.25%** (13) | 75.00% (12) | 75.00% (12) *(−6.25)* | 3 | 4 | 4 |
| **`multi_turn_base`** | 200 | 27.50% (55) | 30.00% (60) | **31.50% (63)** *(**+4.00**)* | 0 | 0 | 0 |
| **`multi_turn_miss_func`** | 200 | 1.00% (2) | 1.00% (2) | **2.00% (4)** *(**+1.00**)* | 0 | 0 | 0 |
| `multi_turn_miss_param` | 200 | **17.00%** (34) | 14.50% (29) | 15.50% (31) *(−1.50)* | 0 | 0 | 0 |
| **`multi_turn_long_context`** | 200 | 24.00% (48) | 29.00% (58) | **30.50% (61)** *(**+6.50**)* | 0 | 0 | 0 |

- **拆分子盘微平均**：
  - `此前参评 9 类（2,724 题）`：`SFT-68 64.68% (1,762)` → `141 59.51% (1,621)` / `187 59.77% (1,628)`
  - `独立未参评 8 类（1,717 题）`：`SFT-68 67.33% (1,156)` → `141 55.68% (956)` / `187 56.61% (972)`
- **一个极具诊断价值的事实（“能力没丢，是门控右移”）**：
  - 在 `simple_python` 上，`GRPO-187` 共有 `159` 题失败，其中 **`132` 题直接输出空调用 `[]`**，真正发出调用的 `268` 题正确 **`241` 题（条件调用准确率 `89.93%`）**，甚至略高于 `SFT-68` 非空时的 `356/397 = 89.67%`；
  - 在 `simple_java` 上，`GRPO-187` 共有 `91` 题失败，其中 **`83` 题输出空调用 `[]`**，真正调用的 `17` 题做对 `9` 题（条件调用准确率 `52.94%`，与 `SFT-68` 的 `59%` 同一量级）；
  - 在 `model-requests.jsonl` 中，失败的 `simple_python_0`（137 output tokens）、`parallel_0`（130 output tokens）、`simple_java_0`（78 output tokens）均表现为 `finish_reason="stop"` 的长段纯文本回复——模型并非截断或语法乱码，而是主动写了长段自然语言拒答/提示词。

### 3.3 API-Bank Full（1,281 步：753 调用 + 528 回复）详表

| 指标 | `SFT-68` | `GRPO-141` | `GRPO-187` | 变化解读 |
| --- | ---: | ---: | ---: | --- |
| **调用总准确率 (`753`)** | **67.73% (510)** | 64.01% (482) | 64.14% (483) | −3.59 ~ −3.72 pp |
| `L1` (`389`) | **79.95% (311)** | 76.35% (297) | 76.61% (298) | `missing_tool_call`: `6 -> 15 / 13` |
| `L2` (`119`) | **49.58% (59)** | 44.54% (53) | 44.54% (53) | `missing_tool_call`: `3 -> 10 / 9` |
| `L3` (`245`) | **57.14% (140)** | 53.88% (132) | 53.88% (132) | `missing_tool_call`: `20 -> 31 / 33`（但 `unexpected_multiple_calls` 由 `10 -> 1`） |
| **应调未调合计 (`missing_tool_call`)** | **29** | **56** | **55** | **几乎翻倍**（直接占去 `L1/L2/L3` 掉分的绝大部分） |
| **回包正文 `<tool_call>` 泄漏 (`528`)** | 128 (24.24%) | **82 (15.53%)** | 86 (16.29%) | **显著改善（−42 ~ −46 次）** |
| **回复 ROUGE-L F1 (`528`)** | 0.3582 | **0.3835** | 0.3791 | **+0.021 ~ +0.025**（L1/L2/L3 全线上升） |

> API-Bank 的失败归因与 BFCL 完全同源：`SFT-68` 总错误中仅 `29` 次 `missing_tool_call`，而 `GRPO-141 / 187` 分别升至 `56 / 55` 次（增加 `+27 / +26` 次），几乎完全等于调用准确率下降的 `28 / 27` 题！与此同时，`GRPO-141/187` 显著压缩了冗余多调用（`L3 unexpected_multiple_calls: 10 -> 1`）与回包标签泄漏（`128 -> 82/86`）。

### 3.4 推理吞吐与延迟对比（BFCL Core 4,441 题）

| 指标 | `SFT-68` | `GRPO-141` | `GRPO-187` |
| --- | ---: | ---: | ---: |
| `generation_seconds` / `wall_seconds` | **`737.4 s` / `743.1 s`** | `995.4 s` / `1000.7 s` | `988.5 s` / `996.4 s` |
| `latency_p95_seconds` | **`16.08 s`** | `25.24 s` | `24.59 s` |
| `cases_per_second` | **6.02** | 4.46 | 4.49 |
| `output_tokens` *(GRPO)* | — | 1,374,935 | 1,369,685 |
| `request_errors` / `context_rejections` / `length_stops` | 5 / 5 / **54** | 6 / 6 / 78 | **3 / 3** / 86 |
| `peak_gpu_used_mib` | 38,387 MiB | 38,349 MiB | 38,349 MiB |

> 因大量本应输出短 JSON 调用的题改写成了较长的自然语言拒答/解释，以及多轮场景的更长响应，`length_stops`（1024 截断）由 `54` 升至 `78 / 86`，生成耗时拉长约 34%。

---

## 4. 根因深度归因：为何 `val` 涨至 `3.37`，实测却出现过度拒调？

结合 `grpo_v2/train.jsonl`、`grpo_v2/val.jsonl` 的真实数据分布与 [grpo_reward.py](../../src/toolcall/training/grpo_reward.py) 的激励结构，可以定位出 **4 条相互放大的结构性根因**：

1. **根因一：`non_call` 与“单工具 (`len(tools)==1`)”之间形成了极强的虚假负相关捷径**
   - 检查 `data/processed/grpo_v2/train.jsonl`（1,500 条）的实际工具数分布：
     - `non_call` 共 `399` 条，其中 **`255` 条（`63.9%`）仅有 1 个候选工具（`len(tools) == 1`）**；
     - 在 `irrelevance_refusal`（225 条）中，高达 **`206` 条（`91.6%`）为 `len(tools) == 1`**（`xlam-60k:adv_1tool_irrelevance` 190 条 + `toolace:adv_1tool_irrelevance` 9 条 + `toolace_irrelevant_refusal` 7 条）；`missing_param`（150 条）中也有 `49` 条为 `len(tools) == 1`。
   - 对照 BFCL 掉分最惨的类别：`simple_java`、`simple_javascript`、`simple_python`、`parallel`、`live_simple` 在输入形态上全都是“极简上下文 + `1` 个工具”。在强化学习探索中，策略迅速捕捉到 **“`len(tools) == 1` 且参数/语境稍不明显 → 拒调最安全”** 的捷径，因此在单工具类别上发生 `33%~83%` 的大面积空调，而多工具类别 `live_multiple` 空调用率仅 `6.8%`（72/1,053）。

2. **根因二：训练集 100% 剔除 `4/4` 简单正例，叠加 `non_call` 与 `call` 之间巨大的非对称方差梯度**
   - 在 `grpo_v2/train.jsonl` 中，`1,101` 条 `call` 样本按 `pass_count ∈ {0,1,2,3}` 筛入（`0/4: 222`、`1/4: 235`、`2/4: 246`、`3/4: 398`），**`4/4` 满通过的日常基础正样本为 `0` 条**。
   - 与此同时，观察 `step = 0`（初始基线 `SFT-68`）的评估奖励裂口：
     - `call_correct_mean = +2.1388`（基线本来就较高且错多集中在细微参数/多值对齐上，单组内方差相对温和）；
     - `non_call_correct_mean = +0.2513`（因为 `ckpt-68` 在难拒调样本上大概一半尝试调用得 `-3.0`、一半拒调得 `+3.0`，组内出现 **`-3.0` vs `+3.0` 满幅双峰分化**）。
   - GRPO 的组内归一化优势（Advantage）使 `non_call` 的拒调行为每步都获得压倒性的正向策略梯度；而另一边没有任何 `4/4` 基础 `call` 正例拉住“常规单轮/并行必须调用”的先验，导致**调用门控（Call Threshold）整体大幅右移**。

3. **根因三：`missing_param` + `irrelevance` 奖励过于宽松，诱导“疑罪从拒”**
   - 在 [grpo_reward.py](../../src/toolcall/training/grpo_reward.py#L395-L430) 中：
     - `irrelevance_refusal` 只要无调用意图标且纯文本 `>= 12` 字符即直接拿 **`+3.0`**（含格式分共 **`+4.0`** 满分）；
     - `missing_param` 只要文本提到任一缺参或 `provide / specify / missing / need / required / parameter` 即可稳定拿 **`+2.0 ~ +3.0`**；
     - 相比之下，`call` 任务对名称、参数键、参数值与数据类型都有连续惩罚。遇到 BFCL 中常见的隐式默认值、Java/JS 类型修饰或并行拆分歧义，模型一旦探索出“礼貌询问/说明缺失条件”，在 RL 样本池中往往尝到高回报甜头，最终外溢到可调样本上。

4. **根因四：内部 `val.jsonl` 缺少正常边界调用锚点，单凭标量 `eval_reward` 掩盖了门控漂移**
   - 如第 2.2 节表所示，`step=0 -> 187` 间 `val` 总奖励涨 `0.75`，完全掩盖了 `call_correct` 原地踏步（`2.14 -> 2.18`）与 `non_call` 单边拉升的失衡。而 `grpo_v2/val.jsonl` 同样来自同分布难例池（且缺乏 Java/JS/隐式默认单轮正例），因此“内部验证单调最优 ≠ 外部通用调用安全”。

---

## 5. 下一步改进与 `GRPO-v3` 纠偏路线

本次全量实验已经彻底跑通双卡 `colocate` 训练（53 分钟 / 20.2 GiB VRAM / zero OOM）与双卡并行全量评测（21 分钟同时跑完两个全量 Core+API-Bank），且**验证了 GRPO 对多轮状态追踪、`irrelevance` 拒调与回包泄漏消除的巨大潜力**。下一步无需大改工程底座，只需聚焦以下 3 条纠偏路线：

1. **先做零训练成本排查：权重线性插值探测（LoRA Alpha Merge Sweep）**
   - 既然 `GRPO-187` 的调用生成精度还在（`simple_python` 非空时 `89.93%`）且多轮/拒调/防泄漏强，可在双卡上直接把 `GRPO-187` LoRA 以较小缩放系数（如 `α ∈ {0.25, 0.35, 0.5}`）叠加回 `SFT-68` 做 5 分钟 Quick 探针，立即验证是否存在能同时守住 `Non-Live AST > 80%` 与拉高 `Irrelevance / Multi-Turn` 的现成甜点权重。
2. **构建 `GRPO-v3` 训练配比（解除 `tools==1 -> 拒调` 绑定并恢复基础正样本锚点）**
   - **恢复 15%~20% 的 `4/4`（或 `3/4` 高置信）稳态 `call` 锚点进训练集**：尤其覆盖 `len(tools)==1` 单轮调用、`parallel_call` 及多语言风格函数签名；
   - **砍重配平 `adv_1tool_irrelevance`**：将当前占比过重的 `adv_1tool_irrelevance`（`206/225` 单工具）压降至 `<= 50~60` 条，并强制掺入多工具 `irrelevance` 样本；将 `irrelevance + missing_param` 合并占比压缩到总训练集 **`15%~18%`**（本次为 `25%`，且有效方差梯度更高）；
   - **审计 `missing_param` 样本**：剔除仅缺少可选参数（optional）或可通过上下文合理推断却被标为拒答的模糊样本。
3. **调优 RL 正则与门控指标**：
   - 将 KL 惩罚 `beta` 从 `0.02` 提高至 **`0.04 ~ 0.05`**，并拆分监控 `val/non_call` 与 `val/call_single_1tool`，避免单一优势项拖偏基础策略；
   - 将 **BFCL 轻量门控快筛（含 `simple_python / simple_java / parallel / irrelevance` 抽样空输出率监测：`simple_python <= 3%`、`parallel <= 2%`）** 作为 `GRPO-v3` 推全量 Core 前的必经前置 Gate。

---

## 6. 复现索引与产物路径

- **训练配置**：[configs/train/grpo_v2_2b_colocate.yaml](../../configs/train/grpo_v2_2b_colocate.yaml)
- **奖励源码**：[src/toolcall/training/grpo_reward.py](../../src/toolcall/training/grpo_reward.py)
- **训练集 / 验证集**：`data/processed/grpo_v2/{train,val}.jsonl`
- **训练轻量产物（本地）**：[runs/train/grpo-v2-2b-colocate-20261125T115813Z/train_summary.json](../../runs/train/grpo-v2-2b-colocate-20261125T115813Z/train_summary.json) · [checkpoint-187/trainer_state.json](../../runs/train/grpo-v2-2b-colocate-20261125T115813Z/checkpoint-187/trainer_state.json)
- **远端权重（`rl-server`）**：`/root/ToolCall-SFT/runs/train/grpo-v2-2b-colocate-20261125T115813Z/{checkpoint-47,checkpoint-94,checkpoint-141,checkpoint-187,final_adapter}`
- **双卡评测产物（本地）**：
  - `checkpoint-47`：[BFCL summary.json](../../runs/eval/grpo-v2-checkpoint-47-sharded2-20261125T133712Z/bfcl/summary.json) · [API-Bank summary.json](../../runs/eval/grpo-v2-checkpoint-47-sharded2-20261125T133712Z/api-bank/summary.json)
  - `checkpoint-141`：[BFCL summary.json](../../runs/eval/grpo-v2-core-dual-20261125T125449Z/core-checkpoint-141/bfcl/summary.json) · [API-Bank summary.json](../../runs/eval/grpo-v2-core-dual-20261125T125449Z/core-checkpoint-141/api-bank/summary.json)
  - `checkpoint-187`：[BFCL summary.json](../../runs/eval/grpo-v2-core-dual-20261125T125449Z/core-checkpoint-187/bfcl/summary.json) · [API-Bank summary.json](../../runs/eval/grpo-v2-core-dual-20261125T125449Z/core-checkpoint-187/api-bank/summary.json)
- **SwanLab 运行**：训练 [`7te8a410`](https://swanlab.cn/@sikm/toolcall-sft/runs/7te8a410) · `47` BFCL [`zwc4uglo`](https://swanlab.cn/@sikm/toolcall-sft/runs/zwc4uglo) · `47` API-Bank [`2fh802p5`](https://swanlab.cn/@sikm/toolcall-sft/runs/2fh802p5) · `141` BFCL [`41lnhnpf`](https://swanlab.cn/@sikm/toolcall-sft/runs/41lnhnpf) · `141` API-Bank [`xyhxlae4`](https://swanlab.cn/@sikm/toolcall-sft/runs/xyhxlae4) · `187` BFCL [`6pbzlu7r`](https://swanlab.cn/@sikm/toolcall-sft/runs/6pbzlu7r) · `187` API-Bank [`nqoamerq`](https://swanlab.cn/@sikm/toolcall-sft/runs/nqoamerq)
