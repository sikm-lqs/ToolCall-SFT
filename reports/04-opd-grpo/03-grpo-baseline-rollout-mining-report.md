# GRPO Baseline (`merged-checkpoint-68`) 双卡 Rollout 难度摸底报告

- **日期**：2026-11-25 18:39:29 ~ 18:45:10（UTC+8）
- **任务标识**：`grpo-pool-dp2-20261125T103929Z`
- **评测模型**：`SFT-v3` `Qwen3.5-2B`（`r=64` `checkpoint-68` merge 后静态底模 `merged-checkpoint-68`）
- **依据方案**：[09-grpo难度筛选与rollout数据挖掘方案.md](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/docs/09-grpo难度筛选与rollout数据挖掘方案.md)

---

## 1. 候选池构成与防泄漏核验

候选池构建配置文件：[grpo_rollout_pool.yaml](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/configs/data/grpo_rollout_pool.yaml)，构建模块：[build_grpo_rollout_pool.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/src/toolcall/data/build_grpo_rollout_pool.py)。

- **源数据**：仅使用清洗去重后的 `xlam-60k` 与 `toolace`，严格排除 `sft_v3/val`、`sft_v1r1_clean/val`、`BFCL/API-Bank/τ²` 评测集与 `τ²` 域任务。
- **候选总数**：**`7,499` 条唯一轨迹样本**（`unique_group_keys = 7,499`、`excluded_val_overlap = 0`、`token_equivalence_verified = true`）。
- **Prompt 长度**：均值 `612.5` tokens，最大 `3,440` tokens（`max_prompt_tokens = 4096`）。

---

## 2. 双卡 DP=2 性能与算力压榨实测

两张 `NVIDIA A100-SXM4-40GB` 各自拉起独立 `vllm serve`（`:8000` 与 `:8001`、`--tensor-parallel-size 1`、`--enable-prefix-caching`、`--language-model-only`、`--gpu-memory-utilization 0.88`），采用 `n=4` 共享 KV Cache + 每卡 `concurrency=64` 异步抢占式队列调度：

| 指标 | 实测结果 |
| --- | --- |
| **双卡 GPU 利用率** | GPU 0：**`100%`**（`38,919 / 40,960 MiB`）；GPU 1：**`100%`**（`38,333 / 40,960 MiB`） |
| **单卡 vLLM 日志吞吐** | Prefill `~24,800 – 26,240 tok/s`；Decode **`~4,500 – 4,790 tok/s`** |
| **Rollout 总规模** | `7,499` prompts × `4` = **`29,996` completions**（共生成 `2,126,600` tokens） |
| **总耗时（含评分落盘）** | **`226.24 s`（3 分 46 秒）** |
| **端到端聚合速率** | **`33.15 prompts/s`** / **`132.58 completions/s`** / **`9,399.71 gen tokens/s`** |
| **全局格式与泄漏指标** | `format_pass_rate = 0.9987`、`leak_rate = 0.0010`、`mean_completion_tokens = 70.90` |

---

## 3. Baseline 难度与 `reward_std` 分布结果

所有回复均使用训练同源函数 [rollout_mine.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/src/toolcall/training/rollout_mine.py) + `score_completion()`（`n=4, T=0.9, top_p=0.95, top_k=512, max_tokens=512, stop=["<|im_end|>"]`）打分：

### 3.1 七大能力桶全量分布

| 能力分桶（`bucket`） | 候选数 | `0/4`（其中 `std>0`） | `1/4` | `2/4` | `3/4` | **`25%–75%` 合计** | `4/4`（饱和占比） | `std>0` 样本数（比例） | 均分 `mean_reward` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_turn_call` | 2,500 | 315（129） | 90 | 120 | 180 | **390** | 1,795（71.8%） | 519（20.8%） | 3.6195 |
| `parallel_call` | 2,000 | 352（185） | 114 | 111 | 159 | **384** | 1,264（63.2%） | 569（28.5%） | 3.5421 |
| `irrelevance_refusal` | 1,085 | 81（2） | 77 | 62 | 82 | **221** | 783（72.2%） | 226（20.8%） | 2.9450 |
| `missing_param` | 405 | 57（5） | 33 | 27 | 56 | **116** | 232（57.3%） | 121（29.9%） | 2.4716 |
| `multi_turn_continue` | 270 | 80（37） | 25 | 24 | 25 | **74** | 116（43.0%） | 111（41.1%） | 3.1520 |
| `multi_turn_miss_func_recover` | 800 | 89（46） | 42 | 44 | 78 | **164** | 547（68.4%） | 210（26.3%） | 3.5918 |
| `multi_turn_post_tool_response` | 439 | 8（0） | 1 | 5 | 13 | **19** | 412（93.8%） | 19（4.3%） | 3.9300 |
| **总计** | **7,499** | **982（404）** | **382** | **393** | **593** | **1,368** | **5,149（68.7%）** | **1,775（23.7%）** | **3.4377** |

### 3.2 关键细分源对比

| 细分子类 | 样本数 | `0/4` | `1–3/4` | `4/4`（饱和占比） | `std>0` 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `irrelevance: adv_1tool`（对抗单工具干扰拒答） | 785 | 77 | **194** | 514（65.5%） | **25.4%** |
| `irrelevance: toolace_native`（ToolACE 原生拒答） | 300 | 4 | **27** | 269（89.7%） | **9.0%** |
| `single_turn: 1-tool` | 1,125 | 166 | **207** | 752（66.8%） | **24.6%** |
| `single_turn: multi-tool` | 1,375 | 149 | **183** | 1,043（75.9%） | **17.6%** |
| `parallel: 1-tool` | 456 | 62 | **64** | 330（72.4%） | **20.6%** |
| `parallel: multi-tool` | 1,544 | 290 | **320** | 934（60.5%） | **30.8%** |

---

## 4. 核心结论与 `grpo_v2` 构建决策

1. **盲抽为何效率低**：全池随机下 `68.7%`（5,149/7,499）在 `T=0.9` 下已是 `4/4` 满分饱和样本，`zero_std_rate = 76.33%`（与 Smoke 中 `T=0.7` 的 `86.46%` 完全吻合）。剔除 `4/4` 后，单位计算有效梯度可提升 **3–4 倍**。
2. **`0/4` 中隐藏大量半对有效梯度**：982 条 `0/4` 样本中，有 **`404` 条（`41.1%`）存在 `reward_std > 0`**（集中在 `parallel_call` 185 条、`single_turn_call` 129 条、`multi_turn_miss_func_recover` 46 条、`multi_turn_continue` 37 条）。在保留约 20% `0/4` 样本时优先纳入 `std > 0` 样本，既提供即时对比梯度又保留部分 `std == 0` 高难边界探索。
3. **子桶信号强弱差异明显**：
   - `multi_turn_post_tool_response` 已 93.8% 饱和，仅需少量保留维持防回退；`multi_turn`（20%）主力应由 `multi_turn_continue` 与 `multi_turn_miss_func_recover` 承担；
   - `irrelevance_refusal` 的有效挖矿密度几乎全靠对抗单工具拒答（`adv_1tool`）；
   - `single_turn_call` 在当前 2,500 条中产出 `390` 条 `1-3/4`，为满足 Train 40%（600 条）+ Val 的 80/20 配比，先从剩余 `29,288` 条 `single_turn_call` 池中补跑 **`2,500` 条**（双卡 DP=2 预计 ~1 分钟完成）。
4. **Val 集双层结构定版**：第一版 `grpo_v2/val.jsonl` 采用「**~150 条难度筛选主验证**（看 `1-3/4` 与 `0/4` 上的真实提升敏感度）+ **~40 条 `4/4` 稳态锚点**（监控基础能力是否退化）」。

---

## 5. 产物路径

- 汇总报告 JSON：[rollout_summary.json](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/rollout/grpo-pool-dp2-20261125T103929Z/rollout_summary.json)
- 全量评分明细 JSONL：[scored_rollouts.jsonl](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/rollout/grpo-pool-dp2-20261125T103929Z/scored_rollouts.jsonl)
- 双卡 vLLM 日志：[vllm_gpu0.log](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/rollout/grpo-pool-dp2-20261125T103929Z/vllm_gpu0.log)、[vllm_gpu1.log](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/rollout/grpo-pool-dp2-20261125T103929Z/vllm_gpu1.log)
