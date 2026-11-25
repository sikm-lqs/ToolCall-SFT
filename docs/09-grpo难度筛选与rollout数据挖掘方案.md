# GRPO 难度筛选与双卡 Rollout 数据挖掘方案

- **日期**：2026-11-25
- **基座模型**：`SFT-v3` `Qwen3.5-2B`（`r=64` `checkpoint-68` merge 产物：`merged-checkpoint-68`）
- **前置依据**：[02-grpo-smoke-2b-report.md](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/reports/04-opd-grpo/02-grpo-smoke-2b-report.md)、[08-grpo方案与数据设计.md](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/docs/08-grpo方案与数据设计.md)

---

## 1. 动机：解决 `frac_reward_zero_std = 86.46%` 问题

在 `grpo-smoke-2b`（300 条随机分层采样、`G=4`、`T=0.7`）实测中，训练集 `frac_reward_zero_std` 均值高达 **`86.46%`**（范围 `75%–100%`）：大量样本在 4 次采样中全部拿满分 `+4.0` 或全部同一种错法，导致组内方差为 `0`、策略梯度为 `0`，有效梯度利用率仅约 `13.5%`。

因此，在正式启动全量 `grpo_v1` 训练之前，先利用当前基座模型 `merged-checkpoint-68` 对大规模清洗候选池跑一次 **4-shot Rollout 离线难度摸底与数据挖掘**：
- 剔除已经饱和、无梯度的 **`4/4` 全对题**；
- 以 **命中率 `25%–75%`（即 4 题对 1/2/3 题）** 的高信息熵题为主力（目标占 **~80%**）；
- 保留少量 **`0/4` 全错高难/边界题**（目标占 **~20%**），配合更高训练探索或后续分析；
- 根据实际存活数量与各桶通过率分布，再科学定版 `1,500 – 2,000` 条的最终 `train` 与 `val` 划分。

---

## 2. 候选数据源与容量瓶颈核验（仅限 `xlam-60k + toolace`）

对本地清洗去重后的 `toolace` 与 `xlam-60k` 全池统计（复用 [build_grpo.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/src/toolcall/data/build_grpo.py) 的去重与防污染规则，严格排除 `sft_v3/val`、`sft_v1r1_clean/val`、`BFCL/API-Bank/τ²` 评测问句与 `τ²` 域任务）：

| 能力分桶（`bucket`） | 去重全池可用上限 | 瓶颈分析 | Rollout 候选抽样策略 |
| --- | ---: | --- | --- |
| `multi_turn_continue` | **302**（唯一轨迹切片） | **硬瓶颈** | **全量纳入**（扣除与下面桶共享的轨迹组后全部送测） |
| `multi_turn_post_tool_response` | **784**（去重后剔除已选 `continue` 约剩 **~482** 独立轨迹） | **偏紧** | **全量纳入**剩余独立轨迹切片 |
| `missing_param` | **486** | **硬瓶颈** | **全量纳入**（排除验证集重叠后全部送测） |
| `irrelevance_refusal` | 原生 **370** + 对抗单工具合成 | 原生偏紧 | 原生全部 + 合成 `adv_1tool_irrelevance` 扩容至 **~800**（总计 **~1,170**） |
| `multi_turn_miss_func_recover` | 从 single 合成 | 可按需扩容 | 合成 **~800** 条（优先不与已抽 `single/parallel` 重复 `group_key`） |
| `parallel_call` | **34,533** | 极充足 | 抽样 **2,000** 条 |
| `single_turn_call` | **31,788** | 极充足 | 抽样 **2,500** 条（保持 `len(tools)==1` 占 ~45%） |
| **合计候选池** | — | — | **约 7,700 条唯一轨迹样本**（~`30,800` 次采样） |

> [!NOTE]
> 之所以把候选规模设为「**稀缺桶全量 + 充足桶放大到 2,000–2,500**」（总计 ~7,700 条，而非等比盲抽 5,000 条），是因为按照 Smoke 中 ~13.5% 的中位非零方差率估算，等比盲抽 5,000 条在严格筛掉 `4/4` 后可能仅存活 800–1,000 条，且稀缺桶会被严重筛空；通过稀缺桶全量兜底 + 充足桶放大候选，在双卡下仅多花 3–5 分钟即可一次拿全分布视图。

---

## 3. 双卡最大化压榨架构（DP=2 并行 `vllm serve`）

`Qwen3.5-2B`（BF16 约 `4.2 GB`）在单张 A100-40GB 上属于算力/访存带宽受限而非显存受限；相比引入通信损耗的 `TP=2`，采用 **纯数据并行 `DP=2`** 可 100% 榨干双卡：

1. **双独立推理实例**：
   - **GPU 0**：`vllm serve ... --port 8000 --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.88 --max-model-len 6144 --enable-prefix-caching --language-model-only`
   - **GPU 1**：`vllm serve ... --port 8001 --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.88 --max-model-len 6144 --enable-prefix-caching --language-model-only`
2. **核心加速参数**：
   - 使用预渲染好的 `<tools>` + `<think>\n\n</think>\n\n` prompt 调用 `/v1/completions`；
   - **`n=4` 单请求四采样**：每个 prompt 的 KV Cache 仅 prefill 1 次并由 4 个并行序列共享；
   - **`stop=["<|im_end|>"]`**：显式对齐 `tokenizer_config.json` 中的 chat 终止 token（`248046`，防止回退到 `generation_config` 的 `248044` 导致多吐无效 token）；
   - **采样参数**：`temperature=0.9`、`top_p=0.95`、`top_k=512`、`max_tokens=512`；
   - **高并发调度**：异步调度器向 `:8000` 和 `:8001` 各维持 `concurrency=64`（全局 `128` prompt = `512` 并发生成序列）。
3. **耗时预期**：
   - 参考本机 `core-ckpt-68` 同型模型在单张 A100-40GB 的 `vllm.log` 实测吞吐（prefill `15k–26k tok/s`，decode `1,250–2,348 tok/s`）；
   - 双卡 DP=2 下 `~7,700 × 4 ≈ 30,800` 条完成预期仅需 **`10 – 16 分钟`**（其中模型加载+建图约 1.5 分钟，推理约 8–14 分钟，CPU 奖励打分约数秒）。

---

## 4. 评分与难度分桶口径

每条 prompt 的 4 个回复全部使用与训练完全一致的 [grpo_reward.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/src/toolcall/training/grpo_reward.py) `score_completion()` 计算：
- **`is_pass`（答对判定）**：
  - `task_type == "call"`：`r_format == 1.0` 且 `r_correct >= 2.999`（等价于无格式泄漏且工具名、参数键与 BFCL 宽松参数值完全命中，满分 `r_final = 4.0`）；
  - `task_type == "non_call"`：`r_format == 1.0`、`not has_leak`、`not has_call_intent` 且 `r_correct >= 2.5`（拒答 / 缺参追问 / 工具回包收尾零泄漏达标，详见 `compute_non_call_correct_reward`）。
- **同时记录 `pass_count` 与 `reward_std`**：
  - 难度档位：`0/4`（全错）、`1/4`（25%）、`2/4`（50%）、`3/4`（75%）、`4/4`（全对饱和）；
  - 因部分 `call`（部分参数对）与 `non_call`（如追问语气差 `r_correct=2.0 vs 3.0`）即便 `pass_count=0` 或 `4` 也可能存在组内分差，故同步统计 `reward_std > 0` 样本量，供定版阶段一并参考。

---

## 5. 执行流程与阶段产出

1. **生成候选池**（[build_grpo_rollout_pool.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/src/toolcall/data/build_grpo_rollout_pool.py) + [grpo_rollout_pool.yaml](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/configs/data/grpo_rollout_pool.yaml)）：本地构建 `data/processed/grpo_rollout_pool/candidates.jsonl` 与 SHA-256 清单，并同步至 `rl-server`。
2. **双卡 DP=2 Rollout + 离线打分**（[rollout_mine.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/src/toolcall/training/rollout_mine.py) + [run_rollout_dp2.sh](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/scripts/remote/run_rollout_dp2.sh)）：在远端 `tmux` 快照中执行，完成后自动关闭两路 vLLM 服务。
3. **回拉分布汇总并讨论定版**：拉回 `scored_rollouts.jsonl` 与 `rollout_summary.json` 到本地，输出各桶 `0/4 ~ 4/4` 通过率与 `reward_std` 分布，再与用户共同敲定最终 `1,500 / 2,000` 条 `train` 与 `val` 划分方案。
