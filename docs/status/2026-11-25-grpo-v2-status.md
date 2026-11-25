# 阶段进展快照：Stage 1 SFT 收官与 `GRPO-v2` 全量结果（2026-11-25）

> 快照时间：2026-11-25 21:55（Asia/Shanghai）
> 算力节点：`rl-server`（2× A100-40GB PCIe），当前双卡空闲（`14 MiB / 40960 MiB`，无驻留 `tmux` 任务）

---

## 1. 当前总体进展

1. **Stage 1（SFT）已确立稳定基线 `SFT-v3 checkpoint-68`**：
   - `BFCL Core`（17 类 / 4,441 题）：加权分 **`47.02%`**、全量微平均 **`65.71% (2,918/4,441)`**、`Non-Live AST` **`82.25%`**、`Live` **`75.06%`**、`Multi-Turn` **`17.38%`**、`Irrelevance` **`72.63%`**；
   - `API-Bank Full`（1,281 步）：调用准确率 **`67.73% (510/753)`**、ROUGE-L `0.3582`、回包 `<tool_call>` 泄漏 `128/528`；
   - 详报见 [05-sft-v3-2b-ckpt68-core-eval-report.md](../../reports/03-sft/05-sft-v3-2b-ckpt68-core-eval-report.md)。
2. **Stage 2（`GRPO-v1/v2`）基础设施、数据挖掘与全量训练已跑通**：
   - 双卡 DP=2 Rollout 挖掘完成 `9,999` 条去重轨迹评估，构建了高方差难例数据集 `grpo_v2`（`train=1,500`, `val=190`，见 [03-grpo-baseline-rollout-mining-report.md](../../reports/04-opd-grpo/03-grpo-baseline-rollout-mining-report.md)）；
   - 双卡 `colocate`（`G=8, T=0.9`, LoRA r=32）全量跑完 1 epoch（187 步，耗时 `3,223.49 s` / 53.7 分钟，峰值显存 `20.176 GiB`），验证集奖励单调从 `2.6227` 升至 `3.3720`；
   - 新增双卡分片全量评测流程 [run_core_eval_dp2.sh](../../scripts/remote/run_core_eval_dp2.sh)，将单模型全量 `BFCL Core` 生成时间压缩至 **`543.8 s`**（~9 分钟）。
3. **`GRPO-v2` 全量评测完成，但效果未达替换预期（正向分化 + 单轮/并行过度拒调）**：
   - 已完整评测 `checkpoint-47`、`checkpoint-141`、`checkpoint-187` 三个保存节点，全部记录归档于 [05-grpo-v2-full-and-eval-report.md](../../reports/04-opd-grpo/05-grpo-v2-full-and-eval-report.md)：

| 模型 | BFCL Core 加权 | BFCL 微平均 | Non-Live AST | Live | Multi-Turn | Irrelevance | API-Bank Call | ROUGE-L | 回包泄漏 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SFT-v3 ckpt-68` *(基线)* | **47.02%** | **65.71%** | **82.25%** | **75.06%** | 17.38% | 72.63% | **67.73%** | 0.3582 | 128/528 |
| `GRPO-v2 ckpt-47` | 42.00% | 57.44% | 44.46% | 63.95% | 18.12% | **89.22%** | 64.14% | **0.3875** | **77/528** |
| `GRPO-v2 ckpt-141` | 42.65% | 58.03% | 46.40% | 64.47% | 18.62% | 89.14% | 64.01% | 0.3835 | 82/528 |
| `GRPO-v2 ckpt-187` | 43.43% | 58.55% | 47.31% | 65.36% | **19.88%** | 88.33% | 64.14% | 0.3791 | 86/528 |

---

## 2. 结论与下一步待办

- **结论**：`GRPO-v2` 有效提升了无关拒调（`+15~17 pp`）、多轮状态（`multi_turn_base +4.0 pp` / `long_context +6.5 pp`）和回包防泄漏（`-36% ~ -40%`），但因 `non_call` 中 `63.9%` 为单工具（且 `adv_1tool_irrelevance` 206 条叠加 `train` 100% 剔除 `4/4` 正调用锚点 + `-3/+3` 满幅方差），前 `47` 步即触发单工具与并行场景的**过度拒调（输出 `[]`）**，即使早停在 `ckpt-47` 也无法缓解。
- **下一步备选**：
  1. 先对 `ckpt-187` LoRA 做零训练成本 **Alpha 缩放探针**（`α ∈ {0.25, 0.35, 0.5}`）测试快速甜点；
  2. 构建 **`GRPO-v3`**：恢复 `15%~20%` `4/4` 单工具/并行稳态正锚点进训练集、压降 `adv_1tool_irrelevance` 占比并提高 KL `beta`（`0.04~0.05`），将 `simple_python / simple_java / parallel` 空调用率纳入前置门控。
