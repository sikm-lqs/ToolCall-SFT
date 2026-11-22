# 工具调用评测

默认使用 **BFCL V4** 衡量通用调用能力、**τ²-Bench** 衡量多轮任务完成能力。两者都可以先跑固定子集，再按实验需要扩大范围。安装方式见[项目 README](../../README.md#评测安装)。

## BFCL V4

使用 [`bfcl.yaml`](bfcl.yaml) 中固定的 `bfcl-eval==2026.3.23`。把本地 Qwen 服务注册到评测器后，用 `bfcl generate` 保存预测，再用 `bfcl evaluate` 评分。两步使用同一模型标识和类别。使用 `--test-category` 选类别，或用 `--run-ids` 生成固定题目并在评分时加 `--partial-eval`。

| 范围 | 用途 |
|---|---|
| `simple_python` 的 3 条固定题 | Smoke，检查模型输出、工具解析与评分链路 |
| `non_live`、`live` | 基础单轮调用、真实场景单轮调用与误调用 |
| `multi_turn` | 多轮行为，关注缺参追问、工具缺失及长上下文 |
| `all_scoring` | 完整 V4 计分范围，包含 Memory 和 Web Search |

子集结果标注 **BFCL V4 + 类别名 + 样本量**。跑齐 V4 计分类别后再报告 V4 总分。比较原模型与微调模型时，固定 FC/Prompt 模式、聊天模板、工具解析器、Thinking 和生成设置。 [官方评测入口](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md) · [V4 计分](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)

### 复现 V1–V3 历史分数

与旧论文对照时，检出对应 tag，分别安装评测环境，并用同版本的题集生成、评分。各版本可指向同一个 Qwen 服务，适配器需兼容对应评测器。V4 的 `non_live`、`live` 和 `multi_turn` 分数按 V4 子集口径报告。 [官方发布记录](https://github.com/ShishirPatil/gorilla/releases)

| 历史版本 | 官方 Git tag | 评分入口 |
|---|---|---|
| BFCL V1 | `v1.0` | 在 `eval_checker/` 运行 `python eval_runner.py`，可先选 `ast`。 [V1 说明](https://github.com/ShishirPatil/gorilla/blob/v1.0/berkeley-function-call-leaderboard/README.md#evaluating-the-llm-generations) |
| BFCL V2 | `v1.1` | 同样运行 `python eval_runner.py`，可先选 `live_simple`。 [V2 说明](https://github.com/ShishirPatil/gorilla/blob/v1.1/berkeley-function-call-leaderboard/README.md#evaluating-the-llm-generations) |
| BFCL V3 | `v1.3` | 用该版本的 `bfcl generate` 和 `bfcl evaluate`。 [V3 说明](https://github.com/ShishirPatil/gorilla/blob/v1.3/berkeley-function-call-leaderboard/README.md#evaluating-generated-responses) |

V1/V2 的部分执行类题还需要历史 REST API 和凭据。只跑 AST 或 Live 子集时，报告对应子集分数。

## τ²-Bench

使用 [`tau2.yaml`](tau2.yaml) 中固定的 Sierra `v0.2.0`。用 `tau2 run` 运行 Airline、Retail、Telecom。端到端 smoke 可先跑 Retail 2 题、每题 1 次。持续对比时，用 `--task-ids` 固定题目，并固定用户模拟器模型和参数、每题运行次数，以及评测版本。报告时写明域、样本量和每题运行次数，单次运行报任务成功率。 [官方运行说明](https://github.com/sierra-research/tau2-bench/blob/v0.2.0/README.md)

本地状态回放可验证工具环境和状态评分。测模型任务成功率时，运行带用户模拟器的完整对话。

每次评测在 `runs/<run_id>/` 保留配置、原始预测或轨迹、分项得分和耗时，并将指标写入 [SwanLab](../../README.md#swanlab)。
