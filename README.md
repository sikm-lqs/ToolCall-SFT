# ToolCall-SFT

以 Qwen3.5-0.8B 和 Qwen3.5-2B 为起点，研究小模型通用工具调用的后训练。计划比较 SFT、OPD 和 Agentic RL，使用 BFCL V4 评估通用调用能力，用 τ²-Bench 评估多轮任务。训练框架待定。

## 环境与模型

```bash
conda activate agentic-rl
python -m pip install -e .
toolcall-download configs/models/qwen35_0_8b.yaml configs/models/qwen35_2b.yaml
```

下载入口读取 ModelScope 模型 ID 和 revision，打印本地快照目录。默认共享 ModelScope 缓存，也可通过 `--cache-dir` 指定目录。正式实验记录实际使用的快照、权重哈希、推理配置和评测版本。

```text
configs/models/             两个起始模型的下载配置
configs/eval/               benchmark 版本与 smoke 范围
src/toolcall/data/          数据处理模块
src/toolcall/training/      后训练入口
src/toolcall/evaluation/    评测与日志
data/                       本地数据
runs/                       本地实验产物与评测工具
```

## 评测安装

评测范围、子集选择、报告口径和历史版本入口见 [评测说明](configs/eval/README.md)。

两个评测工具均放在 `runs/tools/`，用 `agentic-rl` 的 Python 创建独立环境。BFCL 固定 NumPy 1.26.4，独立安装可以避免修改训练环境。评测服务通过 OpenAI 兼容接口运行时，可独立选择模型推理后端。

### BFCL V4

```bash
python -m venv --system-site-packages runs/tools/bfcl
runs/tools/bfcl/bin/python -m pip install "bfcl-eval==2026.3.23" "pandas<3" soundfile
runs/tools/bfcl/bin/bfcl generate --help
runs/tools/bfcl/bin/bfcl evaluate --help
```

PyPI 包包含题目、参考答案和评分器，无需再下载 Hugging Face 数据集。按 `configs/eval/bfcl.yaml` 中的 3 个样例验收单轮原生工具调用。正式评测时将本地模型或推理服务注册到官方评测工具，再按 `--run-ids` 选择样例，评估时用 `--partial-eval`。完整运行前，先确认 Web Search 工具和 Memory 子项所需服务可用，并固定 FC/Prompt 模式、thinking、解码设置和并发。

[官方排行榜](https://gorilla.cs.berkeley.edu/leaderboard.html) · [官方评测说明](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md) · [PyPI 版本](https://pypi.org/project/bfcl-eval/2026.3.23/)

### τ²-Bench

```bash
git clone --depth 1 --filter=blob:none --sparse --branch v0.2.0 \
  https://github.com/sierra-research/tau2-bench.git runs/upstream/tau2-bench
git -C runs/upstream/tau2-bench sparse-checkout set src data/tau2/domains data/tau2/user_simulator
python -m venv --system-site-packages runs/tools/tau2
runs/tools/tau2/bin/python -m pip install -e runs/upstream/tau2-bench
runs/tools/tau2/bin/tau2 check-data
```

仓库自带模拟环境、任务与数据库，无需另行下载数据集。固定在原始 τ²-Bench 的 `v0.2.0`，覆盖 Retail、Airline、Telecom。后续 τ³ 和新版多语种发布的口径不同。端到端 smoke 从 Retail 开始，固定受测模型和用户模拟器，用 `--num-tasks 2 --num-trials 1 --max-concurrency 1`。按官方说明配置所用模型服务的凭据，或提供可运行的本地用户模拟器。

[官方仓库与 v0.2.0 运行说明](https://github.com/sierra-research/tau2-bench/blob/v0.2.0/README.md) · [官方论文](https://arxiv.org/abs/2506.07982)

## 运行时间估算

- **BFCL V4**。固定版本加载 22 个计分类别，共约 5,200 条。两款模型在本机分别用 CPU 生成了 3 条单轮结果，用时约 15 秒和 31 秒。完整跑分包含多轮、Memory 和 Web Search。按所有子集混合吞吐 0.2–1 条/秒估算，单模型约 1.5–7.3 小时，实际需用分层 pilot 测并发和吞吐。
- **τ²-Bench**。固定版本有 278 个任务。每条轨迹若耗时 20–90 秒，并发 4，则单模型单轮约 23–104 分钟，4 trials 约 1.5–7 小时。上述是排期估算，实际还受用户模拟器时延和重试影响。

## SwanLab

```bash
run_id=example
toolcall-log-eval "runs/${run_id}/metrics.json" --run-name "$run_id" --config "runs/${run_id}/config.json"
swanlab watch runs/swanlab
```

`metrics.json` 存放数值分项，例如 `{"bfcl/simple_python_accuracy": 0.67}`。默认本地模式。需要云端图表时，登录 SwanLab 并传入 `--mode online`。每次运行建议同时保留配置、原始预测或轨迹、分项分数和用时。权重、数据、日志和本地凭据由 Git 忽略。
