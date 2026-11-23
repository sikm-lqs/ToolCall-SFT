# 工具调用评测

从项目根目录运行 `toolcall-eval`。三个评测共用一个 OpenAI 兼容推理入口，各自使用独立 Python 环境和固定版本的上游数据。默认受测模型为 `Qwen/Qwen3.5-2B`，接口为 `http://127.0.0.1:18003/v1`，Thinking 关闭，温度为 0，单次生成上限为 1,024 tokens。

## 安装与启动

```bash
conda activate agentic-rl
python -m pip install -e .
toolcall-eval prepare all
```

可以把 `all` 换成 `bfcl-v4`、`api-bank` 或 `tau2`。准备命令会创建 `runs/tools/` 下的评测环境，准备固定的上游数据和向量检索模型。首次准备需要 GitHub、PyPI 和 Hugging Face 的网络访问。服务器已有独立环境时，可以用 `--worker-python /path/to/python`。启动受测模型时可用下面的 vLLM 命令，其中 `MODEL_PATH` 指向本机已有的权重快照。

```bash
MODEL_PATH=/absolute/path/to/qwen35-2b-snapshot
vllm serve "$MODEL_PATH" --served-model-name Qwen/Qwen3.5-2B \
  --host 127.0.0.1 --port 18003 --dtype bfloat16 \
  --language-model-only --max-model-len 32768 \
  --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --generation-config vllm --no-enable-prefix-caching
```

本地 τ² Retail smoke 在 10,240 的配置下发生过上下文拒绝，在 32,768 的配置下完成了两段对话。正式评测仍需检查上下文拒绝和截停。换权重时设置 `--model` 为服务上暴露的模型名，用 `--checkpoint-revision` 记录 checkpoint 版本或哈希。受测接口需要身份验证时，通过环境变量 `TOOLCALL_AGENT_API_KEY` 提供密钥。

## 快速验收

```bash
toolcall-eval bfcl-v4
toolcall-eval api-bank
toolcall-eval tau2
```

BFCL smoke 是固定的 3 道 Python 单轮题。API-Bank smoke 从 L1、L2、L3 各抽 20 个调用步骤，另外抽 40 个回复步骤，随机种子为 42。τ² smoke 默认跑 Retail 的前两题，单次试验，每题最多 20 步。默认用受测端点的同一模型充当用户模拟器，便于在一张 GPU 上接通链路。正式比较时用 `--user-model`、`--user-base-url` 固定一个独立的模拟器，必要时通过 `TOOLCALL_USER_API_KEY` 提供凭据。

## 正式评测与子集

| 执行命令 | 覆盖范围 |
|---|---|
| `toolcall-eval bfcl-v4 --profile no-web` | V4 的 Non-live、Live、Multi-turn 和 Memory，共 4,906 道计分题，另含 Memory 初始化任务 |
| `toolcall-eval bfcl-v4 --profile full` | V4 全部计分类别。Web Search 需要 `SERPAPI_API_KEY` |
| `toolcall-eval bfcl-v4 --categories parallel parallel_multiple live_multiple multi_turn_base multi_turn_miss_func` | 五个指定的完整类别 |
| `toolcall-eval api-bank --profile full` | 314 段参考对话里的 753 个调用步骤和 528 个回复步骤 |
| `toolcall-eval tau2 --profile full --trials 1` | Retail、Airline、Telecom 的固定版本默认任务 |
| `toolcall-eval tau2 --profile full --domains retail --task-ids 0 1 --trials 2` | 同一域的固定任务，每题重复两次 |

BFCL 也支持 `--case-ids-file`，格式为类别到指定题目 ID 的 JSON 映射。Memory 固定 ID 会补齐对应历史初始化步骤。按类别或 ID 选题时查看分项和逐题正确率。No-Web 的加权分移除缺测 Web Search 权重后归一化，原始官方 CSV 依然保存。完整 V4、No-Web 和任意子集的分数各自标明口径。 [BFCL 官方入口](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md) · [V4 计分](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)

API-Bank 使用原生 FC，在每一步提供参考历史，并复用上游本地执行检查器。翻译、字典步骤用公布的离线夹具回放。L2 同时记录全量准确率、参考工具可见率和可见条件下的准确率。历史基线有 13 步的参考工具未进入当前适配器提供的 `tools`，其中 7 步此前没有搜索记录，6 步搜索返回了其他工具。文本协议对照为 7/119，其中 1 条来自这 13 步。报告中请保留受测口径及分母。 [API-Bank 官方入口](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank#evaluation)

τ² 按域记录成功率、平均奖励、`pass_hat_k`、终止原因和完整轨迹。比较不同版本时，固定模型调用参数、任务、用户模拟器、`--seed` 和 `--trials`。配置固定在原始 `v0.2.0` 的提交。 [τ² 官方运行说明](https://github.com/sierra-research/tau2-bench/blob/v0.2.0/README.md)

## SwanLab 与结果文件

每次评测生成新的 `runs/eval/<run_id>/`。使用 `--output` 可指定一个新目录。

| 文件 | 内容 |
|---|---|
| `config.json` | 模型、checkpoint 标识（手动提供）、服务版本与上下文（端点提供时）、数据版本、样本和推理参数 |
| `summary.json`、`metrics.json` | 分项分数、覆盖率、耗时、请求错误、截停、吞吐和时延 |
| `scores.png`、`resources.png` | 分项分数和资源曲线。API-Bank 另有 ROUGE-L 和工具覆盖图 |
| `benchmark.log`、`resources.jsonl` | 运行日志、vLLM 排队与 KV cache 采样。推理和评测同机时，传 `--gpu-index 0` 记录整卡显存和 GPU 利用率 |
| 原始预测、官方分数或模拟轨迹 | 按各评测原格式保留。τ² 每个域一份 JSON |
| `swanlab.json` | SwanLab 模式、运行 ID 和日志目录 |

默认自动记录到本地 SwanLab，并按评测名分组。服务器上可以运行 `swanlab watch runs/swanlab`，从本机通过 SSH 转发面板端口。已登录且需要云端同步时设置 `--swanlab online`。只需要本地文件时用 `--swanlab off`。可用 `toolcall-eval report runs/eval/<run_id>` 重绘并重新记录已保存的评测，推理不会重跑。评测进程出错时仍可检查该运行的日志和可用遥测，状态标为失败。 [SwanLab 本地面板](https://docs.swanlab.cn/en/guide_cloud/offline-board.html)

## 复现 BFCL 旧版历史分数

与旧论文对照时，检出对应 tag，按该版本的题集及评分规则生成和评分。

| 历史版 | Git tag | 评测入口 |
|---|---|---|
| BFCL V1 | `v1.0` | `eval_checker/eval_runner.py`，可先选 AST |
| BFCL V2 | `v1.1` | `eval_checker/eval_runner.py`，可先选 `live_simple` |
| BFCL V3 | `v1.3` | 该版本的 `bfcl generate` 和 `bfcl evaluate` |

部分旧版执行题需要历史 REST API 和凭据。 [BFCL 官方发布记录](https://github.com/ShishirPatil/gorilla/releases)
