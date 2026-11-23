# ToolCall-SFT

以 `Qwen/Qwen3.5-0.8B` 和 `Qwen/Qwen3.5-2B` 为起点，研究小模型工具调用（Tool Call）与多轮 Agentic 能力的后训练链路：**SFT → On-Policy Distillation（OPD）→ Agentic RL（GRPO）**。评测采用 **BFCL V4**、**API-Bank** 与 **τ²-Bench**，分别覆盖单轮/并行/拒答工具调用、检索与执行、以及真实多轮环境交互与策略遵循。

## 文档索引

- [0. 服务器环境安装](docs/0-服务器环境安装.md)：A100 40GB（Ubuntu 24.04 + 驱动 595.84 / CUDA 13.0）从零装机、国内镜像配置、隔离 venv 与双模型 smoke 验收。
- [1. 评测运行与产物](docs/1-评测运行与产物.md)：评测范围、`runs/` 目录结构、指标口径与 SwanLab。
- [2. 基线评测](docs/2-基线评测.md)：2B / 0.8B 全量评测矩阵、参数审核与性能记录。

## 项目结构

```text
configs/models/             起始模型配置（Qwen3.5-0.8B / 2B）
configs/eval/               三套 benchmark 固定版本与默认参数
src/toolcall/
├── models.py               ModelScope 模型快照下载入口（toolcall-download）
├── evaluation/             三套评测统一入口、评分器、遥测与出图（toolcall-eval）
├── data/                   训练数据清洗、去重与模板对齐（后续阶段）
└── training/               SFT / OPD / Agentic RL 训练入口（后续阶段）
data/                       本地原始与处理后训练数据（Git 忽略）
docs/                       按流程编号的操作文档
runs/                       运行环境、服务日志与实验产物（Git 忽略）
├── envs/                   训练（train）与推理（serve）独立 venv
├── tools/                  三套评测隔离 worker venv（bfcl / api-bank / tau2）
├── eval/                   评测配置、预测、指标、图表与遥测
├── train/                  训练权重、checkpoint 与日志
├── services/               vLLM 服务日志
├── upstream/               评测上游源码、数据夹具与检索模型
└── swanlab/                SwanLab SDK 本地日志
```

## 环境架构原则

服务器自带的 conda base（Python 3.14）与系统 Python 3.12（受 PEP 668 限制）均不直接用于项目依赖。项目统一采用 **`uv` + 国内镜像源 + 托管 Python 3.11** 进行环境隔离：

| 环境 | 职责 | 说明 |
| --- | --- | --- |
| `.venv` | 主 CLI（`toolcall-eval` / `toolcall-download`）、ModelScope、SwanLab、绘图 | 轻量调度环境 |
| `runs/envs/serve` | `vllm` 推理服务 | `cu129` PyTorch + vLLM，隔离推理依赖锁 |
| `runs/envs/train` | SFT / OPD / GRPO 训练 | `cu129` PyTorch + `transformers>=5.0` + `trl>=0.27` + `peft` + `flash-linear-attention` |
| `runs/tools/{bfcl,api-bank,tau2}` | 三套评测 worker | 彼此隔离，避免 `numpy<2` 等依赖冲突 |

## 本地开发与服务器实验

在本地进行架构设计、数据处理与训练/评测代码开发，检查差异后推送到 GitHub；A100 服务器拉取同一分支后运行环境安装、训练、vLLM 推理与评测：

```bash
# 本地提交示例
git status --short
git diff
git add -- path/to/changed-file
git commit -m "Describe the experiment change"
git push -u origin HEAD
```

```bash
# 服务器同步示例（以 main 为例）
git status --short --branch
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

权重、数据集、checkpoint、运行日志与密钥均由 `.gitignore` 忽略，不提交到 Git；关键汇总结果可通过 SwanLab 云端查看，或用 `rsync` / `scp` 取回本地分析。

## 评测与实验记录

首次部署请先完成 [0. 服务器环境安装](docs/0-服务器环境安装.md)。之后激活 `.venv` 并启动本地 vLLM OpenAI 兼容端点（默认 `http://127.0.0.1:18003/v1`）：

```bash
source .venv/bin/activate
toolcall-eval prepare all
toolcall-eval bfcl-v4
toolcall-eval api-bank
toolcall-eval tau2
```

默认参数运行 `smoke` 链路检查。正式全量范围、固定子集及独立 User Simulator 要求见 [1. 评测运行与产物](docs/1-评测运行与产物.md) 与 [2. 基线评测](docs/2-基线评测.md)。

每次评测默认自动上传指标与图表到 SwanLab 云端项目 `toolcall-sft`（按 benchmark 分组），首次运行前需执行：

```bash
swanlab login
```

对比不同训练阶段或变体时可通过 `--variant`（如 `--variant sft-lora`、`--variant opd`、`--variant grpo`）标记。若需离线重绘图表或补传 SwanLab，可执行：

```bash
toolcall-eval report "$run_dir" --swanlab online
```
