# ToolCall-SFT

以 `Qwen/Qwen3.5-0.8B` 和 `Qwen/Qwen3.5-2B` 为起点，研究小模型工具调用（Tool Call）与多轮 Agentic 能力的后训练链路：**SFT → On-Policy Distillation（OPD）→ Agentic RL（GRPO）**。评测采用 **BFCL V4**、**API-Bank** 与 **τ²-Bench**，分别覆盖单轮/并行/拒答工具调用、检索与执行、以及真实多轮环境交互与策略遵循。

> **科研协作范式**：本项目采用**「本地 Agent（Codex / Antigravity）+ SSH + tmux + rsync + Git」**分离架构——代码修改、数据分析、报告撰写与 Git 版本存档全部在本地完成；GPU 服务器（`rl-server`）保持国内网络作为纯算力执行机，无需安装 AI Agent 或配置代理。详见 **[AGENTS.md](AGENTS.md)** 与 **[本地及服务器现状盘点](docs/status/2026-11-24-local-and-server-status.md)**。

## 文档与报告索引

- **协作规范与现场盘点**：
  - [AGENTS.md](AGENTS.md)：本地 Agent + `ssh` + `tmux` + `rsync` + `git` 研究范式与安全边界。
  - [本地与服务器现状盘点（2026-11-24）](docs/status/2026-11-24-local-and-server-status.md)：本地与 `rl-server` 环境/数据/基线/SFT 进度及当前阻塞诊断。
  - [实验与分析报告索引](reports/README.md)：按阶段分类编号的基线报告、数据图鉴与实验汇总。
- **主线操作文档**（完整索引见 [docs/README.md](docs/README.md)）：
  - [00. 服务器环境安装](docs/00-服务器环境安装.md)：A100 40GB（Ubuntu 24.04 + 驱动 595.84 / CUDA 13.0）从零装机、国内镜像配置、隔离 venv 与双模型 smoke 验收。
  - [01. 评测运行与产物](docs/01-评测运行与产物.md)：评测范围、`runs/` 目录结构、指标口径与 SwanLab。
  - [02. 基线评测](docs/02-基线评测.md)：2B / 0.8B 全量评测矩阵、参数审核与性能记录。
  - [03. SFT 数据集调研](docs/03-sft数据集调研.md)：训练源候选和首轮配比。
  - [04. 数据归档与样本图鉴](docs/04-数据归档与样本图鉴.md)：六个训练源、评测案例与 MD / HTML 报告。
  - [05. SFT 训练管线](docs/05-sft训练管线.md)：异构数据标准化、Qwen3.5 模板与 Loss Mask 校验、配比与训练超参。
  - [06. SFT-v1 实验方案](docs/06-sft-v1实验方案.md)：`SFT-v1`（`xlam-60k + toolace`）本地构建实录、模板/打包深坑排查与实验协议。
  - [07. SFT-v1 服务器执行手册](docs/07-sft-v1服务器执行手册.md)：A100 执行步骤、Pre-flight、E0/E1/E2 训练与验收清单。

## 项目结构

```text
AGENTS.md                   本地 Agent + SSH/tmux/rsync/Git 科研协作规范
configs/                    数据、模型、评测与训练 YAML 配置
src/toolcall/
├── models.py               ModelScope 模型快照下载入口（toolcall-download）
├── evaluation/             三套评测统一入口、评分器、遥测与报告（toolcall-eval / toolcall-baseline-report）
├── data/                   SFT 数据标准化、模板构建与校验（toolcall-build-data）
└── training/               SFT 训练、Pre-flight 与 LoRA 合并（toolcall-sft / toolcall-merge-lora）
scripts/
├── data-audit/             样本审计与数据图鉴生成脚本
└── remote/                 本地 <-> rl-server 安全同步与快照启动脚本（sync_code.sh / pull_artifacts.sh / run_tmux.sh）
tests/                      Schema 归一化与 Loss Mask 单元测试
docs/                       两位数编号主线文档与 status/ 节点现状盘点
reports/                    按阶段分类编号的实验/数据报告（00-archive ~ 04-opd-grpo）
data/                       原始与处理后训练数据（Git 忽略）
runs/                       远端环境、服务日志与实验产物（Git 忽略）
├── envs/                   训练（train）与推理（serve）独立 venv
├── tools/                  三套评测隔离 worker venv（bfcl / api-bank / tau2）
├── eval/                   评测配置、预测、指标、图表与遥测
├── train/                  训练权重、checkpoint 与日志
├── services/               vLLM 服务日志
├── upstream/               评测上游源码、数据夹具与检索模型
└── swanlab/                SwanLab SDK 本地日志
```

## 环境架构原则

服务器自带的 conda base（Python 3.14）与系统 Python 3.12（受 PEP 668 限制）均不直接用于项目依赖。项目在 `rl-server` 上统一采用 **`uv` + 国内镜像源 + 托管 Python 3.11** 进行环境隔离：

| 环境 | 职责 | 说明 |
| --- | --- | --- |
| `.venv` | 主 CLI（`toolcall-eval` / `toolcall-download`）、ModelScope、SwanLab、绘图 | 轻量调度环境 |
| `runs/envs/serve` | `vllm` 推理服务 | `cu129` PyTorch + vLLM，隔离推理依赖锁 |
| `runs/envs/train` | SFT / OPD / GRPO 训练 | `cu129` PyTorch + `transformers>=5.0` + `trl>=0.27` + `peft` + `flash-linear-attention` |
| `runs/tools/{bfcl,api-bank,tau2}` | 三套评测 worker | 彼此隔离，避免 `numpy<2` 等依赖冲突 |

## 本地开发与服务器实验（新科研范式）

所有代码修改、配置调整、日志分析与文档撰写均在**本地**完成，并由 Git 在本地负责存档与回溯；`rl-server` 仅通过 `ssh + tmux + rsync` 执行计算与回传轻量产物（详细规范见 [AGENTS.md](AGENTS.md)）：

```bash
# 1. 本地改完代码后，推送受控代码与配置到 rl-server（不影响远端 data/ 与 runs/）
bash scripts/remote/sync_code.sh

# 2. 远端运行单元测试，或通过 run_tmux.sh 在独立代码快照中启动长任务
ssh rl-server "cd /root/ToolCall-SFT && runs/envs/train/bin/python -m unittest discover -s tests -v"
bash scripts/remote/run_tmux.sh sft-run -- runs/envs/train/bin/python -m toolcall.training.sft --config configs/train/e0_smoke_2b.yaml

# 3. 实验结束或巡检时，按需拉回轻量指标与日志到本地分析（不拉权重）
bash scripts/remote/pull_artifacts.sh

# 4. 本地整理报告后，使用 Git 存档与回溯
git status --short
git add -- AGENTS.md README.md docs/ reports/ src/ configs/ scripts/ tests/
git commit -m "Describe the experiment change"
git push -u origin HEAD
```

权重、数据集、checkpoint、运行日志与密钥均由 `.gitignore` 忽略，不提交到 Git；关键汇总结果可通过 SwanLab 云端查看，或用 `scripts/remote/pull_artifacts.sh` 取回本地生成报告。

## 评测与实验记录

首次部署请先完成 [00. 服务器环境安装](docs/00-服务器环境安装.md)。之后在 `rl-server` 激活 `.venv` 并启动本地 vLLM OpenAI 兼容端点（默认 `http://127.0.0.1:18003/v1`）：

```bash
source .venv/bin/activate
toolcall-eval prepare all
toolcall-eval bfcl-v4
toolcall-eval api-bank
toolcall-eval tau2
```

默认参数运行 `smoke` 链路检查。正式全量范围、固定子集及独立 User Simulator 要求见 [01. 评测运行与产物](docs/01-评测运行与产物.md) 与 [02. 基线评测](docs/02-基线评测.md)。

每次评测默认自动上传指标与图表到 SwanLab 云端项目 `toolcall-sft`（按 benchmark 分组），首次运行前需执行：

```bash
swanlab login
```

对比不同训练阶段或变体时可通过 `--variant`（如 `--variant sft-lora`、`--variant opd`、`--variant grpo`）标记。若需离线重绘图表或补传 SwanLab，可执行：

```bash
toolcall-eval report "$run_dir" --swanlab online
```
