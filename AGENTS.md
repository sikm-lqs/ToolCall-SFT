# AGENTS.md — 科研协作范式与操作规范

本文件定义了 **ToolCall-SFT** 项目的核心科研协作范式。所有在本地工作站协作的 AI Agent（如 Codex、Antigravity）及开发者均须遵守以下架构分工、操作流程与安全边界。

---

## 1. 核心架构：本地 Agent + 远端纯算力节点

```text
+------------------------------------------+          rsync (代码 / 配置)          +------------------------------------------+
| 本地工作站（控制面 & 知识库）             |  ==================================>  | 远端 GPU 服务器：rl-server (A100 40GB)   |
| - Codex / Antigravity（本地对话与开发）   |                                       | - 仅保持现有国内网络，不配外网代理       |
| - 算法设计、改代码、写 YAML 配置          |          ssh + tmux (执行/巡检)       | - 不安装任何 AI Agent                    |
| - 分析取回的日志、指标、出图与写报告      |  <=================================>  | - ModelScope 下载模型 / 隔离 venv 执行 |
| - Git 唯一提交点：commit / tag / push     |          rsync (日志 / 指标 / 图表)   | - 不在服务器执行 git commit / push       |
+------------------------------------------+  <==================================   +------------------------------------------+
```

| 组件 | 职责定位 | 严格禁忌 |
| --- | --- | --- |
| **本地 Agent（Codex / Antigravity）** | 算法设计、修改 `src/` / `configs/` / `tests/`、发起远端任务、拉回产物分析、生成 `reports/` 与 `docs/` | 禁止直接把本地无 GPU 环境当成训练/CUDA 验证环境 |
| **SSH（`rl-server`）** | 免密连接 `/root/ToolCall-SFT` 执行远端命令：运行单测、查询 `nvidia-smi`、读写 `tmux` 日志、用国内镜像拉模型/依赖 | 禁止在服务器配置 AI Agent、禁止把本地代理/密钥透传到服务器 |
| **tmux（远端）** | 托管所有耗时任务（训练、vLLM 常驻服务、全量评测），确保 SSH 断开或本地休眠/关机后实验不中断 | 禁止直接用阻塞前台命令或裸 `nohup` 跑长实验 |
| **rsync（双向按需）** | 开跑前把本地代码/配置推送到 `rl-server`；跑完后仅拉回轻量日志、指标 JSON、曲线与图表到本地 | 禁止对远端 `runs/`、`data/` 使用 `--delete`；禁止把数 GB 级权重 checkpoint 拉回本地 |
| **Git（仅本地）** | 代码、配置、文档（`docs/`）与报告（`reports/`）的唯一版本存档、记录与回溯源 | 禁止在 `rl-server` 上直接修改并提交代码（避免双端分叉） |

---

## 2. 标准科研实验循环（SOP）

### Step 1：本地设计与修改
- 在本地完成代码、配置、测试编写与静态检查。
- 涉及文档或报告时，按规范目录与两位数序号存放：
  - `docs/`：`00-...` ~ `07-...` 为主线操作文档；`docs/status/` 存放本地与服务器状态快照。
  - `reports/`：按阶段归入 `00-archive/`、`01-baseline/`、`02-data-audit/`、`03-sft/`、`04-opd-grpo/`，文件以两位数字编号开头（如 `01-sft-v1-report.md`）。

### Step 2：推送代码到 `rl-server`
使用封装脚本（白名单同步 `src/`、`configs/`、`tests/`、`scripts/`、`docs/`、`reports/`、`pyproject.toml`、`README.md`、`AGENTS.md`、`.gitignore` 等受控目录，自动保护远端 `runs/`、`data/`、`.venv/`）：

```bash
# 先 dry-run 预览（可选）
bash scripts/remote/sync_code.sh --dry-run

# 同步代码与配置到 rl-server:/root/ToolCall-SFT
bash scripts/remote/sync_code.sh
```

若修改了 CLI 入口或 `pyproject.toml`，同步后在远端更新可编辑安装：
```bash
ssh rl-server "cd /root/ToolCall-SFT && uv pip install --python .venv/bin/python --no-deps -e . && uv pip install --python runs/envs/train/bin/python --no-deps -e ."
```

### Step 3：在远端运行单测与隔离快照 `tmux` 长任务
- **快速单测**（秒级，可直接 `ssh` 同步执行）：
  ```bash
  ssh rl-server "cd /root/ToolCall-SFT && runs/envs/train/bin/python -m unittest discover -s tests -v"
  ```
- **长任务（训练 / vLLM / 评测）**：必须通过 [scripts/remote/run_tmux.sh](scripts/remote/run_tmux.sh) 在远端 `runs/snapshots/<session>-<timestamp>/` **独立代码快照** 中启动 `tmux` 会话（建议格式：`sft-<exp>`、`vllm-<model>`、`eval-<bench>`），保证后续 `sync_code.sh` 不会污染正在运行实验的延迟加载模块或评测子进程：
  ```bash
  # 示例：同步并在独立代码快照中启动 SFT 长跑（自动落盘 runs/logs/tmux/）
  bash scripts/remote/run_tmux.sh --sync sft-e1-2b -- \
    runs/envs/train/bin/python -m toolcall.training.sft --config configs/train/sft_v1_2b_lora.yaml
  ```
- **日常巡检**：
  ```bash
  ssh rl-server "tmux ls; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
  ssh rl-server "tmux capture-pane -pt sft-e1-2b | tail -n 30"
  ```

### Step 4：按需拉回产物到本地分析
实验结束或阶段检查时，使用封装脚本把远端的指标、汇总 JSON 与日志拉回本地（默认跳过 `*.safetensors`、`*.bin`、`*.pt`、模型权重、venv 与大缓存）：

```bash
# 预览将拉回的轻量产物
bash scripts/remote/pull_artifacts.sh --dry-run

# 拉回 runs/{eval,train,services} 的摘要、指标与日志到本地 runs/
bash scripts/remote/pull_artifacts.sh
```

### Step 5：本地分析、出图与 Git 存档
- 在本地用 `toolcall-baseline-report`、自定义分析脚本或可视化脚本生成报告至 `reports/<stage>/`。
- 在本地复核 `git status` 与 `git diff`，确认只包含代码、配置、文档与轻量报告后，执行本地 `git commit` 与 `git push` 完成可追溯存档。

---

## 3. 远端环境与数据约定

| 环境路径（`rl-server`） | 用途 |
| --- | --- |
| `/root/ToolCall-SFT/.venv` | 主调度 CLI（`toolcall-eval`、`toolcall-download`、`toolcall-baseline-report`） |
| `/root/ToolCall-SFT/runs/envs/train` | SFT / OPD / GRPO 训练与 PyTorch/Transformers 单元测试 |
| `/root/ToolCall-SFT/runs/envs/serve` | `vllm serve` 推理服务 |
| `/root/ToolCall-SFT/runs/tools/{bfcl,api-bank,tau2}` | 三套评测隔离 worker 环境 |
| `/root/ToolCall-SFT/runs/snapshots/<session>-<stamp>/` | 长任务隔离代码快照（`run_tmux.sh` 自动创建，`src/configs/scripts` 独立拷贝 + `data/runs/.venv` 软链） |

- **数据推送**：新冻结的训练数据集（如 `data/processed/sft_v1.tar.gz`）不进 Git，打包后用 `rsync -avz --progress` 定向推送到 `rl-server:/root/ToolCall-SFT/data/processed/` 并在远端校验 SHA-256。
- **节点现状基线**：见 [docs/status/2026-11-24-local-and-server-status.md](docs/status/2026-11-24-local-and-server-status.md)。
