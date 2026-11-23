# ToolCall-SFT

以 Qwen3.5-0.8B 和 Qwen3.5-2B 为起点，研究小模型通用工具调用的后训练。第一阶段选用 TRL 的 `SFTTrainer`，后续研究 OPD 和 Agentic RL。BFCL V4 和 API-Bank 用于评估工具调用，τ²-Bench 用于评估多轮任务。

## 文档

- [0. 服务器环境安装](docs/0-服务器环境安装.md)：A100 40GB、vLLM 与双模型 smoke。
- [1. 评测运行与产物](docs/1-评测运行与产物.md)：数据集、运行目录与指标文件。
- [2. 基线评测](docs/2-基线评测.md)：双模型全量矩阵、参数审核和性能记录。

## 项目结构

```text
configs/models/             两个起始模型的下载配置
configs/eval/               benchmark 版本与 smoke 范围
src/toolcall/data/          数据处理目录，目前占位
src/toolcall/training/      后训练目录，目前占位
src/toolcall/evaluation/    三套评测入口、指标与图表
data/                       本地数据
docs/                       按流程编号的操作文档
runs/eval/<评测集>/<模型>/   每次运行的配置、预测、指标和遥测
runs/swanlab/               SwanLab SDK 运行日志
```

## 本地开发与服务器实验

在本地研究算法、设计数据和实验配置、开发代码，由 Codex、Antigravity 等本地 Agent 协助。检查差异后把代码、轻量配置和文档提交到 GitHub。服务器从同一个 GitHub 仓库更新代码，运行 GPU 训练、推理和评测。

本地在项目根目录提交并推送当前分支。把下面的文件名替换成这次要提交的文件。

```bash
git status --short
git diff
git add -- path/to/changed-file
git diff --cached
git commit -m "Describe the experiment change"
git push -u origin HEAD
```

服务器首次克隆和安装见[服务器环境安装](docs/0-服务器环境安装.md)。后续运行前，在服务器的项目根目录检查当前分支并更新。下面以 `main` 为例。专题分支可将两处 `main` 替换成已推送的分支名，首次切换前先运行 `git fetch origin`。

```bash
git status --short --branch
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

服务器侧将试验性代码修改带回本地提交。已经启动的实验保留当时的代码，等结束后再更新对应工作目录。每次实验记录 Git commit、数据与模型版本、配置、随机种子和环境版本。权重、数据、checkpoint、日志和凭据保存在服务器或本地的忽略目录，结果可用 SCP 或 rsync 取回。

## 评测

激活项目的 `.venv` 后，用同一入口准备和运行三个评测。跑评测前启动 OpenAI 兼容模型服务，默认地址为 `http://127.0.0.1:18003/v1`。

```bash
toolcall-eval prepare all
toolcall-eval bfcl-v4
toolcall-eval api-bank
toolcall-eval tau2
```

默认是 smoke。完整评测、固定子集和用户模拟器见[评测运行与产物](docs/1-评测运行与产物.md)。正式基线先走[人工审核](docs/2-基线评测.md)。

## SwanLab 云端实验

```bash
swanlab login
```

统一入口将每次评测的配置、分项指标和图表上传到 SwanLab 云端的 `toolcall-sft` 项目，按评测集分组。配置、指标、原始预测或轨迹和图表也保存在 `runs/eval/`。本地和服务器使用同一 SwanLab 工作空间。团队空间可通过 `SWANLAB_WORKSPACE` 指定。 [SwanLab 环境变量](https://docs.swanlab.cn/en/api/environment-variable.html)

评测自动命名为 `模型[-变体]-评测集-范围-UTC时间`，例如 `qwen3.5-0.8b-bfcl-v4-smoke-20261123-031500Z`。对比训练方法或 checkpoint 时用 `--variant sft-lora` 等短标签。每次手动记录或今后新增训练实验也使用描述性的名称，例如 `qwen3.5-0.8b-sft-lora-20261123-031500Z`。

用 `toolcall-eval report "$run_dir"` 重绘已保存的运行，其中 `run_dir` 指向 CLI 打印的运行目录。权重、数据、日志和本地凭据由 Git 忽略。
