# 本地与 GPU 服务器（`rl-server`）项目现状盘点

> 盘点时间：2026-11-24 18:10 CST · 目的：为切换「本地 Agent + SSH + tmux + rsync + Git」科研范式建立准确的基线快照与待办清单。

---

## 1. 节点与工具链总览

| 维度 | 本地工作站 | 远端 GPU 服务器（`rl-server`） |
| --- | --- | --- |
| 路径 / 主机 | `/home/sikm/Project/RL-Learn/ToolCall-SFT` | `ubuntu24` (`180.127.11.167:17128`, `/root/ToolCall-SFT`) |
| 硬件资源 | 本地开发机（无训练 GPU） | **NVIDIA A100-SXM4-40GB**（空闲 `40,428 / 40,960 MiB`） |
| 磁盘空间 | — | `/` 共 `196G`（已用 `59G`，可用 `128G`，使用率 `32%`） |
| 范式工具链 | `ssh`、`tmux 3.4`、`rsync 3.2.7`、`git` | `tmux 3.4`、`rsync 3.2.7`、`git`、`python3` |
| Python 运行环境 | 仓库内无 `.venv`；默认 `conda base` 未装 `transformers` | `.venv`（`114M`）、`runs/envs/{serve,train}`、`runs/tools/{bfcl,api-bank,tau2}` |
| 角色定位 | **主控大脑**：算法设计、改代码、写配置、日志分析、报告、Git 存档 | **纯算力执行节点**：单测、训练、vLLM 推理、评测（不装 Agent、不配外网代理、不做 Git 提交） |

### 1.1 新范式端到端实测结果
- **免密 SSH**：`~/.ssh/config` 已配置 `Host rl-server`（`IdentityFile ~/.ssh/id_ed25519`），`ssh -o BatchMode=yes rl-server` 可直接执行远端命令。
- **rsync + tmux 闭环**：已完成一次非破坏性往返实测（本地 `rsync` 推送探针 → `ssh` 在远端新建 `tmux` 分离会话写日志 → `rsync` 拉回本地读取），全流程正常通过。

---

## 2. Git 与文件同步差异（盘点时发现并已治理）

盘点时（基线提交 `f0ff52d`）两侧存在以下差异，已在重构中完成回收与清理：

1. **远端未提交修改（已回收到本地）**
   - `rl-server` 工作区原有未提交的 `reports/baseline-report.md` 与 `reports/baseline-report.html`（`2026-11-24 07:37 UTC` 重新生成，细化了 BFCL `multi_turn:*` 错因分类）。
   - 已通过 `rsync` 拉回本地并归入 [reports/01-baseline/01-baseline-report.md](../../reports/01-baseline/01-baseline-report.md) 与 [01-baseline-report.html](../../reports/01-baseline/01-baseline-report.html)，随后已清理远端脏工作区。
2. **远端遗留后台进程（已清理）**
   - `rl-server` 上残留一条旧式服务器 Agent（`.zcode`）发起的轮询进程（PID `56735`，死循环探测 `http://127.0.0.1:18003/v1/models`），已终止。
   - 远端仍保留 `/root/ToolCall-SFT/.zcode` 与 `/root/.zcode` 历史目录，不再使用（`.gitignore` 已忽略 `.zcode/`）。
3. **本地未跟踪报告文件（已分类归档）**
   - 本地原有两个未跟踪文件 `reports/bfcl-evaluation.html` 与 `reports/bfcl-budget-comparison.png`，已统一归档至 `reports/00-archive/`。

---

## 3. 数据、模型与运行产物分布

| 目录 / 位置 | 本地现状 | `rl-server` 现状 | 说明与同步策略 |
| --- | --- | --- | --- |
| `data/raw/` 等 | `5.0G`（含 6 份原始语料、`manifests/`、`*-inspection/`） | 无（不需冗余存放） | 原始数据与样本审计保留在本地 |
| `data/processed/sft_v1/` | `~85M`（`train.jsonl`、`val.jsonl`、`smoke_v0.jsonl`、模板与 `manifest.json`） | `101M`（含 `sft_v1.tar.gz`） | 双方已有冻结版本；后续新增数据集打包后定向 `rsync` 推送 |
| 基座模型缓存 | 无 | `~/.cache/modelscope/hub/models/Qwen/{Qwen3.5-0.8B,Qwen3.5-2B}` | 仅远端通过 ModelScope 国内源下载缓存 |
| `runs/` 总体积 | `1.5G`（早期 `20261123-103*`、`history-*` 与 `archive/`） | `22G`（含隔离 venv、64K 基线评测、`E0` 训练产物、`swanlab/`、`upstream/`） | **禁止反向覆盖**：只按需拉回轻量指标、摘要与日志，不拉权重文件 |

---

## 4. 实验进度现场（`rl-server`）

### 4.1 已完成：双模型 64K 基线评测（`runs/eval/`）
远端保存有最新的 64K 正式基线运行目录（摘要已汇总至 [01-baseline-report.md](../../reports/01-baseline/01-baseline-report.md)）：
- `Qwen/Qwen3.5-2B`：`bfcl-v4/qwen3.5-2b/core-baseline-20261123T144655.947343Z`（**43.48%**，`2632/4441`）、`api-bank/qwen3.5-2b/full-baseline-20261123T150622.824657Z`（**66.53%**，`501/753`）
- `Qwen/Qwen3.5-0.8B`：`bfcl-v4/qwen3.5-0.8b/core-baseline-20261123T150821.031121Z`（**34.30%**，`2321/4441`）、`api-bank/qwen3.5-0.8b/full-baseline-20261123T152129.962853Z`（**57.24%**，`431/753`）

### 4.2 已完成训练、`vLLM` 待修复：`SFT-v1 E0-smoke-2b`
1. **训练阶段成功**（`runs/train/e0-smoke-2b-20261124T093852Z/train_summary.json`）：
   - `status: "completed"`，1.0 epoch / 39 steps，耗时 `611.84s`，峰值显存 `7.197 GiB`。
   - `flash_linear_attention=true`、`use_kernels=true`、`packing=true`、`padding_free=true` 全部生效。
   - `loss_mask_check: passed`、`forward_check: passed`；`train_loss = 0.3708`、`eval_loss = 0.1930`、`eval_mean_token_accuracy = 0.9470`。
   - `merged-best/` 已导出：`architectures = ["Qwen3_5ForCausalLM"]`、`model_type = "qwen3_5_text"`。
2. **`vllm serve` 启动报错（当前主要阻塞点）**：
   - 日志位置：`rl-server:/root/ToolCall-SFT/runs/services/sft-v1-e0-smoke-2b/20261124T095016.953825030Z/vllm.log`
   - 异常栈指向 `vllm/multimodal/processing/processor.py` → `vllm/model_executor/models/qwen3_5.py:110`：
     `TypeError: Invalid type of HuggingFace config. Expected type: <class 'vllm.transformers_utils.configs.qwen3_5.Qwen3_5Config'>, but found type: <class 'transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig'>`
   - **原因初步定位**：`merge_lora.py` 按纯文本 `Qwen3_5ForCausalLM` 合并后，模型目录的 `config.json` 为 `Qwen3_5TextConfig`；而当前 `runs/envs/serve` 中的 vLLM `Qwen3_5` 渲染/多模态注册路径默认预期多模态顶层 `Qwen3_5Config`。

---

## 5. 后续优先待办（按新范式推进）

1. **同步代码与验证单测**：在本地运行 `bash scripts/remote/sync_code.sh` 同步重构后的仓库到 `rl-server`，并用 `ssh rl-server "cd /root/ToolCall-SFT && runs/envs/train/bin/python -m unittest discover -s tests -v"` 确认 9/9 单测通过。
2. **拉回轻量基线与 `E0` 产物**：运行 `bash scripts/remote/pull_artifacts.sh` 将远端 `runs/eval/`（指标与汇总）、`runs/train/e0-smoke-2b-*/train_summary.json`、`runs/services/sft-v1-e0-smoke-2b/*/vllm.log` 拉回本地分析。
3. **修复 `E0-smoke-2b` merged 模型在 vLLM 的加载问题**：在本地分析 `config.json` / `merge_lora.py` 与 vLLM `qwen3_5` 适配方式，修改后通过 `bash scripts/remote/run_tmux.sh --sync` 在独立代码快照中复测 `vllm serve` 与 8 类 BFCL 快筛，再继续执行 `E1 (2B)` 与 `E2 (0.8B)`。

回到[项目首页](../../README.md) · 查看[科研协作范式](../../AGENTS.md) · 查看[文档索引](../README.md)。
