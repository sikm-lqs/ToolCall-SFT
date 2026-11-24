# 07. `SFT-v1` 服务器端执行与验收手册

本文档记录在远程 **A100-SXM4-40GB**（`rl-server`）上执行 `SFT-v1` 训练与验收的标准步骤。按照 [科研协作范式规范](../AGENTS.md)，本地 Agent 通过 `scripts/remote/sync_code.sh` 推送代码与配置，使用 `scripts/remote/run_tmux.sh` 在独立代码快照中启动 `tmux` 长任务，再经 `scripts/remote/pull_artifacts.sh` 拉回日志与指标到本地分析存档。请按顺序执行，严禁跳过 Pre-flight 校验或更改数据配比。

---

## 1. 实验设计背景与核心认知（做之前先读懂为什么）

1. **我们要修哪些基线短板？**（详见 [02-基线评测.md](02-基线评测.md) 与 [06-sft-v1实验方案.md](06-sft-v1实验方案.md)）
   - `Qwen3.5-2B`（64K 基线）：`BFCL core_weighted = 43.48%`，`API-Bank full = 66.53%`；主要病灶是 `parallel = 50.5%`、参数忠实度错 287 次、`live_irrelevance` 误调用 328 次、`API-Bank` 回复轮泄漏 `<tool_call>` 106 次，以及多轮短板（`multi_turn_base = 29.5%` / `miss_param = 15.0%` / `miss_func = 2.5%`）。
   - `Qwen3.5-0.8B`（64K 基线）：`BFCL core_weighted = 34.30%`，`API-Bank full = 57.24%`；`parallel = 31.5%`、`multiple = 63.5%`、`multi_turn_base = 10.5%`。
2. **为什么 `SFT-v1` 只选用 `xlam-60k + toolace`（27,058 条）？**
   - 本阶段是**干净的数据消融起点**：先用这两套最高纯度的函数调用源修复**并行调用（52.3%）、多候选单选（80.4%）、标准 JSON 参数格式与拒答/回复边界（1,498 条 `no_call` + 2,256 条 `3×` 上采样多轮）**；不在 V1 提前掺杂噪声大的数据源。
   - 若 `SFT-v1` 在 `parallel / multiple / API-Bank` 大涨而 `multi_turn_*` 或 `irrelevance` 仍有缺口，再通过 `SFT-v2`（补 `apigen-mt-5k + smoltalk`）或直接进 **OPD / GRPO** 攻克。
3. **代码里已经替你绕开了哪些隐蔽坑？（切勿退回原生写法）**
   - **Qwen3.5 缺失 `{% generation %}` 标记**：原生 `chat_template.jinja` 直接开 `assistant_only_loss=True` 会导致 mask 全 0。必须使用数据集随附的 `qwen35_chat_template_train.jinja`（仅训练期用于计算 mask，渲染与原生模板 100% 字节等价；**推理 merge 后恢复使用基座官方 `chat_template.jinja`**）。
   - **PyArrow 嵌套字典污染**：训练加载必须走 `toolcall.data.loader.load_tokenized_sft_dataset()` 预转为扁平 `input_ids + labels`，严禁改用 `datasets.load_dataset("json")` 读嵌套 `tool_calls.arguments`。
   - **Transformers 5.x 参数变化**：使用 `warmup_steps: 0.05`（浮点比例）和 `dtype=torch.bfloat16`；`AutoModelForCausalLM` 会自动加载为 `Qwen3_5ForCausalLM` 纯文本因果模型。
   - **Packing 与 Step 数 / 显存校准**：开启 `packing=True`（8192）后每条 packed 序列约塞入 ~9 条样本；同时 `Qwen3.5` 词表为 `248,320`，`bs=1` + `gradient_checkpointing=True` + `grad_accum=8`（有效 ~65K tok/step，~375 steps/epoch）可彻底规避 40GB A100 logits OOM；且 `eval_steps: 0.25 / save_steps: 0.25` 会自动按 `0.5 / 1.0 / 1.5 / 2.0 epoch` 均匀评估与存盘。

---

## 2. 同步代码、更新 `.venv` 与核验冻结数据集

### Step 2.1 同步本地代码并更新远端 CLI 入口

```bash
# 1. 在本地执行（不要在服务器 git pull/commit）
bash scripts/remote/sync_code.sh

# 2. 在 rl-server 上确认新命令已注册进 .venv 与 runs/envs/train
cd /root/ToolCall-SFT
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python runs/envs/train/bin/python --no-deps -e .

# 确认训练环境与 flash-linear-attention 就绪
runs/envs/train/bin/python - <<'PY'
import torch, transformers, trl, peft, swanlab, fla
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("torch:", torch.__version__, "tf:", transformers.__version__, "trl:", trl.__version__, "peft:", peft.__version__)
PY
```

*(若 `runs/envs/train` 尚未安装 `PyYAML`，补一句 `uv pip install --python runs/envs/train/bin/python PyYAML`；确保 `.venv` 与 `runs/envs/train` 已登录 SwanLab。)*

### Step 2.2 接收并校验 `data/processed/sft_v1/`

确认本地通过 `scp` 传来的 `sft_v1.tar.gz` 解压在 `data/processed/sft_v1/` 后，运行 SHA-256 与单元测试双校验：

```bash
mkdir -p data/processed
# 如尚未解压：tar -xzf data/processed/sft_v1.tar.gz -C data/processed/

runs/envs/train/bin/python - <<'PY'
import sys
sys.path.insert(0, "src")
from pathlib import Path
from toolcall.training.preflight import verify_data_manifest
print("Verified:", verify_data_manifest(
    Path("data/processed/sft_v1"),
    ["train.jsonl", "val.jsonl", "smoke_v0.jsonl", "qwen35_chat_template_train.jinja"],
))
PY

runs/envs/train/bin/python -m unittest discover -s tests -v
```

**通过条件**：4 个文件 SHA-256 全部通过，且 `tests/` 全部 `OK`（0 failures / 0 skipped）。

---

## 3. 执行顺序：`Pre-flight → E0 烟囱 → E1 (2B) → E2 (0.8B)`

### 阶段 A：开训前 100% Loss Mask、模板 & GPU Kernel Pre-flight 自检

在不消耗训练步数的前提下，用真实 `SFTTrainer` 跑通数据加载、DataLoader 打包、**1 步真实 GPU 前向探针（验证 `causal_conv1d_fn` + `cu_seq_lens_q`）** 并把 batch 解码落盘检查：

```bash
runs/envs/train/bin/toolcall-sft \
  --config configs/train/e0_smoke_2b.yaml \
  --preflight-only
```

- **Kernel 自动激活说明**：配置已开启 `use_kernels: true`，`sft.py` 会在把模型移上 CUDA 并切到 `model.train()` 后显式调用 `model.set_use_kernels(True, mode=Mode.TRAINING)` 激活 `kernels-community/mamba-ssm` 等 Hub kernel（之前仅安装 `kernels` 而未调用 `set_use_kernels` 不会生效）。
- **如果 `--preflight-only` 在 GPU 前向探针阶段因 Hub kernel / `cu129` ABI 报错**：不要硬调 `causal-conv1d` 源码编译，直接改走已实测验证的零风险非拼接兜底（自动设 `packing=False, padding_free=False, bs=2, accum=16`，避免任何 `cu_seqlens` 依赖与样本串扰，切记不可用 `packing=True + padding_free=False`）：
  ```bash
  runs/envs/train/bin/toolcall-sft \
    --config configs/train/e0_smoke_2b.yaml \
    --preflight-only --no-packing --no-use-kernels
  ```
  *(若兜底方案通过，后续 `E0 / E1 / E2` 统一追加 `--no-packing --no-use-kernels` 运行即可；均值 843 tokens 的 27k 数据集在 A100 上依然很快。)*

检查最新生成的 `runs/train/e0-smoke-2b-*/preflight_loss_mask_check.json` 与 `preflight_loss_mask_batches.txt`：
- **必须确认 `active_ratio` 在 `0.10 ~ 0.20` 之间**（或 `--no-packing` 下 padded batch `0.06 ~ 0.18`）；
- **确认 `forward_check.status == "passed"`** 且 loss 为有限正数；
- **人工抽看 `preflight_loss_mask_batches.txt`**：确认 `[TRAIN]` 区间**严格只有** `<tool_call>...<|im_end|>` 和自然语言回复 `<|im_end|>`，绝无 `<|im_start|>system`、`<|im_start|>user`、`<tool_response>` 或 `<think>\n\n</think>\n\n`。

---

### 阶段 B：`E0-smoke` 全链路烟囱闭环

验证 SFT 训练、SwanLab 记录、LoRA 合并、vLLM 工具解析与 BFCL 8 类快筛。以下长任务从**本地仓库根目录**发起：

```bash
bash scripts/remote/run_tmux.sh --sync sft-e0-2b -- \
  runs/envs/train/bin/python -m toolcall.training.sft --config configs/train/e0_smoke_2b.yaml
```

训练结束后，`auto_merge_best: true` 将最佳权重导出到该 run 的 `merged-best/`。**训练仍用纯文本 CausalLM；推理导出恢复官方 `Qwen3_5ForConditionalGeneration` 外层、合并后的语言模型权重和底座视觉权重**。vLLM 0.19.1 使用该外层注册模型，文本工具评测以 `--language-model-only` 启动；不能靠给旧纯文本目录改 `config.json` 解决权重键不匹配。

```bash
# 将路径设为刚完成的远端 run，避免误用历史 run。
merged_model="runs/train/<本次E0运行目录>/merged-best"
bash scripts/remote/run_tmux.sh vllm-e0-2b -- \
  runs/envs/serve/bin/vllm serve "$merged_model" --served-model-name sft-v1-e0-smoke-2b \
  --host 127.0.0.1 --port 18003 --tensor-parallel-size 1 --dtype bfloat16 \
  --language-model-only --max-model-len 65536 --gpu-memory-utilization 0.85 \
  --max-num-seqs 256 --max-num-batched-tokens 32768 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --generation-config vllm --seed 0 --enable-prefix-caching
```

如有历史 checkpoint，仅需用修复后的 `python -m toolcall.training.merge_lora --adapter-dir … --output-dir …` 在新快照中重新导出到独立目录，不需要重训。服务就绪后，检查 `/v1/models`，再用含工具 schema 的请求确认 `tool_choice=auto` 返回结构化 `message.tool_calls`，无原始 XML 泄漏。通过后启动快筛：

```bash
bash scripts/remote/run_tmux.sh eval-e0-bfcl -- \
  .venv/bin/python -m toolcall.evaluation.cli bfcl-v4 --categories \
  simple_python multiple parallel parallel_multiple irrelevance \
  multi_turn_base multi_turn_miss_func multi_turn_miss_param \
  --variant e0-smoke --model sft-v1-e0-smoke-2b \
  --base-url http://127.0.0.1:18003/v1 --concurrency 128 --timeout 300 --gpu-index 0
```

评测后保留摘要与日志，再停止本次 E0 的 vLLM 会话释放显存。E0 如出现 OOM、服务请求错误、解析失败或 loss 异常，先定位并复测，不直接启动正式训练。实测见 [E0 重跑记录](../reports/03-sft/01-e0-smoke-run.md)。

### 阶段 C：正式训练 `E1` (`Qwen3.5-2B`) 与 `E2` (`Qwen3.5-0.8B`)

1. **停掉 E0 的 vLLM 进程释放显存**，确认 `nvidia-smi` 空闲。
2. **启动 `E1`（2B 主力实验）**：
   ```bash
   runs/envs/train/bin/toolcall-sft --config configs/train/sft_v1_2b_lora.yaml
   ```
   - 自动每 `0.5` epoch 在 `val.jsonl` 评估一次并保存共 4 个 checkpoint（`0.5 / 1.0 / 1.5 / 2.0 epoch`）。
   - 训练完后查看 `<e1_run_dir>/train_summary.json`、`preflight_loss_mask_check.json` 与 SwanLab（项目 `toolcall-sft`）曲线。
3. **批量合并 2B 的 4 个半轮 Checkpoint 并跑 BFCL 8 类快筛（1,840 题）**：
   ```bash
   e1_dir=$(ls -td runs/train/sft-v1-2b-lora-* | head -n 1)
   runs/envs/train/bin/toolcall-merge-lora \
     --model-id Qwen/Qwen3.5-2B \
     --run-dir "$e1_dir" --all-checkpoints
   ```
   对每个 `merged-checkpoint-*` 依次起 vLLM 并跑 8 类快筛（每次仅几分钟）：
   ```bash
   # 示例：替换 ckpt_tag（如 ckpt-0p5ep / ckpt-1p0ep / ckpt-1p5ep / ckpt-2p0ep）与对应 merged 目录
   .venv/bin/toolcall-eval bfcl-v4 --categories \
     simple_python multiple parallel parallel_multiple irrelevance \
     multi_turn_base multi_turn_miss_func multi_turn_miss_param \
     --variant "sft-v1-${ckpt_tag}" --model "sft-v1-2b-${ckpt_tag}" \
     --base-url http://127.0.0.1:18003/v1 --concurrency 128 --timeout 300 --gpu-index 0
   ```
4. **对 2B 最佳 Checkpoint 补跑全量正式验收（必跑！）**：
   用选出的最佳 `merged-checkpoint-*` 保持 vLLM 运行，补齐完整基线同口径全量基准测试：
   ```bash
   # 1) BFCL core（17 类 4,441 题）
   .venv/bin/toolcall-eval bfcl-v4 --profile core \
     --variant "sft-v1-best-${ckpt_tag}" --model "sft-v1-2b-${ckpt_tag}" \
     --base-url http://127.0.0.1:18003/v1 --concurrency 128 --timeout 300 --gpu-index 0

   # 2) API-Bank full（314 段对话 / 1,281 步）
   .venv/bin/toolcall-eval api-bank --profile full \
     --variant "sft-v1-best-${ckpt_tag}" --model "sft-v1-2b-${ckpt_tag}" \
     --base-url http://127.0.0.1:18003/v1 --concurrency 96 --timeout 300 --gpu-index 0
   ```
5. **同理完成 `E2`（`Qwen3.5-0.8B`）训练、8 类选优与最佳 Checkpoint 全量补跑**：
   ```bash
   # 停掉 vLLM 后启动 0.8B 训练
   runs/envs/train/bin/toolcall-sft --config configs/train/sft_v1_0_8b_lora.yaml

   # 批量 merge 并快筛选出 0.8B 最佳 ckpt 后，同样补跑 BFCL --profile core + API-Bank --profile full
   e2_dir=$(ls -td runs/train/sft-v1-0.8b-lora-* | head -n 1)
   runs/envs/train/bin/toolcall-merge-lora \
     --model-id Qwen/Qwen3.5-0.8B \
     --run-dir "$e2_dir" --all-checkpoints
   ```

---

## 4. 关键监控指标与验收标准清单

本地 Agent 在跟踪远端 `tmux` 运行、验收并生成本地报告时逐项核对以下指标：

### 4.1 训练期必须记录与核对的指标（来自 `train_summary.json` 与 SwanLab）
- **Pre-flight Loss Mask 校验**：`active_ratio`（预期 `0.11–0.18`）、`has_cu_seq_lens_q=true`、`byte_identical=true`、`status="passed"`。
- **Loss 收敛曲线**：`train/loss` 起点与终点、`eval/loss`（在 `0.5 / 1.0 / 1.5 / 2.0 epoch` 的数值，观察何时达到最低点/是否过拟合回升）。
- **Token 准确率与梯度稳定性**：`train/mean_token_accuracy`、`eval/mean_token_accuracy`、`train/grad_norm`（是否平稳无连续 `>5` 尖峰或 NaN）、`train/learning_rate`。
- **硬件与效率**：`peak_vram_gib`（应 `< 24 GiB`）、`wall_seconds`、`train_samples_per_second`、`flash_linear_attention=true`。

### 4.2 评测验收标准（对比 `reports/01-baseline/01-baseline-report.md`）
- **2B 基线参考**：`BFCL core_weighted = 43.48%`（`parallel 50.5%` / `multiple 82.5%` / `irrelevance 78.75%` / `live_irrelevance 68.67%` / `multi_turn_base 29.5%`）、`API-Bank full = 66.53%`（`response_leaked_tool_call = 106`）。
- **0.8B 基线参考**：`BFCL core_weighted = 34.30%`（`parallel 31.5%` / `multiple 63.5%` / `multi_turn_base 10.5%`）、`API-Bank full = 57.24%`（`response_leaked_tool_call = 79`）。
- **验收与选优原则**：
  1. 绝不可仅看 `eval_loss` 选 ckpt，必须结合 8 类 BFCL 快筛分数（尤其是 `parallel` 涨幅 vs `irrelevance` 留存）。
  2. 全量补跑必须记录 **0 系统级报错**（或逐一归因 `request_errors` / `context_rejections` / `length_stops`），重点对比 `response_leaked_tool_call` 是否下降及 `multi_turn_*` 的净增益。

### 4.3 完成后的产物回收与本地 Git 存档要求
- **勿将模型权重、`runs/` 或 `data/` 提交到 Git**（保持 `.gitignore` 规则），且**不要在服务器上执行 Git commit / push**。
- 通过 `bash scripts/remote/pull_artifacts.sh` 将远端 `runs/train/` 摘要、`runs/services/` 日志和 `runs/eval/` 指标拉回本地分析。
- 在本地 `reports/03-sft/` 新增一份按序号编排的 `SFT-v1` 实验与评测汇总报告（例如 `reports/03-sft/01-sft-v1-report.md`），附带：
  - `E0 / E1(2B) / E2(0.8B)` 的 `train_summary.json` 核心指标、`val_loss` 曲线与最佳 ckpt 判定依据；
  - `BFCL 8 类快筛（4 个 ckpt 对比表）`；
  - 最佳 ckpt 的 **`BFCL --profile core`（4,441 题）+ `API-Bank --profile full`（1,281 步）** 全量逐类对比基线表与错误归因；
  - 所有 SwanLab Run ID（训练 + 评测）。
- 在本地将代码改动与 `reports/03-sft/` 文档 commit + push 到 `origin/main` 存档，并据此决策进 **OPD/GRPO** 还是补 **`SFT-v2`**。
