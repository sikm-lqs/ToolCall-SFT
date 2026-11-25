# GRPO v2 Smoke (`grpo-v2-smoke-2b-colocate`) 双卡 Colocate 实验分析报告

- **实验时间**：2026-11-25 19:36:38 ~ 19:51:20（UTC+8）
- **实验标识**：`grpo-v2-smoke-2b-colocate-20261125T113638Z`
- **SwanLab 追踪**：<https://swanlab.cn/@sikm/toolcall-sft/runs/0ljt1ie9>
- **硬件与架构**：`rl-server`，双卡 `NVIDIA A100-SXM4-40GB`，**`vllm_mode="colocate"` + DDP (`world_size=2`) + `vllm_enable_sleep_mode=true`**
- **数据集**：`data/processed/grpo_v2_smoke/`（难度挖掘分层抽样：`train=240`、`val=60`，`91.25%` 训练组 `reward_std > 0`）
- **基座与起点**：`merged-checkpoint-68` + 新建 LoRA（`r=16, lora_alpha=32, lora_dropout=0.05, target_modules=all-linear`）

---

## 1. 实验目的与 Colocate 配置概览

本轮实验用于同时验证两项关键升级，作为启动全量 `grpo_v2`（1,500 条）前的最终预演：
1. **架构升级**：由 `server` 物理隔离（单卡生成 + 单卡训练）切换为 **双卡 DDP + `vLLM colocate`（`tp=1, world_size=2`, `gpu_memory_utilization=0.38`, `enable_sleep_mode=true`）**，消除单卡瓶颈与网络/NCCL 跨进程开销；
2. **数据与探索升级**：采用基于基线 4-shot 挖矿的 `grpo_v2_smoke`（240 train / 60 dual-tier val），并将采样提升至 **`G=8, temperature=0.9`**，同时开启 **`eval_on_start: true`** 获取严格 Step 0 基线。

| 配置项 | `grpo-smoke` (Server 基准) | **`grpo-v2-smoke-colocate` (本轮)** |
| --- | --- | --- |
| 架构模式 | `server`（GPU 0 推理 / GPU 1 训练） | **`colocate` + DDP**（双卡并行推理 + 并行训练） |
| 睡眠卸载 (`sleep_mode`) | N/A | **`true`**（`CuMemAllocator` level 2） |
| vLLM 显存配额 | GPU 0 独占 `0.85` | 双卡各 **`0.38`**（`max_model_len=6144`） |
| `train` / `val` 规模 | 300 / 60（随机抽样） | **240 / 60**（`grpo_v2` 难度筛选：`190 mid + 50 zero` / `48 mined + 12 anchor`） |
| 有效批大小 / `G` | `bs=4, accum=8`, `G=4` → `8 prompts (32 comp)/step` | `bs=4, accum=8, world=2`, **`G=8`** → **`8 prompts (64 comp)/step`** |
| 采样参数 | `T=0.7, top_p=0.95, top_k=512` | **`T=0.9`**, `top_p=0.95, top_k=512` |
| 训练步数 / 验证点 | 37 steps；eval @ 19, 37 | **30 steps**；eval @ **0 (on_start), 15, 30** |

---

## 2. Colocate 工程适配细节

为确保 `Qwen3.5-2B` + `torchrun` DDP + `vLLM colocate` 在 `rl-server` 稳定运行，在 [grpo.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/src/toolcall/training/grpo.py) 与 [run_grpo_colocate.sh](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/scripts/remote/run_grpo_colocate.sh) 完成以下适配（远端 `20/20` 项 GRPO 单元测试通过）：
1. **Rank 设备绑定与共享时间戳**：`torch.cuda.set_device(LOCAL_RANK)` + `model.to(cuda:LOCAL_RANK)`，并通过 `RUN_STAMP` 与 rank-0 guard 保证双进程落盘目录、SwanLab 实例唯一无竞争。
2. **Colocate 超参透传与整形**：透传 `vllm_tensor_parallel_size`、`vllm_gpu_memory_utilization`、`vllm_max_model_length`、`vllm_enable_sleep_mode`、`eval_on_start`、`ddp_find_unused_parameters=false`，并修正 `eval_steps/save_steps >= 1` 的整型解析。
3. **in-GPU 权重前缀对齐**：复用 `install_vllm_param_name_prefix_adapter` 将 `Qwen3_5ForCausalLM` 的 `model.*` 权重直接喂给每卡本地 `LLM`（`Qwen3_5ForConditionalGeneration`）的 `load_weights`。
4. **Sleep/Wake 预验通过**：预验证确认 `vLLM sleep(level=2)` 在 `0.31 s` 内释放 `10.59 GiB`，`wake_up` 仅耗时 `0.22 s`。

---

## 3. 效率、显存与架构对比（Colocate vs Server）

### 3.1 时间与吞吐对比

- **本轮总耗时**：`wall_seconds = 765.94 s`（12 分 46 秒），其中包含 **3 次完整验证**（Step 0 含首次 Triton 前向预热耗时 `114.54 s`，Step 15 耗时 `59.07 s`，Step 30 耗时 `61.98 s`，合计验证耗时 `235.6 s`）及 **前 6 步内的一次性 Triton 反向 kernel autotune**（Step 2 `53.00 s`、Step 6 `32.22 s`）。
- **稳态训练单步速度**（Step 8–30，已缓存 Triton 反向核后）：平均 **`12.80 s/step`**（范围 **`10.19 s – 25.13 s`**，其中含 `G=8 × 8 = 64 completions/step`）。

| 性能维度 | `Server`（Smoke v1, `G=4`） | **`Colocate`（Smoke v2, `G=8`）** | 提升 / 变化 |
| --- | ---: | ---: | --- |
| 单步采样规模 | `32 completions/step` | **`64 completions/step`** | **2.0×** |
| 稳态单步耗时 | `10.90 s/step` | **`12.80 s/step`**（最低 `10.19 s`） | 仅增加 `17%` |
| **等效生成+训练通量** | **~`2.94 comp/s`** | **~`5.00 comp/s`** | **+70% 有效算力利用率** |
| **GPU 0 显存峰值（`nvidia-smi`）** | `39,781 MiB`（**97.1%**，紧贴 OOM） | **`21,385 MiB`**（**52.2%**，睡后 `8.2–11.5 GiB`） | **释放 ~`18.4 GiB` 安全余量** |
| **GPU 1 显存峰值（`nvidia-smi`）** | `17,337 MiB`（42.3%） | **`21,413 MiB`**（**52.3%**，PyTorch 峰值 `17.676 GiB`） | **双卡完全对称均衡** |
| **GPU 实测利用率** | GPU 0 间歇闲置 / GPU 1 轮替等待 | **双卡同时 `98%–100%` 满载** | 彻底消除单卡等待气泡 |
| **TIS 重要性采样比均值** | `0.728 – 0.922` | **`0.993 – 1.029`（均值 `1.0048`）** | **近乎完美对齐（~1.00）** |

- **全量 `grpo_v2`（1,500 条）耗时外推**：在相同 `G=8`、`188` steps 配置下，纯训练约 `188 × 12.8 s ≈ 40 分钟`，加上 4–5 次验证约 `4–5 分钟`，全量运行预计仅需 **~45 分钟**即可完成 **`12,000` 次采样生成 + 全步长更新**。

---

## 4. 训练动态与验证集效果分析

### 4.1 训练过程明细（`logging_steps=2`）

| Step | LR | Loss | Grad Norm | KL | Entropy | Reward | Reward Std | Zero-Std 比例 | Format | Call Corr | Non-Call Corr | Mean Len | Step Time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5.00e-6 | -0.00591 | 0.2739 | 0.000000 | 0.1284 | 2.523 | 2.009 | 0.2500 | 1.000 | +2.232 | -1.031 | 76.9 | 53.00s |
| 4 | 9.97e-6 | -0.00599 | 0.4067 | 0.000082 | 0.0966 | 2.319 | 2.050 | 0.1250 | 1.000 | +1.745 | -1.250 | 73.6 | 16.90s |
| 6 | 9.72e-6 | +0.00398 | 0.4039 | 0.000315 | 0.2184 | 1.712 | 2.194 | 0.3125 | 1.000 | +1.129 | +0.375 | 58.2 | 32.22s |
| 8 | 9.23e-6 | +0.00450 | 0.3803 | 0.000129 | 0.0866 | 2.931 | 1.614 | 0.1875 | 1.000 | +2.283 | +0.875 | 58.3 | 11.41s |
| 10 | 8.54e-6 | -0.01876 | 0.4296 | 0.000703 | 0.1715 | 1.958 | 2.350 | 0.1250 | 1.000 | +1.812 | -0.562 | 75.0 | 11.68s |
| 12 | 7.66e-6 | +0.01588 | 0.2868 | 0.000113 | 0.0295 | 3.148 | 1.172 | 0.0625 | 0.992 | +2.156 | N/A | 73.0 | 25.13s |
| 14 | 6.65e-6 | -0.01674 | 0.3402 | 0.001426 | 0.2048 | 3.129 | 1.334 | 0.2500 | 1.000 | +2.031 | +2.250 | 80.6 | 11.38s |
| 16 | 5.56e-6 | -0.01873 | 0.2530 | 0.001668 | 0.0832 | 3.161 | 1.238 | 0.1875 | 0.992 | +2.181 | +1.875 | 92.3 | 11.74s |
| 18 | 4.44e-6 | -0.01177 | 0.2887 | 0.000847 | 0.2078 | 3.115 | 1.334 | 0.3750 | 1.000 | +1.783 | +2.766 | 71.0 | 10.19s |
| 20 | 3.35e-6 | -0.00911 | 0.0671 | 0.000488 | 0.1745 | 3.605 | 0.746 | 0.5000 | 1.000 | +2.509 | +2.922 | 81.4 | 12.64s |
| 22 | 2.34e-6 | -0.01389 | 0.3032 | 0.002713 | 0.1161 | 2.382 | 2.085 | 0.1875 | 0.984 | +1.272 | +1.094 | 104.4 | 12.07s |
| 24 | 1.46e-6 | +0.00829 | 0.2422 | 0.001831 | 0.2552 | 2.920 | 1.633 | 0.3125 | 1.000 | +1.568 | +2.625 | 68.9 | 10.76s |
| 26 | 7.66e-7 | -0.02641 | 0.3177 | 0.002128 | 0.1644 | 3.023 | 1.562 | 0.2500 | 1.000 | +1.846 | +2.422 | 74.4 | 11.34s |
| 28 | 2.81e-7 | -0.04911 | 0.3130 | 0.004682 | 0.2927 | 2.805 | 1.841 | 0.3125 | 1.000 | +1.896 | +1.802 | 83.0 | 11.62s |
| 30 | 3.14e-8 | -0.00986 | 0.2873 | 0.003303 | 0.2159 | 3.588 | 1.016 | 0.5000 | 1.000 | +2.617 | +2.562 | 61.1 | 13.70s |

### 4.2 关键训练指标对比（Smoke v1 vs Smoke v2 Colocate）

| 汇总指标 | Smoke v1（`grpo_smoke` + `G=4`） | **Smoke v2（`grpo_v2_smoke` + `G=8`）** |
| --- | ---: | ---: |
| `frac_reward_zero_std` 均值 | `0.8646`（仅 13.5% 组有效） | **`0.2625`（高达 73.75% 组有梯度！）** |
| `reward_std` 均值 | `0.7860` | **`1.6119`（2.05×）** |
| `grad_norm` 均值 | `0.1141`（多次跌破 `0.001`） | **`0.3062`（全部步 `0.067–0.430`）** |
| `reward/format_mean` / `leak_rate` | `1.0000` / `0.0000` | `0.9979` / `0.0010` |
| 保存权重 `lora_B` 更新情况 | `186/186` 非零（`max_abs=0.000155`） | **`186/186` 非零（`max_abs=0.000146`）** |

### 4.3 验证集进步曲线（Step 0 → Step 15 → Step 30）

| Eval Step | Runtime | `eval_reward` | `eval_reward_std` | `eval_zero_std` | `format_mean` | `call_correct` | **`non_call_correct`** | `leak_rate` | `eval_kl` | `is_mean` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0 (基线)** | 114.54s | `2.6006` | `1.3126` | `0.4333` | `0.9917` | `2.0890` | **`+0.3250`** | `0.0083` | `0.000000` | `0.9938` |
| **15** | 59.07s | `2.9200` | `1.1599` | `0.5000` | `1.0000` | `2.0001` | **`+1.6750`** | `0.0000` | `0.001451` | `0.9983` |
| **30** | 61.98s | **`3.0073`** | `1.1679` | `0.4667` | `0.9958` | `1.9446` | **`+2.1083`** | `0.0042` | `0.005456` | `1.0117` |

**客观分析**：
1. **`eval_reward` 仅用 30 步即净涨 `+0.4067`**（`2.6006 → 3.0073`），证明了基于难度挖矿构造的 `grpo_v2` 配合 `colocate (G=8)` 能够让策略迅速学到真实增量。
2. **收益主力来自 `non_call` 能力跃升（`+0.3250 → +2.1083`，净涨 `+1.7833`）**：训练前段模型在抗干扰拒答和缺参追问上仍频繁误调（Step 2/4 `noncall = -1.03 / -1.25`），到 Step 14–30 已稳定拉升至 `+2.2 ~ +2.9`。
3. **需关注的微调信号**：`eval_reward/call_correct_mean` 略有小幅回落（`2.0890 → 1.9446`，`-0.1444`），可能原因有二：一是仅 240 条小步数训练下模型先大幅矫正了 `non_call` 误调倾向（可能略微向保守偏移）；二是 `T=0.9` 下 `call` 组探索噪声尚在收敛初期。全量 `grpo_v2`（1,500 条、188 步）具备充足步数同时打磨 `call` 与 `non_call` 平衡，建议在全量训练结束后用 `BFCL + API-Bank + τ²` 做严格端到端门控验证。

---

## 5. 全量 `grpo_v2` 启动建议

基于本轮验证结果，全量 `grpo_v2`（1,500 train / 190 val）**已完全具备启动条件**，建议配置：
- 直接复用本轮验证成功的 **`colocate + sleep_mode=true`、`gpu_memory_utilization=0.38`、`G=8`、`T=0.9`** 架构；
- 全量约 **`188` steps**，预计耗时 **~45 分钟**，每 0.25 epoch（约 47 steps）做一次验证并保存 checkpoint（`eval_on_start: true`）；
- 若希望进一步压低 `call` 早期波动，可在全量配置中继续保持 `beta: 0.02`、`lr: 1.0e-5`（或试 `beta: 0.03`、`lr: 8e-6` 微稳策略）。

---

## 6. 产出索引

- 远端权重（仅留服务器）：`/root/ToolCall-SFT/runs/train/grpo-v2-smoke-2b-colocate-20261125T113638Z/`（`checkpoint-15`、`checkpoint-30`、`final_adapter`）
- 本地回拉状态与日志：
  - [train_summary.json](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/train/grpo-v2-smoke-2b-colocate-20261125T113638Z/train_summary.json)
  - [trainer_state.json](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/train/grpo-v2-smoke-2b-colocate-20261125T113638Z/checkpoint-30/trainer_state.json)
  - [resolved_config.yaml](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/train/grpo-v2-smoke-2b-colocate-20261125T113638Z/resolved_config.yaml)
  - [grpo-v2-smoke-colocate-20261125T113638Z.log](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/logs/tmux/grpo-v2-smoke-colocate-20261125T113638Z.log)
