# GRPO Smoke (`grpo-smoke-2b`) 双卡闭环实验分析报告

- **实验时间**：2026-11-25 17:56:46 ~ 18:07:45（UTC+8）
- **实验标识**：`grpo-smoke-2b-20261125T095728Z`
- **SwanLab 追踪**：<https://swanlab.cn/@sikm/toolcall-sft/runs/lq4wmdte>（2,453 条记录已上传完成）
- **硬件拓扑**：`rl-server`，双卡 `NVIDIA A100-SXM4-40GB`（PCIe `PHB` 互联），`vllm_mode="server"` 物理隔离（GPU 0 专职 vLLM rollout，GPU 1 专职 `GRPOTrainer` + LoRA 更新）
- **基座与起点**：`SFT-v3` `Qwen3.5-2B`（LoRA `r=64` `checkpoint-68` merge 后静态底模 `merged-checkpoint-68`）+ 新建 LoRA（`r=16, lora_alpha=32, lora_dropout=0.05, target_modules=all-linear`）

---

## 1. 实验目的与配置概览

本轮 Smoke 实验旨在验证双卡物理隔离同步架构下，基于 `ToolRL + 3.4-B 对称奖励` 的 GRPO 完整训练闭环（vLLM rollout、NCCL 权重同步、重要性采样校正、LoRA 更新与 KL 正则、mid-run eval、checkpoint 落盘与 SwanLab 监控），并采集吞吐、显存与奖励动态作为放大至全量 `grpo_v1`（1,500 条）前的决策依据。

| 配置维度 | 参数值 |
| --- | --- |
| 训练集 / 验证集 | `data/processed/grpo_smoke/`：train `300` 条、val `60` 条（SHA-256 校验通过） |
| 采样与批次 | `G=4`、`per_device_train_batch_size=4`、`gradient_accumulation_steps=8`（有效 `8 prompts/step`、即 `32 completions/step`，1 epoch = `37` steps） |
| 采样参数 | `temperature=0.7`、`top_p=0.95`、`top_k=512`、`max_completion_length=512`、`importance_sampling_correction=true` |
| 优化器与 RL 参数 | `loss_type="grpo"`、`beta=0.02`、`lr=1.0e-5`（cosine、`warmup_steps=0.05`）、`weight_decay=0.01`、`max_grad_norm=1.0` |
| vLLM 服务端（GPU 0） | `trl.scripts.vllm_serve`（`vllm==0.30.0`），`gpu_memory_utilization=0.85`、`max_model_len=6144`、`enable_prefix_caching=true`、`--language-model-only`、NCCL `group_port=51216` |
| 训练端（GPU 1） | `bf16=true`、`gradient_checkpointing=true`、`use_kernels=true`（`kernels==0.18.2` + `fla==0.6.0`）、`logging_steps=2`、`eval_steps=0.5`、`save_steps=0.5` |

### 数据集能力分桶构成（`grpo_smoke`）

| 能力分桶（`bucket`） | `task_type` | Train（300 条） | Val（60 条） | 数据来源构成（Train） |
| --- | --- | ---: | ---: | --- |
| `single_turn_call` | `call` | 120（40.0%） | 24（40.0%） | `xlam-60k` + `toolace` |
| `parallel_call` | `call` | 45（15.0%） | 9（15.0%） | `xlam-60k` + `toolace` |
| `multi_turn_continue` | `call` | 24（8.0%） | 5（8.3%） | `toolace_multi_turn` |
| `multi_turn_miss_func_recover` | `call` | 18（6.0%） | 4（6.7%） | `xlam-60k:miss_func_recover`（15）+ `toolace:miss_func_recover`（3） |
| `multi_turn_post_tool_response` | `non_call` | 18（6.0%） | 3（5.0%） | `toolace_multi_turn` |
| `irrelevance_refusal` | `non_call` | 45（15.0%） | 9（15.0%） | `xlam-60k:adv_1tool_irrelevance`（23）+ `toolace_irrelevant_refusal`（20）+ `toolace:adv_1tool_irrelevance`（2） |
| `missing_param` | `non_call` | 30（10.0%） | 6（10.0%） | `toolace_missing_param`（30） |
| **合计** | `call` 207 / `non_call` 93 | **300** | **60** | `xlam-60k` 178 / `toolace` 122 |

---

## 2. 联调暴露与修复的工程问题

此前仅靠单卡 `preflight` 和原生 HF 生成未能覆盖双卡 `vllm_serve` + NCCL 联调路径；本轮真实启动过程中共发现并修复 4 个工程问题：

1. **`trl.scripts.vllm_serve` 布尔参数解析**（[run_grpo_dual.sh](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/scripts/remote/run_grpo_dual.sh)）：`TrlParser(ScriptArguments)` 要求 `--enable-prefix-caching true`，裸 flag 会报错 `expected one argument`。
2. **`runs/envs/rl` 缺少 `kernels` 包**：`grpo_smoke_2b.yaml` 启用了 `use_kernels: true`，而此前只部署了 `fla==0.6.0`。补装 `kernels==0.18.2`（及 `tomlkit==0.15.3`）后，成功离线激活本地缓存的 `kernels-community/mamba-ssm` 与 `fla` 训练核（`Activated HF Hub kernels in Mode.TRAINING on CUDA`）。
3. **TRL 1.18.2 重要性采样 `top_k` 校验**（[grpo.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/src/toolcall/training/grpo.py)、[grpo_smoke_2b.yaml](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/configs/train/grpo_smoke_2b.yaml)、[grpo_v1_2b.yaml](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/configs/train/grpo_v1_2b.yaml)）：在 `use_vllm=True`、`vllm_importance_sampling_correction=True` 且 `top_p=0.95 < 1.0` 时，TRL 要求显式提供正数 `top_k`。补充透传并设置 `top_k: 512`。
4. **Qwen3.5 VL/Text 权重命名空间不一致（关键发现）**：
   - `merged-checkpoint-68/config.json` 保留上游架构声明 `Qwen3_5ForConditionalGeneration`，vLLM 即使开启 `--language-model-only` 仍按多模态命名加载权重（期望 `model.language_model.*` 映射至 `language_model.model.*`）；
   - 训练端通过 `AutoModelForCausalLM` 仅加载纯文本主干 `Qwen3_5ForCausalLM`（320 个张量全部以 `model.*` 开头，核对与 safetensors 中 `model.language_model.*` 严格 1:1 对齐、无缺失无多余）；
   - 在 [grpo.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/src/toolcall/training/grpo.py) 新增 `install_vllm_param_name_prefix_adapter` 并在 [test_grpo_training.py](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/tests/test_grpo_training.py) 增加单测（14/14 全部通过）。修复后 vLLM 服务端日志记录 **`/update_weights` 38 次、`/reset_prefix_cache` 38 次、`/v1/completions` 157 次，HTTP 500 与 ERROR 均为 0**。

---

## 3. 训练耗时、吞吐与双卡显存指标

### 3.1 时间与吞吐

- **总训练墙钟时间**（`trainer.train()`，含 37 步训练 + 第 19/37 步两次 60 条验证 + 两次 checkpoint 保存）：**`616.45 s`（约 10 分 16 秒）**（`train_runtime=616.20 s`，`train_samples_per_second=0.487`，`train_steps_per_second=0.06`）。
- **单步耗时分布**：
  - Step 2 耗时 **`51.12 s`**（含首次 NCCL 权重同步建图与首批 rollout 冷启动）；
  - 稳态训练阶段（Step 4–36）：平均 **`10.90 s/step`**（范围 **`7.44 s – 26.20 s/step`**；Step 22 因该批出现 `max_length=397.5` 的较长生成而升至 `26.20 s`）。
- **验证阶段耗时**：Step 19 eval（60 prompts × G=4 = 240 completions）耗时 **`54.81 s`**，Step 37 eval 耗时 **`54.36 s`**。
- **全量耗时外推基准**：按相同架构跑 `grpo_v1`（1,500 prompts、188 steps、4 次 eval），在相同 `G=4` 下预计约 **40–45 分钟**；若放大 `G` 则按 rollout 生成与训练前反向比例相应增加。

### 3.2 双卡显存实测（`nvidia-smi` 与 PyTorch 统计）

| GPU / 进程 | 阶段与用途 | 观测显存占用 | 占比（总计 `40,960 MiB`） | 备注 |
| --- | --- | ---: | ---: | --- |
| **GPU 0**（`vllm_serve`） | CUDA Graph 捕获完成后 | `34,157 MiB` | 83.4% | `gpu_memory_utilization=0.85` |
| **GPU 0**（`vllm_serve`） | NCCL 同步 + rollout 运行态 | **`39,777 – 39,781 MiB`** | **97.1%** | 余量仅 ~`1.18 GiB`，需注意 OOM 安全边界 |
| **GPU 1**（`GRPOTrainer`） | PyTorch `max_memory_allocated()` | `6.465 GiB` | — | 张量级峰值分配 |
| **GPU 1**（`GRPOTrainer`） | 训练启动 → 稳态训练 → eval | `7,679 → 15,021 →` **`17,337 MiB`** | **42.3%** | 仍有 ~`23.6 GiB` 富余显存 |

> [!IMPORTANT]
> GPU 0 在 `gpu_memory_utilization=0.85` 下加上 NCCL 通信缓冲区与 CUDA context，实际 `nvidia-smi` 显存已逼近 `39,781 / 40,960 MiB`（97.1%）。后续跑更长序列或更大并发时，建议将 `gpu_memory_utilization` 降至 **`0.75 – 0.78`** 以避免 OOM 抖动；与此同时 GPU 1 尚有 ~23.6 GiB 闲置显存，具备增大 `G` 或 `per_device_train_batch_size` 的空间。

---

## 4. 训练过程曲线与指标分析

### 4.1 训练步指标明细（`logging_steps=2`）

| Step | Epoch | LR | Loss | Grad Norm | KL | Entropy | Reward | Reward Std | Zero-Std 比例 | Format | Call Corr | Non-Call Corr | Mean Len | Step Time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.054 | 5.00e-6 | +0.01675 | 0.1322 | 0.000000 | 0.0913 | 3.013 | 2.053 | 0.8125 | 1.000 | 2.184 | 1.375 | 84.1 | 51.12s |
| 4 | 0.108 | 9.98e-6 | +0.00459 | 0.1781 | 0.000109 | 0.0925 | 3.506 | 0.837 | 0.7500 | 1.000 | 2.259 | 3.000 | 77.8 | 11.61s |
| 6 | 0.162 | 9.82e-6 | +0.01754 | 0.0005 | 0.000147 | 0.1309 | 3.766 | 0.606 | 0.8125 | 1.000 | 2.719 | 2.812 | 67.1 | 11.20s |
| 8 | 0.216 | 9.50e-6 | +0.00259 | 0.1369 | 0.000131 | 0.1212 | 3.514 | 0.827 | 0.9375 | 1.000 | 2.312 | 2.917 | 68.7 | 9.22s |
| 10 | 0.270 | 9.05e-6 | +0.01530 | 0.0008 | 0.000252 | 0.1791 | 3.309 | 1.236 | 0.9375 | 1.000 | 1.960 | 2.667 | 56.9 | 8.14s |
| 12 | 0.324 | 8.46e-6 | +0.01430 | 0.1260 | 0.000278 | 0.1475 | 3.606 | 1.184 | 0.8750 | 1.000 | 2.764 | 2.625 | 50.7 | 7.44s |
| 14 | 0.378 | 7.75e-6 | +0.00666 | 0.2082 | 0.000447 | 0.1168 | 3.719 | 0.756 | 0.9375 | 1.000 | 2.850 | 2.500 | 58.8 | 8.25s |
| 16 | 0.432 | 6.97e-6 | +0.00595 | 0.1162 | 0.000472 | 0.1082 | 3.852 | 0.654 | 0.8750 | 1.000 | 3.000 | 2.469 | 60.7 | 8.64s |
| 18 | 0.486 | 6.11e-6 | -0.00619 | 0.1974 | 0.000332 | 0.1427 | 3.743 | 0.616 | 0.8750 | 1.000 | 2.689 | 2.833 | 50.5 | 8.41s |
| 20 | 0.541 | 5.22e-6 | -0.00386 | 0.1441 | 0.000103 | 0.0443 | 3.911 | 0.307 | 0.8750 | 1.000 | 2.869 | 3.000 | 58.0 | 10.20s |
| 22 | 0.595 | 4.33e-6 | -0.00039 | 0.0006 | 0.001348 | 0.1285 | 3.545 | 0.989 | 0.9375 | 1.000 | 2.496 | 2.625 | 81.4 | 26.20s |
| 24 | 0.649 | 3.45e-6 | +0.00373 | 0.3989 | 0.000199 | 0.0896 | 3.742 | 0.741 | 0.7500 | 1.000 | 2.697 | 3.000 | 59.9 | 10.05s |
| 26 | 0.703 | 2.63e-6 | -0.00521 | 0.1428 | 0.000506 | 0.2576 | 3.553 | 0.845 | 0.8750 | 1.000 | 1.829 | 2.958 | 63.6 | 10.04s |
| 28 | 0.757 | 1.88e-6 | +0.00203 | 0.0685 | 0.000094 | 0.0333 | 3.969 | 0.177 | 0.9375 | 1.000 | 2.964 | 3.000 | 78.6 | 12.13s |
| 30 | 0.811 | 1.23e-6 | +0.00429 | 0.0909 | 0.000102 | 0.0354 | 3.745 | 0.625 | 0.7500 | 1.000 | 2.731 | 3.000 | 72.0 | 10.02s |
| 32 | 0.865 | 7.08e-7 | -0.00377 | 0.0003 | 0.000137 | 0.0819 | 3.562 | 0.777 | 0.8750 | 1.000 | 2.434 | 3.000 | 51.7 | 9.27s |
| 34 | 0.919 | 3.19e-7 | +0.00001 | 0.0006 | 0.000351 | 0.1293 | 3.756 | 0.444 | 1.0000 | 1.000 | 2.675 | 3.000 | 67.9 | 13.10s |
| 36 | 0.973 | 8.04e-8 | +0.00207 | 0.1112 | 0.000565 | 0.1603 | 3.772 | 0.476 | 0.7500 | 1.000 | 2.684 | 2.917 | 58.2 | 11.42s |

- **汇总均值**（18 个训练日志点）：`reward=3.6434`、`reward_std=0.7860`、`frac_reward_zero_std=0.8646`、`reward/format_mean=1.0000`、`reward/leak_rate=0.0000`、`reward/call_correct_mean=2.5620`、`reward/non_call_correct_mean=2.7610`、`kl=0.0003`、`entropy=0.1161`、`completions/mean_length=64.81`。

### 4.2 验证集指标（`eval_steps=0.5`）

| Eval Step | Epoch | Eval Runtime | Eval Reward | Eval Reward Std | Eval Zero-Std | Format | Call Corr | Non-Call Corr | Leak Rate | Clipped Ratio | Eval KL | Eval Entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 19 | 0.51 | 54.81s | `3.2636` | `0.2193` | `0.8167` | `0.9833` | `2.3409` | `2.1389` | `0.0167` | `0.0167` | `0.000362` | `0.1265` |
| 37 | 1.00 | 54.36s | `3.2056` | `0.2270` | `0.8500` | `0.9833` | `2.2817` | `2.0833` | `0.0167` | `0.0167` | `0.001005` | `0.1246` |

---

## 5. 核心结论与关键诊断发现

1. **工程闭环完全打通**：双卡 `vllm_mode="server"` 架构、NCCL 权重同步、LoRA 梯度回传与离线内核全部跑通；保存的 `final_adapter/adapter_model.safetensors`（65 MB，372 个张量）经核验 **`186/186` 个 `lora_B` 矩阵全部非零**（`max_abs=0.000155`），训练确有真实参数更新。
2. **策略与格式高度稳定，无模型坍塌/漏标签**：训练集 18 个日志步中 `reward/format_mean` 恒为 **`1.000`**、`reward/leak_rate` 恒为 **`0.000`**、`kl` 保持极低水平（`0.0000–0.0013`），验证集截断率与 leak 率仅 `0.0167`（60 组中仅极个别样本）。
3. **最重要科学发现：零方差组占比过高（`frac_reward_zero_std` 均值 `86.46%`）**：
   - 在 `G=4`、`temperature=0.7` 下，经过 SFT-v3（`ckpt-68`）训练的 2B 模型面对当前 Smoke 数据分布已经高度确定性：`entropy` 仅 `0.03–0.26`（均值 `0.116`），平均每步约 `86.5%` 的 prompt 组内 4 个 rollout 得分完全相同（或全部满分 `+4.0`，或全部陷入同一种错法）。
   - 在原生 GRPO 中，同组方差为 0 的样本对策略梯度贡献为 0，直接导致多个日志步（如 Step 6、10、22、32、34）的 `grad_norm` 跌至 `0.0003–0.0008`，有效样本利用率仅约 `13.5%`。
4. **验证集得分未现显著泛化提升（Smoke 预期内现象）**：
   - 训练集 `reward` 从 Step 2 的 `3.013` 升至后段 `3.55–3.97`（其中 `non_call` 多次触及 `3.000` 满分），但验证集 `eval_reward` 在 Step 19（`3.2636`）到 Step 37（`3.2056`）之间基本持平（`-0.058`，在随机采样波动范围内）；且因未配置 Step 0 基线验证，无法从本表直接量化相比 `ckpt-68` 的初始增益。

---

## 6. 面向下一阶段（`grpo_v1` / 架构优化）的讨论与建议

在扩大到全量 `grpo_v1`（1,500 条）之前，基于本轮实测数据建议重点评估以下 4 项调整：

1. **提高组内探索多样性以压低 `frac_reward_zero_std`（高优先级）**：
   - GPU 1 尚有 `23.6 GiB` 显存富余，可将 **`num_generations` 从 `4` 提升至 `8`**，同时将 **`temperature` 从 `0.7` 提高至 `0.9–1.0`**，在不爆显存的前提下把有效学习梯度组比例拉高。
2. **调优数据难度配比或引入动态过滤（DAPO-style / 难例聚焦）**：
   - 当前 `single_turn_call`（40%）在 `ckpt-68` 上大多直接全对零方差；后续可降低简单单轮调用权重或对零方差组做动态过采样/过滤，将算力集中到模型仍会犯错的 `multi_turn`、`parallel_call`、`missing_param` 与对抗拒答样本。
3. **调整 GPU 0 vLLM 显存水位与基线监控**：
   - 将 `run_grpo_dual.sh` 的 `gpu_memory_utilization` 从 `0.85` 适度调低至 **`0.78`**，避免 GPU 0 长期贴近 `39.78 / 40.96 GiB` 边缘；
   - 增加 **Step 0 eval**（或用 `merged-checkpoint-68` 与 `grpo-smoke` `final_adapter` 跑一次快速 benchmark 对比），建立严格的 `RL vs SFT` 增量基线。
4. **环境与日志告警收尾**：
   - 当前 GPU 0 上 vLLM 进程尚未启用 `--return-sampling-mask`（vLLM 0.30 参数提示），且 GPU 1 上仍有 `causal_conv1d_fn` 回退 PyTorch 参考实现的提示（当前 `10.9 s/step` 已较流畅，可按需决定是否编译 `causal_conv1d` wheel）。

---

## 7. 产出索引

- 远端模型目录（仅留服务器）：`/root/ToolCall-SFT/runs/train/grpo-smoke-2b-20261125T095728Z/`（含 `checkpoint-19`、`checkpoint-37`、`final_adapter`）
- 本地回拉日志与状态：
  - [train_summary.json](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/train/grpo-smoke-2b-20261125T095728Z/train_summary.json)
  - [trainer_state.json](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/train/grpo-smoke-2b-20261125T095728Z/checkpoint-37/trainer_state.json)
  - [resolved_config.yaml](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/train/grpo-smoke-2b-20261125T095728Z/resolved_config.yaml)
  - [vllm_serve.log](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/train/grpo-dual-20261125T095645Z/vllm_serve.log)
  - [grpo-smoke-2b-20261125T095645Z.log](file:///home/sikm/Project/RL-Learn/ToolCall-SFT/runs/logs/tmux/grpo-smoke-2b-20261125T095645Z.log)
