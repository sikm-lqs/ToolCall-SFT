# GRPO 单卡最小闭环验收

- 日期：2026-11-25（Asia/Shanghai；运行名时间戳为 UTC）
- 目的：验证冻结数据、原生生成、现有 reward、GRPO 更新、LoRA 保存闭环；不是能力评测。
- 节点：`rl-server`，单张 A100 40 GB。用 `run_tmux.sh` 启动两个独立快照。未修改正式训练配置、冻结数据及 reward。

## 环境与输入

- `runs/envs/rl` 已完成安装：PyTorch `2.13.0+cu129`、TRL `1.18.2`、Transformers `5.24.0`、PEFT `0.22.0`、vLLM `0.30.0`。核心导入、CUDA BF16 矩阵计算、GRPO 相关 13 项单测均通过。未启动 vLLM 服务。
- `uv pip check` 仍报项目通用依赖 `modelscope` 和 `matplotlib` 未安装；本实验使用已经在本地磁盘的合并模型、不绘图，训练路径不依赖这两项。原生生成的 `causal_conv1d` 使用参考实现，未为本轮另装加速包。
- 两套冻结数据的 train、val 和 audit ledger 通过 SHA-256 校验，加载得到 Smoke 300/60、V1 1500/150。2,010 个 prompt 都用实际训练起点目录中的 tokenizer 复核了预渲染与模板 token 等价及 token 数。
- 两份正式 YAML 用原生生成模式都能构造 `GRPOConfig`；按原样使用 vLLM server 模式则会触发采样参数校验，见下文。

## 单卡实跑结果

同一起点：合并后的 SFT-v3 r64 checkpoint-68，挂新 LoRA r16，`G=4`、`beta=0.02`、常数学习率 `1e-5`，无 weight decay。使用原生 HF 生成，没有启用 SwanLab 或验证集生成。

| 运行 | 训练步 / 采样数 | 训练计时（不含模型加载） | PyTorch 峰值已分配显存 | 更新证据 |
| --- | ---: | ---: | ---: | --- |
| `grpo-minimal-2b-20261125T054130Z` | 4 / 16 | 162.53 s | 5.346 GiB | 4/4 组同分，梯度均为零；186 个 LoRA B 张量均为零 |
| `grpo-minimal-signal-2b-20261125T054552Z` | 6 / 24 | 40.94 s | 5.049 GiB | 第一步 reward std = 0.80、grad norm = 0.3421；最终 186/186 个 LoRA B 张量非零，保存张量均有限 |

第一轮直接取冻结 Smoke 的前 4 条，跑完但没有学习信号。第二轮从同一冻结文件原样选取零基索引 `13,19,65,111,275,284`，使用更高采样温度 `1.15`、`top_p=0.98` 和 256 token 生成上限，诊断集、配置及带源哈希的清单在 `runs/diagnostics/grpo-minimal-signal-20261125T054120Z/`。第二轮 5/6 组仍为零方差；这是有意挑样的工程验证，不能用于估计正式训练集的零方差率或外推吞吐。

第二轮 adapter SHA-256：`51720ba3251962d87c573c2af6178cdd4e136e1fc3bc7441701c8fe44a43dc2c`。两轮日志和轻量统计已同步回本地，权重仅留在服务器。

## 双卡交接

1. 当前 `rl-server` 只可见一张 GPU；未验收 vLLM 服务启动、权重同步、重要性采样及双卡训练。
2. 两份正式 YAML 的 `top_p=0.95`、`importance_sampling_correction=true` 与缺失的 `top_k` 不兼容。若保留 nucleus sampling 和校正，需要让 trainer 透传并显式配置正数 `top_k`，再复验；本轮不改正式采样策略。
3. 用远端 TRL 1.18.2 的参数解析器做了无启动测试：双卡 runner 的 `--enable-prefix-caching` 需要显式值（如 `true`），且 `trl.scripts.vllm_serve` 不接受 `--language-model-only`。两个参数修正后 CLI 可解析，但不代表服务已通过验收。
