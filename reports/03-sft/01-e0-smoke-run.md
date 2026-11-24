# SFT-v1 E0：从基座重跑的实验记录

日期：2026-11-24（Asia/Shanghai）。Qwen3.5-2B 从微调前的底座重跑 E0，不复用历史 checkpoint。**训练、合并、推理、评测链路跑通；拒调退化，不作为 E1 直接放行的依据。**

## 运行与问题记录

| 时间（CST） | 阶段 | 观察 / 处理 |
| --- | --- | --- |
| 18:34 | 同步、单测、启动 | 同步本地 `415c323` 代码，远端训练 venv 9/9 通过；`run_tmux.sh` 启动 `sft-e0-2b-fresh` 独立快照。使用冻结的 3,000 条 smoke 与 500 条验证数据；未传入恢复 checkpoint。 |
| 18:34 | 预检 | 文件 SHA-256 校验通过，32 个抽查样本模板字节一致。3,000 行打包成 311 条序列；两个真实 batch 的有效标签占比 12.1% / 8.0%，分段边界检查通过，GPU packed forward loss 0.2476。8.0% 低于手册的经验范围，但在代码硬阈值内，且不存在训练目标泄漏、空标签或跨样本边界信号，不调整配置。 |
| 18:34 | 训练警告 | Hub kernel 对部分 RMSNorm 层回退原生实现；训练所需的 packed forward 已通过。Transformers 对 `gradient_checkpointing` 发出未来弃用提示，Trainer 将 EOS 对齐 tokenizer；均不改动本次基准配置。 |
| 训练期间 | 排查历史 vLLM 阻塞 | 旧日志：`Qwen3_5TextConfig` 进了期望 `Qwen3_5Config` 的多模态处理器；检查远端 vLLM 0.19.1，原生 registry 只注册 Qwen3.5 的 `Qwen3_5ForConditionalGeneration`，而非裸 `Qwen3_5ForCausalLM`。训练路径不变，部署导出恢复官方 wrapper 与权重命名空间，启动时使用与基线相同的 `--language-model-only`。 |
| 18:36–18:37 | 导出回归测试 | 微型 Qwen3.5 底座＋非零 LoRA 在旧逻辑下稳定导出错误架构（RED）；修复后重新加载通过，文本投影权重精确等于底座权重加 LoRA 增量，视觉权重不变，原生模板/处理器存在。远端 10/10 通过。 |
| 18:38 / 18:42 | 训练与验证 | 约半轮时 `eval_loss=0.1971`，1 轮结束 `0.1930`；SwanLab 训练 ID `oi26ur9w`。39 步完成，`train_loss=0.3707`，验证 token 准确率 94.70%，训练阶段 499.11 s，PyTorch 峰值显存 7.197 GiB。 |
| 18:43 | 合并与部署 | 训练快照保留原始纯文本 `merged-best/`。修复后的独立快照从**此次**训练的 `checkpoint-39` 另导出 `merged-vllm/`，不覆盖旧产物、不重训。vLLM 解析到 `Qwen3_5ForConditionalGeneration`，按 64K、text-only 启动并成功加载权重。 |
| 18:45 | 服务就绪与接口探针 | 冷启动约 2 分 15 秒（其中 torch.compile 26.8 秒、首次 profiling/warmup 80.6 秒），不当作挂起；等待 `/v1/models` HTTP 200 且声明 65,536 上下文后再发请求。`auto` 返回结构化 `lookup_temperature({"city": "Paris"})`，注入虚构工具结果后，`none` 正常返回 21°C 文本，两次均无工具 XML 泄漏。 |
| 18:45 | BFCL 快筛启动 | 固定 8 类 1,840 题，使用与 Baseline 相同的生成设置：关思考、temperature 0、1024 输出上限、300 秒超时、128 并发。 |
| 18:51 | 评测完成、回收轻量产物 | 1,840 题，10,167 次请求全部成功，无超上下文，3 次命中 1024 输出上限；评测耗时约 326 s。SwanLab 评测 ID `2e4wuajt`。服务随本次 tmux 任务退出，GPU 恢复空闲。 |

## BFCL 8 类对照

Baseline 取自 [2B 正式 64K 报告](../01-baseline/01-baseline-report.md) 对应运行；逐类核对，两次的题目 ID 集合完全一致。Baseline 是在 17 类 `core` 运行中抽取的 8 类，不是单独重跑的 8 类服务负载对照，因此以下是同题诊断，不是 `core` 加权分（43.48%）的对标。

| 类别 | 2B Baseline | 2B E0 | 差值（百分点） |
| --- | ---: | ---: | ---: |
| `simple_python` | 70.50% (282/400) | 92.50% (370/400) | +22.00 |
| `multiple` | 82.50% (165/200) | 94.50% (189/200) | +12.00 |
| `parallel` | 50.50% (101/200) | 89.00% (178/200) | +38.50 |
| `parallel_multiple` | 71.00% (142/200) | 88.00% (176/200) | +17.00 |
| `irrelevance` | 78.75% (189/240) | 25.00% (60/240) | **−53.75** |
| `multi_turn_base` | 29.50% (59/200) | 23.00% (46/200) | −6.50 |
| `multi_turn_miss_func` | 2.50% (5/200) | 7.50% (15/200) | +5.00 |
| `multi_turn_miss_param` | 15.00% (30/200) | 12.50% (25/200) | −2.50 |
| **8 类题级微平均** | **52.88% (973/1,840)** | **57.55% (1,059/1,840)** | **+4.67** |

**主要行为问题：不该调用时却输出了可解析的调用。** 240 道 `irrelevance` 中，131 道从通过变失败，仅 2 道反向变化；E0 的 180 道失败全部为 `irrelevance_error:decoder_success`。该类无请求失败，仅 1 次输出截断，不能解释 129 道净退化。工具链启动问题已解决，但**拒调退化尚未解决**。单次 smoke 不能归因到某个超参或数据源；E0 含 350/3,000 条不调用/追问样本，仍需区分「有候选但都不匹配」「缺必要参数」「收到工具结果后应收尾」的有效覆盖。正式 27k 的不调用/追问占比是 1,498/27,058（约 5.5%），不能假定扩大样本后会自动修好。

**建议下一步**：先按上述三种场景审计负例与典型误调，不把 BFCL 原题混进训练；保持模型、LoRA、步数和评测协议固定，另做一组调整负例覆盖的 3k 对照训练，可对比半轮和整轮 checkpoint。用 `irrelevance` 设放行线，同时保留 `parallel` 增益，再决定 E1；本次未启动 E1/E2。

## 复现索引

- 底座：`Qwen/Qwen3.5-2B`（项目已缓存快照，非 `-Base` 变体）；训练配置：`configs/train/e0_smoke_2b.yaml`；训练环境：Transformers 5.24.0 / TRL 1.18.2 / PyTorch 2.10.0+cu129；服务环境：vLLM 0.19.1。
- 训练：`runs/train/e0-smoke-2b-20261124T103407Z`，SwanLab `oi26ur9w`；部署采用同目录下的 `merged-vllm/`，adapter SHA-256 记录在评测 `config.json`。
- 快筛：`runs/eval/bfcl-v4/sft-v1-e0-smoke-2b-fresh/custom-e0-smoke-20261124T104541.570211Z`，SwanLab `2e4wuajt`。
- 服务日志/探针：`runs/services/sft-v1-e0-smoke-2b-fresh/20261124T103708.465182680Z`；训练与部署 tmux 日志分别为 `sft-e0-2b-fresh-20261124T103403Z.log`、`vllm-eval-e0-2b-fresh-20261124T103712Z.log`，位于 `runs/logs/tmux/`。本地仅回收摘要、日志、探针输出和图表，权重保留在远端。
- 上游参照：[vLLM 0.19.1 模型注册表](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/registry.py)、[Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)。
