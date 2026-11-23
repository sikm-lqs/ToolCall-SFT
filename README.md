# ToolCall-SFT

以 Qwen3.5-0.8B 和 Qwen3.5-2B 为起点，研究小模型通用工具调用的后训练。计划比较 SFT、OPD 和 Agentic RL，使用 BFCL V4 评估通用调用能力，用 τ²-Bench 评估多轮任务。训练框架待定。

## 环境与模型

```bash
conda activate agentic-rl
python -m pip install -e .
toolcall-download configs/models/qwen35_0_8b.yaml configs/models/qwen35_2b.yaml
```

下载入口读取 ModelScope 模型 ID 和 revision，打印本地快照目录。默认共享 ModelScope 缓存，也可通过 `--cache-dir` 指定目录。正式实验记录实际使用的快照、权重哈希、推理配置和评测版本。

```text
configs/models/             两个起始模型的下载配置
configs/eval/               benchmark 版本与 smoke 范围
src/toolcall/data/          数据处理模块
src/toolcall/training/      后训练入口
src/toolcall/evaluation/    三套评测入口、指标与图表
data/                       本地数据
runs/eval/                  统一评测产物
runs/swanlab/               本地实验面板
```

## 评测

在 `agentic-rl` 环境安装项目后，用同一入口准备和运行三个评测。先开启 OpenAI 兼容模型服务，默认地址为 `http://127.0.0.1:18003/v1`。

```bash
toolcall-eval prepare all
toolcall-eval bfcl-v4
toolcall-eval api-bank
toolcall-eval tau2
```

默认是 smoke。完整评测、固定子集、用户模拟器和 SSH 部署参数见 [评测说明](configs/eval/README.md)。

## SwanLab

```bash
swanlab watch runs/swanlab
```

统一入口自动保存配置、分项指标、原始预测或轨迹、图表，并记录到本地 SwanLab。需要云端同步时传 `--swanlab online`。用 `toolcall-eval report runs/eval/<run_id>` 可以重绘已保存的运行。权重、数据、日志和本地凭据由 Git 忽略。
