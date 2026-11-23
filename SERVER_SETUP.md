# ToolCall-SFT 服务器安装指南

适用环境为 Ubuntu 22.04 x86_64、单张 Tesla V100 32 GB、NVIDIA 驱动 550.127.05。项目要求 Python 3.11 或更高，下面统一使用 3.11。uv 可以在用户目录安装 Python 和管理虚拟环境，系统 Python 3.10 保留给 Ubuntu。

| 用途 | 组件 |
| --- | --- |
| 下载基座权重、运行评测入口、记录结果 | 项目的 `pyproject.toml` 声明 ModelScope、PyYAML、SwanLab、Matplotlib 和 nvidia-ml-py |
| 单卡 SFT | PyTorch 2.6.0 CUDA 12.4 wheel、Transformers 5.5.4、TRL 0.29.1 和 PEFT 0.18.1。TRL 会带入 Accelerate 和 Datasets |
| 评测执行 | `toolcall-eval prepare` 为 BFCL V4、API-Bank 和 τ²-Bench 分别准备 worker 环境和评测资产 |
| 评测推理 | 可访问的 OpenAI 兼容端点，支持 `/v1/models`、`/v1/chat/completions` 和原生工具调用 |

当前仓库已经有下载和评测命令，训练与数据处理目录仍是占位。安装 TRL 可为后续实现 SFT 提前验证 CUDA 和依赖。建议先用 0.8B 做文本 LoRA，单卡用 FP16 混合精度，设置 `fp16=True`、`bf16=False`。训练时可先用 micro-batch 1、约 1,024 tokens、梯度累积和梯度检查点。LoRA 训练保留可训练参数的 FP32 精度，再由 AMP 执行混合精度。vLLM、DeepSpeed、bitsandbytes、FlashAttention 2 和编译加速可以等到有明确需求时再评估。 [TRL SFTTrainer](https://huggingface.co/docs/trl/v0.29.1/en/sft_trainer) · [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)

## 1. 安装 uv 并克隆项目

先确认 `git`、`curl` 可用。缺失时由有权限的账号执行 `sudo apt install git curl ca-certificates`。

```bash
uname -m
nvidia-smi

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.11

git clone https://github.com/sikm-lqs/ToolCall-SFT.git
cd ToolCall-SFT
git rev-parse HEAD

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
python --version
toolcall-download --help
toolcall-eval --help
```

私有仓库可用已有 GitHub SSH 或 HTTPS 凭据。uv 自带包安装能力，以上步骤无需 conda 或系统 pip。评测 worker 内的 pip 由项目运行 `venv` 时自动引导安装。首次安装需要访问 Astral 的 uv/Python 下载地址、GitHub、PyPI 和项目使用的模型仓库。 [uv 安装](https://docs.astral.sh/uv/getting-started/installation/) · [uv 管理 Python](https://docs.astral.sh/uv/guides/install-python/)

## 2. 下载模型

在仓库根目录执行，默认复用 ModelScope 缓存。可加 `--cache-dir /absolute/path/to/cache` 将权重放到大磁盘。命令输出每个模型的本地快照目录，后续直接使用该路径。

```bash
.venv/bin/toolcall-download configs/models/qwen35_0_8b.yaml

# 需要 2B 基线时再执行
.venv/bin/toolcall-download configs/models/qwen35_2b.yaml
```

ModelScope 负责权重和分词器等模型文件，uv 负责 Python 和 Python 包。评测依赖的向量模型和部分数据由仓库现有准备命令从 Hugging Face、GitHub 等来源获取。模型 YAML 当前使用可变的 `master`，正式实验需记录下载的 revision（若可得）、快照路径和权重 SHA-256。 [ModelScope 下载 API](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py)

## 3. 安装单卡训练依赖

从项目根目录执行。训练环境单独放在 Git 忽略的 `runs/envs/train/`，便于独立更新评测和训练依赖。分两次安装以便 PyTorch 使用官方 CUDA 源，其余库走 PyPI。指定含 `+cu124` 的版本可约束第二次解析继续使用同一 CUDA wheel。

```bash
uv venv --python 3.11 runs/envs/train
TRAIN_PY="$PWD/runs/envs/train/bin/python"

uv pip install --python "$TRAIN_PY" \
  --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124'

uv pip install --python "$TRAIN_PY" -e . \
  'torch==2.6.0+cu124' 'transformers==5.5.4' 'trl==0.29.1' 'peft==0.18.1'

uv pip check --python "$TRAIN_PY"
uv pip freeze --python "$TRAIN_PY" > runs/envs/train-packages.txt
```

标准 wheel 自带所需 CUDA 用户态运行库。此安装流程和下面的 CUDA 测试无需 `nvcc`，也无需更改驱动或 CUDA Toolkit。以后确实要编译 CUDA 扩展时，可临时设置 `CUDA_HOME=/usr/local/cuda-12.4`，并将 `$CUDA_HOME/bin` 加入 `PATH`。 [PyTorch CUDA 12.4 官方安装](https://pytorch.org/get-started/previous-versions/#v260)

在服务器执行下面的 CUDA 运算和训练栈导入测试。

```bash
"$TRAIN_PY" - <<'PY'
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.version.cuda == "12.4", torch.version.cuda
assert torch.cuda.get_device_capability(0) == (7, 0)
assert "sm_70" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
x = torch.ones((256, 256), device="cuda", dtype=torch.float16)
assert (x @ x)[0, 0].item() == 256
print(torch.__version__, torch.cuda.get_device_name(0), "FP16 matmul OK")
PY
```

## 4. 准备评测并检查推理端点

vLLM 是可选的推理后端。当前 vLLM 官方 GPU 文档要求计算能力 7.5 起，V100 为 7.0。优先用已有的兼容端点，或在另一台支持当前 vLLM 的机器上部署。若只有这台 V100，可以先用 Transformers 直接验证小模型的生成，后续试跑支持 Qwen3.5 的 llama.cpp CUDA 服务。llama.cpp 使用 GGUF，需单独确认所用权重、工具模板、工具调用解析以及上下文容量。正式比较时记录推理引擎、精度和模板，尤其是采用了 GGUF 量化的实验。 [vLLM GPU 要求](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) · [NVIDIA GPU 计算能力](https://developer.nvidia.com/cuda/gpus/legacy) · [llama.cpp 服务接口](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#openai-compatible-chat-completions-api)

先准备一个评测，后续可以用 `prepare all`。准备阶段可以独立于推理服务运行。

```bash
.venv/bin/toolcall-eval prepare bfcl-v4
```

端点启动后，用 `/v1/models` 查找模型名，再发一个工具调用请求。下面以 0.8B 为例，`MODEL` 需与服务列出的 ID 一致。需要认证时先在本机环境中设置 `TOOLCALL_AGENT_API_KEY`。

```bash
BASE_URL=http://127.0.0.1:18003/v1
MODEL=Qwen/Qwen3.5-0.8B

curl -fsS -H "Authorization: Bearer ${TOOLCALL_AGENT_API_KEY:-local}" \
  "$BASE_URL/models"

curl -fsS "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer ${TOOLCALL_AGENT_API_KEY:-local}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Call get_weather for Paris.\"}],
    \"tools\": [{\"type\": \"function\", \"function\": {
      \"name\": \"get_weather\", \"description\": \"Get weather for a city\",
      \"parameters\": {\"type\": \"object\", \"properties\": {
        \"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}
    }}],
    \"chat_template_kwargs\": {\"enable_thinking\": false},
    \"temperature\": 0,
    \"max_tokens\": 256
  }"
```

检查返回的 `choices[0].message.tool_calls` 是否包含 `get_weather` 和合法的 JSON 参数。然后运行 BFCL smoke。其他评测及其上下文需求见 [评测说明](configs/eval/README.md)。

```bash
.venv/bin/swanlab login
.venv/bin/toolcall-eval bfcl-v4 --base-url "$BASE_URL" --model "$MODEL" --concurrency 1
```

评测结果会上传至 SwanLab 云端 `toolcall-sft` 项目。若项目位于团队空间，请在本地和服务器上设置同一个 `SWANLAB_WORKSPACE`。 [SwanLab 环境变量](https://docs.swanlab.cn/en/api/environment-variable.html)

后续更新从 GitHub 拉取，详见 [本地开发与服务器实验](README.md#本地开发与服务器实验)。依赖声明变动时重新执行对应环境的安装命令。训练产物和评测日志保存在 `runs/` 或实验指定的目录。
