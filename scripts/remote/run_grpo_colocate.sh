#!/usr/bin/env bash
set -euo pipefail

# Launch dual-GPU DDP + vLLM colocate GRPO training on rl-server inside a single tmux session.
# Usage:
#   bash scripts/remote/run_grpo_colocate.sh <config_yaml> [nproc_per_node] [master_port]

readonly cfg_path="${1:-configs/train/grpo_v2_smoke_2b_colocate.yaml}"
readonly nproc="${2:-2}"
readonly master_port="${3:-29518}"

readonly rl_py="${RL_TRAIN_PYTHON:-runs/envs/rl/bin/python}"
readonly torchrun_bin="${RL_TORCHRUN_BIN:-runs/envs/rl/bin/torchrun}"
readonly stamp="$(date -u +%Y%m%dT%H%M%SZ)"
readonly exp_name="$("$rl_py" -c "import yaml; print(yaml.safe_load(open('$cfg_path'))['experiment'])")"
readonly out_dir="/root/ToolCall-SFT/runs/train/${exp_name}-${stamp}"

mkdir -p "$out_dir"

export PATH="$(pwd)/runs/envs/rl/bin:$PATH"
export RUN_STAMP="$stamp"
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "==> Starting DDP+vLLM colocate GRPO ($exp_name, nproc=$nproc, port=$master_port)"
echo "==> Output dir: $out_dir"

CUDA_VISIBLE_DEVICES=0,1 "$torchrun_bin" \
  --nproc_per_node="$nproc" \
  --master_port="$master_port" \
  -m toolcall.training.grpo \
  --config "$cfg_path" \
  --output-dir "$out_dir"
