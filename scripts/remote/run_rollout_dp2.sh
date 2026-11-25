#!/usr/bin/env bash
set -euo pipefail

# Launch two independent vLLM instances (GPU 0 :8000, GPU 1 :8001) and run DP=2 rollout difficulty mining.
# Usage:
#   bash scripts/remote/run_rollout_dp2.sh [candidates_jsonl] [temperature] [concurrency_per_gpu]

readonly candidates_jsonl="${1:-data/processed/grpo_rollout_pool/candidates.jsonl}"
readonly temperature="${2:-0.9}"
readonly concurrency="${3:-64}"

readonly model_path="runs/train/sft-v3-2b-lora-20261124T143953Z/merged-checkpoint-68"
readonly rl_py="${RL_PYTHON:-runs/envs/rl/bin/python}"
readonly vllm_bin="${RL_VLLM_BIN:-runs/envs/rl/bin/vllm}"
readonly stamp="$(date -u +%Y%m%dT%H%M%SZ)"
readonly out_dir="/root/ToolCall-SFT/runs/rollout/grpo-pool-dp2-$stamp"

mkdir -p "$out_dir"

export PATH="$(pwd)/runs/envs/rl/bin:$PATH"
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

pid0=""
pid1=""
cleanup() {
  for p in "$pid0" "$pid1"; do
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      echo "==> Stopping vLLM PID $p"
      kill "$p" 2>/dev/null || true
      wait "$p" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo "==> Output dir: $out_dir"
echo "==> Launching GPU 0 vLLM (:8000) and GPU 1 vLLM (:8001)"

CUDA_VISIBLE_DEVICES=0 FLASHINFER_WORKSPACE_BASE="/tmp/fi_gpu0" "$vllm_bin" serve "$model_path" \
  --served-model-name merged-checkpoint-68 \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 6144 \
  --enable-prefix-caching \
  --language-model-only \
  >"$out_dir/vllm_gpu0.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 FLASHINFER_WORKSPACE_BASE="/tmp/fi_gpu1" "$vllm_bin" serve "$model_path" \
  --served-model-name merged-checkpoint-68 \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 6144 \
  --enable-prefix-caching \
  --language-model-only \
  >"$out_dir/vllm_gpu1.log" 2>&1 &
pid1=$!

wait_health() {
  local port="$1"
  local pid="$2"
  local logf="$3"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "==> vLLM on port ${port} is healthy"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: vLLM on port ${port} exited prematurely" >&2
      tail -n 40 "$logf" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "ERROR: Timed out waiting for vLLM on port ${port}" >&2
  return 1
}

wait_health 8000 "$pid0" "$out_dir/vllm_gpu0.log"
wait_health 8001 "$pid1" "$out_dir/vllm_gpu1.log"

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "==> Running DP=2 async rollout & ToolRL+3.4-B scoring"
"$rl_py" -m toolcall.training.rollout_mine \
  --input "$candidates_jsonl" \
  --output "$out_dir/scored_rollouts.jsonl" \
  --summary "$out_dir/rollout_summary.json" \
  --endpoints "http://127.0.0.1:8000,http://127.0.0.1:8001" \
  --model merged-checkpoint-68 \
  --n 4 \
  --temperature "$temperature" \
  --top-p 0.95 \
  --top-k 512 \
  --max-tokens 512 \
  --concurrency-per-server "$concurrency"
