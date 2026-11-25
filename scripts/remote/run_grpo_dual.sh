#!/usr/bin/env bash
# Dual-GPU GRPO launcher:
#   GPU 0: runs/envs/rl (or runs/envs/rl-serve) running `trl vllm-serve`
#   GPU 1: runs/envs/train (or runs/envs/rl) running `toolcall.training.grpo`
# Must be launched via scripts/remote/run_tmux.sh inside an isolated snapshot.
set -euo pipefail

: "${TOOLCALL_SNAPSHOT_DIR:?launch via scripts/remote/run_tmux.sh}"

if [[ $# -lt 1 ]]; then
  echo "Usage: run_grpo_dual.sh <grpo_config_yaml> [extra grpo args...]" >&2
  exit 2
fi

readonly grpo_config="$1"
shift

readonly gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$gpu_count" -lt 2 ]]; then
  echo "Error: Dual-GPU GRPO requires >= 2 GPUs, found $gpu_count." >&2
  exit 1
fi

readonly serve_py="${RL_SERVE_PYTHON:-$([[ -x runs/envs/rl-serve/bin/python ]] && echo runs/envs/rl-serve/bin/python || echo runs/envs/rl/bin/python)}"
readonly train_py="${RL_TRAIN_PYTHON:-$([[ -x runs/envs/rl/bin/python ]] && echo runs/envs/rl/bin/python || echo runs/envs/train/bin/python)}"
test -x "$serve_py"
test -x "$train_py"

readonly stamp="$(basename "$TOOLCALL_SNAPSHOT_DIR" | grep -oE '[0-9]{8}T[0-9]{6}Z$')"
readonly model_path="$("$train_py" -c 'import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))["model"]["local_path"])' "$grpo_config")"
readonly vllm_port="$("$train_py" -c 'import yaml, sys; url=yaml.safe_load(open(sys.argv[1]))["vllm"]["server_base_url"]; print(url.rsplit(":", 1)[-1].split("/")[0])' "$grpo_config")"
readonly log_dir="runs/train/grpo-dual-${stamp}"
mkdir -p "$log_dir"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1

echo "==> Starting GPU 0 vLLM weight-transfer server on port $vllm_port using $serve_py"
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 "$serve_py" -m trl.scripts.vllm_serve \
  --model "$model_path" \
  --host 127.0.0.1 \
  --port "$vllm_port" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 6144 \
  --enable-prefix-caching true \
  --language-model-only \
  >"$log_dir/vllm_serve.log" 2>&1 &
readonly vllm_pid=$!

trap 'kill -TERM "$vllm_pid" 2>/dev/null || true; wait "$vllm_pid" || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Waiting for vLLM server readiness at http://127.0.0.1:${vllm_port}/health/ ..."
ready=false
for _ in {1..72}; do
  if curl --fail --silent --max-time 3 "http://127.0.0.1:${vllm_port}/health/" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    tail -50 "$log_dir/vllm_serve.log"
    exit 1
  fi
  sleep 5
done
if [[ "$ready" != true ]]; then
  tail -50 "$log_dir/vllm_serve.log"
  exit 1
fi

echo "==> Starting GPU 1 GRPOTrainer using $train_py"
CUDA_VISIBLE_DEVICES=1 "$train_py" -u -m toolcall.training.grpo --config "$grpo_config" "$@"
