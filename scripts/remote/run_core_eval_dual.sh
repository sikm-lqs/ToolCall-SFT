#!/usr/bin/env bash
# Merge the top-2 checkpoints from a GRPO/SFT run and evaluate them concurrently
# on GPU 0 (:18003) and GPU 1 (:18004) with BFCL Core (4,441) + API-Bank Full (1,281).
# Must be launched via scripts/remote/run_tmux.sh inside an isolated snapshot.
set -euo pipefail

: "${TOOLCALL_SNAPSHOT_DIR:?launch via scripts/remote/run_tmux.sh}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ $# -ne 4 ]]; then
  echo "Usage: run_core_eval_dual.sh <train_run_dir> <exp_prefix> <ckpt_gpu0> <ckpt_gpu1>" >&2
  echo "  e.g. run_core_eval_dual.sh runs/train/grpo-v2-... grpo-v2 checkpoint-141 checkpoint-187" >&2
  exit 2
fi

readonly train_run="$1"
readonly exp_prefix="$2"
readonly ckpt0="$3"
readonly ckpt1="$4"

readonly stamp="$(basename "$TOOLCALL_SNAPSHOT_DIR" | grep -oE '[0-9]{8}T[0-9]{6}Z$')"
readonly run_meta="runs/eval/${exp_prefix}-core-dual-${stamp}"
mkdir -p "$run_meta"

printf 'train_run=%s\nexp_prefix=%s\nckpt_gpu0=%s\nckpt_gpu1=%s\n' \
  "$train_run" "$exp_prefix" "$ckpt0" "$ckpt1" | tee "$run_meta/run_paths.txt"

merge_one() {
  local ckpt_name="$1"
  local ckpt_dir="$train_run/$ckpt_name"
  local merged_dir="$train_run/merged-$ckpt_name"
  test -s "$ckpt_dir/adapter_model.safetensors"
  if [[ ! -f "$merged_dir/merge_info.json" ]]; then
    echo "==> Merging $ckpt_dir -> $merged_dir"
    runs/envs/train/bin/python -u -m toolcall.training.merge_lora \
      --adapter-dir "$ckpt_dir" --output-dir "$merged_dir"
  fi
}

merge_one "$ckpt0"
merge_one "$ckpt1"

run_eval_on_gpu() {
  local gpu_idx="$1"
  local port="$2"
  local ckpt_name="$3"

  local base_url="http://127.0.0.1:${port}/v1"
  local ckpt_dir="$train_run/$ckpt_name"
  local merged="$train_run/merged-$ckpt_name"
  local digest
  digest="$(sha256sum "$ckpt_dir/adapter_model.safetensors" | cut -d' ' -f1)"
  local model="${exp_prefix}-${ckpt_name}"
  local phase="core-${ckpt_name}"
  local eval_dir="$run_meta/$phase"
  mkdir -p "$eval_dir"

  if curl --silent --max-time 2 "$base_url/models" >/dev/null; then
    echo "Refusing to replace running service at $base_url" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu_idx" runs/envs/serve/bin/vllm serve "$merged" --served-model-name "$model" \
    --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --dtype bfloat16 \
    --language-model-only --max-model-len 65536 --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 --max-num-batched-tokens 32768 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --generation-config vllm --seed 0 --enable-prefix-caching \
    >"$eval_dir/vllm.log" 2>&1 &
  local service_pid=$!
  trap 'kill -TERM "$service_pid" 2>/dev/null || true; wait "$service_pid" || true' EXIT

  echo "[GPU $gpu_idx] Waiting for vLLM readiness ($model) on port $port"
  local ready=false
  for _ in {1..72}; do
    if curl --fail --silent --max-time 3 "$base_url/models" >"$eval_dir/models.json"; then
      ready=true
      break
    fi
    if ! kill -0 "$service_pid" 2>/dev/null; then
      tail -n 40 "$eval_dir/vllm.log" >&2 || true
      return 1
    fi
    sleep 5
  done
  if [[ "$ready" != true ]]; then
    tail -n 40 "$eval_dir/vllm.log" >&2 || true
    return 1
  fi

  .venv/bin/python - "$base_url" "$model" "$eval_dir" <<'PY'
import json, sys
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

base, model, output = sys.argv[1:]
out = Path(output)
models = json.loads((out / "models.json").read_text())["data"]
assert any(m["id"] == model and m["max_model_len"] == 65536 for m in models), models
tools = [{
    "type": "function",
    "function": {
        "name": "lookup_temperature",
        "description": "Get the current temperature for one city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
messages = [{"role": "user", "content": "Use the available tool to get the current temperature in Paris."}]
opener = build_opener(ProxyHandler({}))

def probe(name):
    payload = dict(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
        max_completion_tokens=1024,
        chat_template_kwargs={"enable_thinking": False},
    )
    req = Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=60) as response:
        data = json.load(response)
    (out / f"probe_{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    return data["choices"][0]

first = probe("tool_call")
msg = first["message"]
calls = msg.get("tool_calls") or []
assert len(calls) == 1 and first["finish_reason"] == "tool_calls", first
assert calls[0]["function"]["name"] == "lookup_temperature", calls[0]
assert json.loads(calls[0]["function"]["arguments"])["city"].lower() == "paris", calls[0]
assert "<tool_call>" not in (msg.get("content") or ""), first

messages.extend([
    {"role": "assistant", "content": msg.get("content"), "tool_calls": calls},
    {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"temperature_c":21}'},
])
final = probe("tool_response")
msg = final["message"]
assert final["finish_reason"] == "stop" and not msg.get("tool_calls"), final
assert "21" in (msg.get("content") or "") and "<tool_call>" not in msg["content"], final
print(f"Structured probes passed on {model}")
PY

  local -a eval_args=(
    --model "$model"
    --base-url "$base_url"
    --timeout 300
    --gpu-index "$gpu_idx"
    --max-tokens 1024
    --temperature 0
    --checkpoint-revision "sha256:$digest"
  )
  local variant="${exp_prefix}-${phase}"

  echo "[GPU $gpu_idx] Running BFCL Core (4,441) for $ckpt_name"
  .venv/bin/python -u -m toolcall.evaluation.cli bfcl-v4 "${eval_args[@]}" \
    --profile core --concurrency 128 --variant "$variant" --output "$eval_dir/bfcl"

  .venv/bin/python - "$eval_dir" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
summary = json.loads((out / "bfcl/summary.json").read_text())
metrics, groups = summary["metrics"], summary["groups"]
assert metrics["cases"] == 4441, metrics
limits = {"irrelevance": .76, "live_irrelevance": .66, "parallel": .85, "simple_python": .88}
gate = {
    name: {
        "accuracy": groups[name]["accuracy"],
        "minimum": target,
        "passed": groups[name]["accuracy"] >= target,
    }
    for name, target in limits.items()
}
gate["all_passed"] = all(row["passed"] for row in gate.values())
(out / "behavior_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
PY

  echo "[GPU $gpu_idx] Running API-Bank Full (1,281) for $ckpt_name"
  .venv/bin/python -u -m toolcall.evaluation.cli api-bank "${eval_args[@]}" \
    --profile full --concurrency 96 --variant "$variant" --output "$eval_dir/api-bank"

  .venv/bin/python - "$eval_dir/api-bank/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    metrics = json.load(f)["metrics"]
assert (metrics["samples"], metrics["call_samples"], metrics["response_samples"]) == (1281, 753, 528), metrics
PY
  echo "[GPU $gpu_idx] Finished BFCL Core + API-Bank Full for $ckpt_name"
}

run_eval_on_gpu 0 18003 "$ckpt0" >"$run_meta/gpu0_${ckpt0}.log" 2>&1 &
pid_gpu0=$!
run_eval_on_gpu 1 18004 "$ckpt1" >"$run_meta/gpu1_${ckpt1}.log" 2>&1 &
pid_gpu1=$!

wait "$pid_gpu0"
wait "$pid_gpu1"

date -u +%FT%TZ > "$run_meta/completed_utc.txt"
echo "Dual-GPU core + API-Bank evaluation completed for $ckpt0 (GPU 0) and $ckpt1 (GPU 1) -> $run_meta"
