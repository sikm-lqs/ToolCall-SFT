#!/usr/bin/env bash
# Merge one checkpoint and evaluate BFCL Core (4,441) + API-Bank Full (1,281) at 2x throughput
# using two independent single-GPU vLLM servers (GPU 0 :18003, GPU 1 :18004) sharded by category,
# then aggregate into one official BFCL Core summary.json and SwanLab run.
# Must be launched via scripts/remote/run_tmux.sh inside an isolated snapshot.
set -euo pipefail

: "${TOOLCALL_SNAPSHOT_DIR:?launch via scripts/remote/run_tmux.sh}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ $# -ne 3 ]]; then
  echo "Usage: run_core_eval_dp2.sh <train_run_dir> <exp_prefix> <ckpt_name>" >&2
  echo "  e.g. run_core_eval_dp2.sh runs/train/grpo-v2-... grpo-v2 checkpoint-47" >&2
  exit 2
fi

readonly train_run="$1"
readonly exp_prefix="$2"
readonly ckpt_name="$3"

readonly stamp="$(basename "$TOOLCALL_SNAPSHOT_DIR" | grep -oE '[0-9]{8}T[0-9]{6}Z$')"
readonly eval_dir="runs/eval/${exp_prefix}-${ckpt_name}-sharded2-${stamp}"
mkdir -p "$eval_dir"

readonly ckpt_dir="$train_run/$ckpt_name"
readonly merged="$train_run/merged-$ckpt_name"
test -s "$ckpt_dir/adapter_model.safetensors"

if [[ ! -f "$merged/merge_info.json" ]]; then
  echo "==> Merging $ckpt_dir -> $merged"
  runs/envs/train/bin/python -u -m toolcall.training.merge_lora \
    --adapter-dir "$ckpt_dir" --output-dir "$merged"
fi

readonly digest="$(sha256sum "$ckpt_dir/adapter_model.safetensors" | cut -d' ' -f1)"
readonly model="${exp_prefix}-${ckpt_name}"
readonly variant="${exp_prefix}-core-${ckpt_name}"

printf 'train_run=%s\nexp_prefix=%s\nckpt_name=%s\nmerged=%s\ndigest=%s\n' \
  "$train_run" "$exp_prefix" "$ckpt_name" "$merged" "$digest" | tee "$eval_dir/run_paths.txt"

service_pids=()
cleanup() {
  for pid in "${service_pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${service_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

start_vllm() {
  local gpu_idx="$1"
  local port="$2"
  local log_path="$eval_dir/vllm_gpu${gpu_idx}.log"
  local base_url="http://127.0.0.1:${port}/v1"

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
    >"$log_path" 2>&1 &
  service_pids+=("$!")
}

wait_and_probe() {
  local gpu_idx="$1"
  local port="$2"
  local pid="$3"
  local base_url="http://127.0.0.1:${port}/v1"
  local log_path="$eval_dir/vllm_gpu${gpu_idx}.log"

  echo "[GPU $gpu_idx] Waiting for vLLM readiness ($model) on port $port"
  local ready=false
  for _ in {1..72}; do
    if curl --fail --silent --max-time 3 "$base_url/models" >"$eval_dir/models_gpu${gpu_idx}.json"; then
      ready=true
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -n 40 "$log_path" >&2 || true
      return 1
    fi
    sleep 5
  done
  if [[ "$ready" != true ]]; then
    tail -n 40 "$log_path" >&2 || true
    return 1
  fi

  .venv/bin/python - "$base_url" "$model" "$eval_dir" "$gpu_idx" <<'PY'
import json, sys
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

base, model, output, gpu_idx = sys.argv[1:]
out = Path(output)
models = json.loads((out / f"models_gpu{gpu_idx}.json").read_text())["data"]
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
    (out / f"probe_gpu{gpu_idx}_{name}.json").write_text(json.dumps(data, indent=2) + "\n")
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
print(f"[GPU {gpu_idx}] Structured probes passed ({model})")
PY
}

start_vllm 0 18003
pid0="${service_pids[0]}"
start_vllm 1 18004
pid1="${service_pids[1]}"

wait_and_probe 0 18003 "$pid0"
wait_and_probe 1 18004 "$pid1"

wall_start=$(date +%s)

# Balanced sharding across both GPUs:
# GPU 0 (:18003): Non-Live 7 cats (1,390) + 2 heavy Multi-Turn (400) + API-Bank Full (1,281)
# GPU 1 (:18004): Live 6 cats (2,251) + 2 Multi-Turn (400)
run_shard0() {
  .venv/bin/python -u -m toolcall.evaluation.cli bfcl-v4 \
    --model "$model" --base-url "http://127.0.0.1:18003/v1" \
    --timeout 300 --gpu-index 0 --max-tokens 1024 --temperature 0 \
    --checkpoint-revision "sha256:$digest" \
    --categories simple_python simple_java simple_javascript multiple parallel parallel_multiple irrelevance multi_turn_base multi_turn_long_context \
    --concurrency 128 --variant "${variant}-s0" --output "$eval_dir/bfcl_shard0"

  .venv/bin/python -u -m toolcall.evaluation.cli api-bank \
    --model "$model" --base-url "http://127.0.0.1:18003/v1" \
    --timeout 300 --gpu-index 0 --max-tokens 1024 --temperature 0 \
    --checkpoint-revision "sha256:$digest" \
    --profile full --concurrency 96 --variant "$variant" --output "$eval_dir/api-bank"
}

run_shard1() {
  .venv/bin/python -u -m toolcall.evaluation.cli bfcl-v4 \
    --model "$model" --base-url "http://127.0.0.1:18004/v1" \
    --timeout 300 --gpu-index 1 --max-tokens 1024 --temperature 0 \
    --checkpoint-revision "sha256:$digest" \
    --categories live_simple live_multiple live_parallel live_parallel_multiple live_irrelevance live_relevance multi_turn_miss_func multi_turn_miss_param \
    --concurrency 128 --variant "${variant}-s1" --output "$eval_dir/bfcl_shard1"
}

run_shard0 >"$eval_dir/shard0.log" 2>&1 &
s0_pid=$!
run_shard1 >"$eval_dir/shard1.log" 2>&1 &
s1_pid=$!

wait "$s0_pid"
wait "$s1_pid"

echo "==> Merging BFCL Core shards into unified official summary ($eval_dir/bfcl)"
.venv/bin/python - "$eval_dir" "$model" "$variant" <<'PY'
import os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from toolcall.evaluation.cli import experiment_name
from toolcall.evaluation.common import read_json, write_json

eval_dir = Path(sys.argv[1]).resolve()
model, variant = sys.argv[2], sys.argv[3]
shard0, shard1 = eval_dir / "bfcl_shard0", eval_dir / "bfcl_shard1"
merged_bfcl = eval_dir / "bfcl"
merged_bfcl.mkdir(parents=True, exist_ok=True)

shutil.copytree(shard0 / "result", merged_bfcl / "result", dirs_exist_ok=True)
shutil.copytree(shard1 / "result", merged_bfcl / "result", dirs_exist_ok=True)

with (merged_bfcl / "model-requests.jsonl").open("w", encoding="utf-8") as out:
    for s in (shard0, shard1):
        out.write((s / "model-requests.jsonl").read_text(encoding="utf-8"))

with (merged_bfcl / "resources.jsonl").open("w", encoding="utf-8") as out:
    for s in (shard0, shard1):
        out.write((s / "resources.jsonl").read_text(encoding="utf-8"))

cfg0 = read_json(shard0 / "config.json")
cfg1 = read_json(shard1 / "config.json")

cats = [*cfg0["resolved_categories"], *cfg1["resolved_categories"]]
cfg = dict(cfg0)
cfg.pop("categories", None)
cfg["profile"] = "core"
cfg["variant"] = variant
cfg["resolved_categories"] = cats
cfg["run_name"] = experiment_name(cfg, datetime.now(timezone.utc))
cfg["run_status"] = "completed"
write_json(merged_bfcl / "config.json", cfg)

worker = Path("runs/tools/bfcl/bin/python").absolute()
env = dict(
    os.environ,
    BFCL_PROJECT_ROOT=str(merged_bfcl),
    OPENAI_BASE_URL=cfg["base_url"],
    OPENAI_API_KEY="local",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    PYTHONPATH=str(Path("src").resolve()),
)
t0 = time.monotonic()
subprocess.run(
    [str(worker), "-m", "toolcall.evaluation.bfcl", str(merged_bfcl), "evaluate", "--model", model, "--test-category", ",".join(cats)],
    env=env,
    check=True,
)
write_json(merged_bfcl / "score_timing.json", {"score_seconds": time.monotonic() - t0})
PY

PYTHONPATH=src runs/tools/bfcl/bin/python - "$eval_dir" <<'PY'
import sys
from pathlib import Path
from toolcall.evaluation.bfcl import score_summary
from toolcall.evaluation.common import read_json, write_json

eval_dir = Path(sys.argv[1]).resolve()
merged_bfcl = eval_dir / "bfcl"
s0 = read_json(eval_dir / "bfcl_shard0/summary.json")
s1 = read_json(eval_dir / "bfcl_shard1/summary.json")
cfg = read_json(merged_bfcl / "config.json")
score_sec = read_json(merged_bfcl / "score_timing.json")["score_seconds"]
gen_sec = max(s0["metrics"]["generation_seconds"], s1["metrics"]["generation_seconds"])
summary = score_summary(merged_bfcl, cfg["resolved_categories"], gen_sec, score_sec, cfg["max_tokens"], partial=False)
summary["status"] = "completed_with_request_errors" if summary["metrics"].get("request_errors") else "completed"
write_json(merged_bfcl / "summary.json", summary)
PY

.venv/bin/python - "$eval_dir" "$wall_start" <<'PY'
import json, sys, time
from pathlib import Path
from toolcall.evaluation.cli import report_run
from toolcall.evaluation.common import read_json, read_jsonl, write_json
from toolcall.evaluation.telemetry import peak_metrics

eval_dir = Path(sys.argv[1]).resolve()
wall_start = float(sys.argv[2])
merged_bfcl = eval_dir / "bfcl"
summary = read_json(merged_bfcl / "summary.json")
resources = read_jsonl(merged_bfcl / "resources.jsonl")
summary["metrics"].update(peak_metrics(resources), wall_seconds=time.time() - wall_start)
write_json(merged_bfcl / "summary.json", summary)
report_run(merged_bfcl, Path.cwd(), upload=True)

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
write_json(eval_dir / "behavior_gate.json", gate)
print("Unified BFCL Core metrics:", json.dumps({
    "core_weighted_accuracy": metrics["core_weighted_accuracy"],
    "microaccuracy": metrics["microaccuracy"],
    "correct": metrics["correct"],
    "cases": metrics["cases"],
    "generation_seconds": metrics["generation_seconds"],
}))
print("Behavior gate:", json.dumps(gate, indent=2))
PY

.venv/bin/python - "$eval_dir/api-bank/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    metrics = json.load(f)["metrics"]
assert (metrics["samples"], metrics["call_samples"], metrics["response_samples"]) == (1281, 753, 528), metrics
PY

date -u +%FT%TZ > "$eval_dir/completed_utc.txt"
echo "Dual-GPU sharded Core + API-Bank evaluation completed for $ckpt_name -> $eval_dir"
