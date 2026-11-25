#!/usr/bin/env bash
# Merge selected LoRA checkpoints sequentially and run the 9-category BFCL + full API-Bank quick eval.
# Must be launched via scripts/remote/run_tmux.sh inside an isolated snapshot.
set -euo pipefail

: "${TOOLCALL_SNAPSHOT_DIR:?launch via scripts/remote/run_tmux.sh}"

wait_session=""
eval_mode="quick"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait-session)
      wait_session="${2:?--wait-session requires a tmux session name}"
      shift 2
      ;;
    --mode)
      eval_mode="${2:?--mode requires quick or core}"
      case "$eval_mode" in quick|core) ;; *) echo "Unknown --mode: $eval_mode" >&2; exit 2;; esac
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 3 ]]; then
  echo "Usage: run_quick_eval.sh [--wait-session <session>] [--mode quick|core] <train_run_dir> <exp_prefix> <step1> [step2 ...]" >&2
  exit 2
fi

readonly train_run="$1"
readonly exp_prefix="$2"
shift 2
readonly steps=("$@")

if [[ -n "$wait_session" ]]; then
  echo "Waiting for tmux session '$wait_session' to finish before starting..."
  while tmux has-session -t "=$wait_session" 2>/dev/null; do
    sleep 10
  done
  echo "Session '$wait_session' exited; proceeding."
fi

readonly stamp="$(basename "$TOOLCALL_SNAPSHOT_DIR" | grep -oE '[0-9]{8}T[0-9]{6}Z$')"
readonly base_url="${TOOLCALL_EVAL_BASE_URL:-http://127.0.0.1:18003/v1}"
readonly port="${base_url##*:}"
readonly host_port="${port%%/*}"
readonly run_meta="runs/eval/${exp_prefix}-${eval_mode}-${stamp}"
mkdir -p "$run_meta"

printf 'train_run=%s\nexp_prefix=%s\neval_mode=%s\nsteps=%s\n' \
  "$train_run" "$exp_prefix" "$eval_mode" "${steps[*]}" | tee "$run_meta/run_paths.txt"

.venv/bin/python - "$train_run/train_summary.json" <<'PY'
import json, math, sys
with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)
assert d["status"] == "completed", d["status"]
for key in ("loss_mask_check", "forward_check"):
    assert d[key]["status"] == "passed", d[key]
for section, key in (("train_metrics", "train_loss"), ("eval_metrics", "eval_loss")):
    assert math.isfinite(d[section][key]), d[section]
PY

for step in "${steps[@]}"; do
  test -s "$train_run/checkpoint-$step/adapter_model.safetensors"
  test -s "$train_run/checkpoint-$step/trainer_state.json"
done

wait_for_idle_gpu() {
  for _ in {1..15}; do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
      return 0
    fi
    sleep 2
  done
  echo "CUDA context still active; refusing to start next GPU job." >&2
  return 1
}

eval_merged() (
  local merged="$1" model="$2" digest="$3" phase="$4"
  local eval_dir="$run_meta/$phase"
  mkdir -p "$eval_dir"
  if curl --silent --max-time 2 "$base_url/models" >/dev/null; then
    echo "Refusing to replace a running service at $base_url" >&2
    exit 1
  fi
  local service_pid
  runs/envs/serve/bin/vllm serve "$merged" --served-model-name "$model" \
    --host 127.0.0.1 --port "$host_port" --tensor-parallel-size 1 --dtype bfloat16 \
    --language-model-only --max-model-len 65536 --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 --max-num-batched-tokens 32768 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --generation-config vllm --seed 0 --enable-prefix-caching \
    >"$eval_dir/vllm.log" 2>&1 &
  service_pid=$!
  trap 'kill -TERM "$service_pid" 2>/dev/null || true; wait "$service_pid" || true' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  echo "Waiting for vLLM readiness ($model); log: $eval_dir/vllm.log"
  local ready=false
  for _ in {1..72}; do
    if curl --fail --silent --max-time 3 "$base_url/models" >"$eval_dir/models.json"; then
      ready=true
      break
    fi
    if ! kill -0 "$service_pid" 2>/dev/null; then
      tail -50 "$eval_dir/vllm.log"
      exit 1
    fi
    sleep 5
  done
  if [[ "$ready" != true ]]; then
    tail -50 "$eval_dir/vllm.log"
    exit 1
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
call = calls[0]
assert call["function"]["name"] == "lookup_temperature", call
assert json.loads(call["function"]["arguments"])["city"].lower() == "paris", call
assert "<tool_call>" not in (msg.get("content") or ""), first

messages.extend([
    {"role": "assistant", "content": msg.get("content"), "tool_calls": calls},
    {"role": "tool", "tool_call_id": call["id"], "content": '{"temperature_c":21}'},
])
final = probe("tool_response")
msg = final["message"]
assert final["finish_reason"] == "stop" and not msg.get("tool_calls"), final
assert "21" in (msg.get("content") or "") and "<tool_call>" not in msg["content"], final
print("Structured call and reply after tool output both passed.")
PY

  local -a eval_args=(
    --model "$model"
    --base-url "$base_url"
    --timeout 300
    --gpu-index 0
    --max-tokens 1024
    --temperature 0
    --checkpoint-revision "sha256:$digest"
  )
  local variant="${exp_prefix}-${phase}"

  if [[ "$eval_mode" = "core" ]]; then
    .venv/bin/python -u -m toolcall.evaluation.cli bfcl-v4 "${eval_args[@]}" \
      --profile core --concurrency 128 --variant "$variant" --output "$eval_dir/bfcl"
  else
    .venv/bin/python -u -m toolcall.evaluation.cli bfcl-v4 "${eval_args[@]}" \
      --categories simple_python multiple parallel parallel_multiple irrelevance live_irrelevance \
        multi_turn_base multi_turn_miss_func multi_turn_miss_param \
      --concurrency 128 --variant "$variant" --output "$eval_dir/bfcl"
  fi

  .venv/bin/python - "$eval_dir" "$eval_mode" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
mode = sys.argv[2]
summary = json.loads((out / "bfcl/summary.json").read_text())
metrics, groups = summary["metrics"], summary["groups"]
expected_cases = 4441 if mode == "core" else 2724
assert metrics["cases"] == expected_cases, metrics
if metrics["request_errors"] != metrics["context_rejections"]:
    print(f"Diagnostic notice: request_errors={metrics['request_errors']} vs context_rejections={metrics['context_rejections']}")
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
if not gate["all_passed"]:
    print("Not all four preset behavior thresholds were met; record as a diagnostic result.")
PY

  .venv/bin/python -u -m toolcall.evaluation.cli api-bank "${eval_args[@]}" \
    --profile full --concurrency 96 --variant "$variant" --output "$eval_dir/api-bank"

  .venv/bin/python - "$eval_dir/api-bank/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    metrics = json.load(f)["metrics"]
assert (metrics["samples"], metrics["call_samples"], metrics["response_samples"]) == (1281, 753, 528), metrics
PY
)

for step in "${steps[@]}"; do
  wait_for_idle_gpu
  ckpt="$train_run/checkpoint-$step"
  digest="$(sha256sum "$ckpt/adapter_model.safetensors" | cut -d' ' -f1)"
  merged="$train_run/merged-checkpoint-$step"
  if [[ ! -f "$merged/merge_info.json" ]]; then
    runs/envs/train/bin/python -u -m toolcall.training.merge_lora \
      --adapter-dir "$ckpt" --output-dir "$merged"
  fi
  eval_merged "$merged" "${exp_prefix}-ckpt-${step}" "$digest" "${eval_mode}-ckpt-${step}"
done

date -u +%FT%TZ > "$run_meta/completed_utc.txt"
echo "${eval_mode} evaluation completed for steps: ${steps[*]} -> $run_meta"
