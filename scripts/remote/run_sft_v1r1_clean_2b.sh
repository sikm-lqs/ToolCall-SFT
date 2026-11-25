#!/usr/bin/env bash
# Run inside a run_tmux.sh snapshot, after submitting the Smoke training session.
set -euo pipefail
: "${TOOLCALL_SNAPSHOT_DIR:?launch via scripts/remote/run_tmux.sh}"
readonly smoke_run="${1:?pass the exact Smoke training run directory}"
readonly smoke_session="${2:?pass the Smoke training tmux session name}"
readonly stage="${3:-all}"
case "$stage" in all|smoke-eval|smoke-api-bank|full|train|full-eval|quick) ;; *) echo "Unknown stage: $stage" >&2; exit 2;; esac
readonly stamp="$(basename "$TOOLCALL_SNAPSHOT_DIR" | grep -oE '[0-9]{8}T[0-9]{6}Z$')"
readonly full_run="${4:-runs/train/sft-v1r1-clean-2b-lora-$stamp}"
readonly smoke_model=sft-v1r1-clean-smoke-2b
readonly full_model=sft-v1r1-clean-full-2b
readonly base_url=http://127.0.0.1:18003/v1
readonly run_meta="runs/train/sft-v1r1-clean-pipeline-$stamp"
mkdir -p "$run_meta"
printf 'smoke_run=%s\nfull_run=%s\n' "$smoke_run" "$full_run" | tee "$run_meta/run_paths.txt"

check_train() {
  .venv/bin/python - "$1/train_summary.json" <<'PY'
import json, math, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
assert d["status"] == "completed", d["status"]
for key in ("loss_mask_check", "forward_check"):
    assert d[key]["status"] == "passed", d[key]
for section, key in (("train_metrics", "train_loss"), ("eval_metrics", "eval_loss")):
    assert math.isfinite(d[section][key]), d[section]
PY
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
    --host 127.0.0.1 --port 18003 --tensor-parallel-size 1 --dtype bfloat16 \
    --language-model-only --max-model-len 65536 --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 --max-num-batched-tokens 32768 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --generation-config vllm --seed 0 --enable-prefix-caching \
    >"$eval_dir/vllm.log" 2>&1 &
  service_pid=$!
  trap 'kill -TERM "$service_pid" 2>/dev/null || true; wait "$service_pid" || true' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  echo "Waiting for vLLM readiness; log: $eval_dir/vllm.log"
  local ready=false
  for _ in {1..72}; do
    if curl --fail --silent --max-time 3 "$base_url/models" >"$eval_dir/models.json"; then
      ready=true
      break
    fi
    if ! kill -0 "$service_pid" 2>/dev/null; then
      tail -50 "$eval_dir/vllm.log"; exit 1
    fi
    sleep 5
  done
  if [[ "$ready" != true ]]; then
    tail -50 "$eval_dir/vllm.log"; exit 1
  fi
  .venv/bin/python - "$base_url" "$model" "$eval_dir" <<'PY'
import json, sys
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

base, model, output = sys.argv[1:]
out = Path(output)
models = json.loads((out / "models.json").read_text())["data"]
assert any(m["id"] == model and m["max_model_len"] == 65536 for m in models), models
tools = [{"type": "function", "function": {
    "name": "lookup_temperature",
    "description": "Get the current temperature for one city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
messages = [{"role": "user", "content": "Use the available tool to get the current temperature in Paris."}]
opener = build_opener(ProxyHandler({}))

def probe(name):
    payload = dict(model=model, messages=messages, tools=tools, tool_choice="auto",
                   temperature=0, max_completion_tokens=1024,
                   chat_template_kwargs={"enable_thinking": False})
    req = Request(base + "/chat/completions", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
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

  local -a eval_args=(--model "$model" --base-url "$base_url" --timeout 300 --gpu-index 0
                     --max-tokens 1024 --temperature 0 --checkpoint-revision "sha256:$digest")
  if [[ "$phase" = smoke || "$phase" = quick-* ]]; then
    local variant=sft-v1r1-clean-smoke
    if [[ "$phase" = quick-* ]]; then variant="sft-v1r1-clean-$phase"; fi
    .venv/bin/python -u -m toolcall.evaluation.cli bfcl-v4 "${eval_args[@]}" \
      --categories simple_python multiple parallel parallel_multiple irrelevance live_irrelevance \
        multi_turn_base multi_turn_miss_func multi_turn_miss_param \
      --concurrency 128 --variant "$variant" --output "$eval_dir/bfcl"
    .venv/bin/python - "$eval_dir" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
summary = json.loads((out / "bfcl/summary.json").read_text())
metrics, groups = summary["metrics"], summary["groups"]
assert metrics["cases"] == 2724, metrics
assert metrics["request_errors"] == metrics["context_rejections"], metrics
limits = {"irrelevance": .76, "live_irrelevance": .66, "parallel": .85, "simple_python": .88}
gate = {name: {"accuracy": groups[name]["accuracy"], "minimum": target,
               "passed": groups[name]["accuracy"] >= target} for name, target in limits.items()}
gate["all_passed"] = all(row["passed"] for row in gate.values())
(out / "behavior_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
if not gate["all_passed"]:
    print("Not all four preset behavior thresholds were met; record as a diagnostic result.")
PY
  elif [[ "$phase" = full ]]; then
    .venv/bin/python -u -m toolcall.evaluation.cli bfcl-v4 "${eval_args[@]}" \
      --profile core --concurrency 128 --variant sft-v1r1-clean-full --output "$eval_dir/bfcl"
  fi
  if [[ "$phase" != smoke ]]; then
    local api_variant=sft-v1r1-clean-full
    case "$phase" in
      quick-*) api_variant="sft-v1r1-clean-$phase" ;;
      smoke-api-bank) api_variant=sft-v1r1-clean-smoke ;;
    esac
    .venv/bin/python -u -m toolcall.evaluation.cli api-bank "${eval_args[@]}" \
      --profile full --concurrency 96 --variant "$api_variant" --output "$eval_dir/api-bank"
    .venv/bin/python - "$eval_dir/api-bank/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    metrics = json.load(f)["metrics"]
assert (metrics["samples"], metrics["call_samples"], metrics["response_samples"]) == (1281, 753, 528), metrics
PY
  fi
)

if [[ "$stage" = all || "$stage" = smoke-eval || "$stage" = smoke-api-bank ]]; then
  echo "Waiting for exact Smoke training session: $smoke_session"
  while tmux has-session -t "=$smoke_session" 2>/dev/null; do sleep 5; done
  check_train "$smoke_run"
  smoke_ckpt="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_model_checkpoint"])' "$smoke_run/train_summary.json")"
  smoke_digest="$(sha256sum "$smoke_ckpt/adapter_model.safetensors" | cut -d' ' -f1)"
  smoke_phase=smoke
  if [[ "$stage" = smoke-api-bank ]]; then smoke_phase=smoke-api-bank; fi
  eval_merged "$smoke_run/merged-best" "$smoke_model" "$smoke_digest" "$smoke_phase"
  echo "Smoke $smoke_phase evaluation completed."
fi

wait_for_idle_gpu() {
  for _ in {1..15}; do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then break; fi
    sleep 2
  done
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "CUDA context still active; refusing to start a second GPU job." >&2
    return 1
  fi
}

if [[ "$stage" = all || "$stage" = full || "$stage" = train ]]; then
  wait_for_idle_gpu
  echo "Starting full SFT from the base model: $full_run"
  test ! -e "$full_run"
  runs/envs/train/bin/python -u -m toolcall.training.sft \
    --config configs/train/sft_v1r1_clean_2b_lora.yaml --output-dir "$full_run"
fi

if [[ "$stage" = all || "$stage" = full || "$stage" = full-eval ]]; then
  check_train "$full_run"
  # Select only on the clean validation set, not on BFCL/API-Bank.
  full_ckpt="$(.venv/bin/python - "$full_run" "$run_meta/checkpoint_selection.json" <<'PY'
import json, math, sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
candidates = []
for path in root.glob("checkpoint-*"):
    step = int(path.name.split("-")[-1])
    state = json.loads((path / "trainer_state.json").read_text())
    rows = [m for m in state["log_history"] if m.get("step") == step and "eval_loss" in m]
    if rows and math.isfinite(rows[-1]["eval_loss"]):
        candidates.append(dict(path=str(path), step=step, epoch=rows[-1]["epoch"],
                               eval_loss=rows[-1]["eval_loss"]))
candidates.sort(key=lambda row: row["step"])
selected = min(candidates, key=lambda row: (row["eval_loss"], row["step"]))
output.write_text(json.dumps({"criterion": "lowest clean validation loss",
                              "candidates": candidates, "selected": selected}, indent=2) + "\n")
print(selected["path"])
PY
)"
  full_digest="$(sha256sum "$full_ckpt/adapter_model.safetensors" | cut -d' ' -f1)"
  merged="$full_run/merged-${full_ckpt##*/}"
  if [[ ! -f "$merged/merge_info.json" ]]; then
    runs/envs/train/bin/python -u -m toolcall.training.merge_lora --adapter-dir "$full_ckpt" --output-dir "$merged"
  fi
  eval_merged "$merged" "$full_model" "$full_digest" full
  echo "Full SFT evaluation completed."
fi

if [[ "$stage" = quick ]]; then
  for step in 137 274; do
    test -s "$full_run/checkpoint-$step/adapter_model.safetensors"
    test -s "$full_run/checkpoint-$step/trainer_state.json"
  done
  for step in 137 274; do
    wait_for_idle_gpu
    ckpt="$full_run/checkpoint-$step"
    digest="$(sha256sum "$ckpt/adapter_model.safetensors" | cut -d' ' -f1)"
    merged="$full_run/merged-checkpoint-$step"
    if [[ ! -f "$merged/merge_info.json" ]]; then
      runs/envs/train/bin/python -u -m toolcall.training.merge_lora \
        --adapter-dir "$ckpt" --output-dir "$merged"
    fi
    eval_merged "$merged" "$full_model-ckpt-$step" "$digest" "quick-ckpt-$step"
  done
fi
date -u +%FT%TZ > "$run_meta/completed_utc.txt"
