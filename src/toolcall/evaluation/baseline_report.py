"""Generate Markdown and standalone HTML baseline evaluation reports from run directories."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

from .common import read_json, read_jsonl

SYSTEM_APIBANK_REASONS = frozenset(
    {
        "inference_error",
        "backend_resource_missing",
        "retriever_exception",
    }
)


def md_cell(text: str) -> str:
    """Escape pipe characters so Markdown table cells stay intact."""
    return text.replace("|", "\\|")


def truncate(text: object, limit: int = 260) -> str:
    """Return a single-line bounded preview of arbitrary text or JSON."""
    if text is None:
        return ""
    raw = (
        text
        if isinstance(text, str)
        else json.dumps(text, ensure_ascii=False, separators=(",", ":"))
    )
    compact = re.sub(r"\s+", " ", raw).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def fmt_num(value: float | int | None, digits: int = 2, suffix: str = "") -> str:
    """Format a numeric metric cleanly or return N/A when missing."""
    if value is None:
        return "N/A"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}{suffix}"
    return f"{value:,.{digits}f}{suffix}"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    """Format a 0-1 fraction as a percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def discover_runs(eval_root: Path) -> list[Path]:
    """Find the most recent valid run per (benchmark, model) under runs/eval."""
    candidates: dict[tuple[str, str], list[tuple[int, float, Path]]] = {}
    if not eval_root.is_dir():
        return []
    for summary_path in eval_root.rglob("summary.json"):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        config = read_json(config_path)
        summary = read_json(summary_path)
        benchmark = str(config.get("benchmark", run_dir.parent.parent.name))
        model = str(config.get("model", "unknown"))
        profile = str(config.get("profile", ""))
        status = str(summary.get("status", ""))
        # Rank clean full/no-web completed runs ahead of smoke/historical runs.
        priority = (
            (3 if profile in {"full", "no-web"} else 1 if profile == "custom" else 0)
            + (2 if status == "completed" else 0)
            + (1 if not run_dir.name.startswith("history-") else 0)
        )
        candidates.setdefault((benchmark, model), []).append(
            (priority, summary_path.stat().st_mtime, run_dir)
        )
    chosen = []
    for key in sorted(candidates):
        best = max(candidates[key], key=lambda item: (item[0], item[1]))
        chosen.append(best[2])
    return chosen


def downsample_resources(rows: list[dict], max_points: int = 120) -> list[dict]:
    """Downsample resources.jsonl rows while keeping endpoints and compact fields."""
    if not rows:
        return []
    step = max(1, len(rows) // max_points)
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    out = []
    for row in sampled:
        if "elapsed_seconds" not in row:
            continue
        out.append(
            {
                "t": round(float(row["elapsed_seconds"]), 2),
                "gpu_mib": round(float(row["gpu_used_mib"]), 1)
                if "gpu_used_mib" in row
                else None,
                "gpu_pct": round(float(row["gpu_utilization_pct"]), 1)
                if "gpu_utilization_pct" in row
                else None,
                "kv_pct": round(float(row["kv_cache_usage_perc"]) * 100, 2)
                if "kv_cache_usage_perc" in row
                else None,
                "running": row.get("num_requests_running"),
                "waiting": row.get("num_requests_waiting"),
            }
        )
    return out


def extract_bfcl_failures(
    run_dir: Path, summary: dict
) -> tuple[dict[str, int], dict[str, int], list[dict]]:
    """Extract system and LLM failure statistics and examples for a BFCL run."""
    metrics = summary.get("metrics", {})
    sys_counts: Counter[str] = Counter()
    llm_counts: Counter[str] = Counter()
    examples: list[dict] = []

    req_errors = metrics.get("request_errors", metrics.get("model_failed_requests", 0))
    ctx_rej = metrics.get(
        "context_rejections", metrics.get("scored_cases_with_context_rejection", 0)
    )
    len_stops = metrics.get("length_stops", metrics.get("length_stop_requests", 0))
    if req_errors:
        sys_counts["request_errors"] = int(req_errors)
    if ctx_rej:
        sys_counts["context_rejections"] = int(ctx_rej)
    if len_stops:
        sys_counts["length_stops"] = int(len_stops)

    requests = read_jsonl(run_dir / "model-requests.jsonl")
    for row in requests:
        if "error" in row and len([e for e in examples if e["kind"] == "system"]) < 2:
            examples.append(
                {
                    "kind": "system",
                    "id": row.get("id", "unknown"),
                    "category": "request",
                    "reason": "request_error",
                    "prompt": f"step={row.get('step')}",
                    "output": truncate(row.get("error")),
                    "expected": "HTTP 200 valid response",
                }
            )
        elif (
            row.get("finish_reason") == "length"
            and len([e for e in examples if e["kind"] == "system"]) < 3
        ):
            examples.append(
                {
                    "kind": "system",
                    "id": row.get("id", "unknown"),
                    "category": "request",
                    "reason": "length_stop",
                    "prompt": f"step={row.get('step')}",
                    "output": f"output_tokens={row.get('output_tokens')} (hit cap)",
                    "expected": "finish_reason in {tool_calls, stop}",
                }
            )

    raw_outputs = {}
    result_root = run_dir / "result"
    if result_root.is_dir():
        for path in result_root.rglob("BFCL_v4_*_result.json"):
            for row in read_jsonl(path):
                if "id" in row:
                    raw_outputs[row["id"]] = row

    seen_single_types: set[str] = set()
    seen_multi_types: set[str] = set()
    single_examples = 0
    multi_examples = 0
    score_root = run_dir / "score"
    if score_root.is_dir():
        for path in sorted(score_root.rglob("BFCL_v4_*_score.json")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines[1:]:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("valid", True):
                    continue
                err = row.get("error")
                err_type = str(
                    row.get("error_type")
                    or (err.get("error_type") if isinstance(err, dict) else None)
                    or "unknown_error"
                )
                if "inference_error" in err_type:
                    sys_counts[err_type] += 1
                    continue
                llm_counts[err_type] += 1
                category = str(row.get("test_category", path.stem))
                is_multi = category.startswith("multi_turn") or err_type.startswith(
                    "multi_turn:"
                )
                if is_multi:
                    take = err_type not in seen_multi_types and multi_examples < 4
                else:
                    take = err_type not in seen_single_types and single_examples < 4
                if take:
                    if is_multi:
                        seen_multi_types.add(err_type)
                        multi_examples += 1
                    else:
                        seen_single_types.add(err_type)
                        single_examples += 1
                    case_id = row.get("id", "unknown")
                    prompt_obj = row.get("prompt", {})
                    question = (
                        prompt_obj.get("question", prompt_obj)
                        if isinstance(prompt_obj, dict)
                        else prompt_obj
                    )
                    raw_res = raw_outputs.get(case_id, {}).get(
                        "result",
                        row.get("model_result_raw", row.get("model_result_decoded")),
                    )
                    if is_multi and isinstance(err, dict) and err.get("error_message"):
                        output_preview = (
                            f"{err['error_message']} | steps={truncate(raw_res, 140)}"
                        )
                    else:
                        output_preview = truncate(raw_res or "(empty tool_calls)")
                    examples.append(
                        {
                            "kind": "llm",
                            "id": case_id,
                            "category": category,
                            "reason": err_type,
                            "prompt": truncate(question),
                            "output": output_preview,
                            "expected": truncate(row.get("possible_answer")),
                        }
                    )

    if not llm_counts:
        # Historical runs may only preserve aggregate group counts without per-case score files.
        for cat, group in summary.get("groups", {}).items():
            wrong = int(group.get("total_count", 0)) - int(group.get("correct_count", 0))
            if wrong > 0:
                llm_counts[f"category_incorrect:{cat}"] += wrong

    return dict(sys_counts), dict(llm_counts), examples


def extract_apibank_failures(
    run_dir: Path, summary: dict
) -> tuple[dict[str, int], dict[str, int], list[dict]]:
    """Extract system and LLM failures from API-Bank scores and predictions."""
    metrics = summary.get("metrics", {})
    sys_counts: Counter[str] = Counter()
    llm_counts: Counter[str] = Counter()
    examples: list[dict] = []

    for key in ("request_errors", "context_rejections", "length_stops"):
        val = metrics.get(key, 0)
        if val:
            sys_counts[key] = int(val)

    scores = read_jsonl(run_dir / "scores.jsonl")
    preds = {row["id"]: row for row in read_jsonl(run_dir / "predictions.jsonl")}
    seen_reasons: set[str] = set()

    for row in scores:
        case_id = row["id"]
        pred = preds.get(case_id, {})
        if row.get("track") == "response":
            content = (pred.get("message") or {}).get("content") or ""
            if "<tool_call>" in content:
                llm_counts["response_leaked_tool_call"] += 1
                if "response_leaked_tool_call" not in seen_reasons:
                    seen_reasons.add("response_leaked_tool_call")
                    examples.append(
                        {
                            "kind": "llm",
                            "id": case_id,
                            "category": f"{row['level']}:response",
                            "reason": "response_leaked_tool_call (tool_choice=none)",
                            "prompt": f"dialogue={row.get('dialogue')} step={row.get('step')}",
                            "output": truncate(content),
                            "expected": "Natural language response without <tool_call>",
                        }
                    )
            continue

        if row.get("correct"):
            continue
        reason = str(row.get("reason") or "unknown")
        oracle_ok = bool(row.get("oracle_correct", True))
        if reason in SYSTEM_APIBANK_REASONS or (
            not oracle_ok
            and (reason == "tool_exception" or reason.startswith("execution_error:"))
        ):
            sys_counts[reason] += 1
            if (
                reason not in seen_reasons
                and len([e for e in examples if e["kind"] == "system"]) < 3
            ):
                seen_reasons.add(reason)
                examples.append(
                    {
                        "kind": "system",
                        "id": case_id,
                        "category": f"{row['level']}:call",
                        "reason": f"{reason} (oracle_correct={oracle_ok})",
                        "prompt": f"gold_api={row.get('gold_api')}",
                        "output": truncate(
                            pred.get("error")
                            or (pred.get("message") or {}).get("tool_calls")
                            or (pred.get("message") or {}).get("content")
                        ),
                        "expected": f"oracle_reason={row.get('oracle_reason')}",
                    }
                )
        else:
            llm_counts[reason] += 1
            if (
                reason not in seen_reasons
                and len([e for e in examples if e["kind"] == "llm"]) < 5
            ):
                seen_reasons.add(reason)
                msg = pred.get("message") or {}
                examples.append(
                    {
                        "kind": "llm",
                        "id": case_id,
                        "category": f"{row['level']}:call",
                        "reason": reason,
                        "prompt": f"gold_api={row.get('gold_api')} (offered={row.get('gold_in_tools')})",
                        "output": truncate(
                            msg.get("tool_calls") or msg.get("content") or "(empty)"
                        ),
                        "expected": f"api={row.get('gold_api')}",
                    }
                )

    return dict(sys_counts), dict(llm_counts), examples


def extract_tau2_failures(
    run_dir: Path, summary: dict
) -> tuple[dict[str, int], dict[str, int], list[dict]]:
    """Extract system and LLM failures from tau2 domain simulation JSON files."""
    metrics = summary.get("metrics", {})
    sys_counts: Counter[str] = Counter()
    llm_counts: Counter[str] = Counter()
    examples: list[dict] = []

    for key in (
        "agent_request_errors",
        "agent_context_rejections",
        "agent_length_stops",
    ):
        val = metrics.get(key, 0)
        if val:
            sys_counts[key] = int(val)

    seen_reasons: set[str] = set()
    for path in sorted(run_dir.glob("*.json")):
        if path.name in {"config.json", "summary.json", "swanlab.json"}:
            continue
        domain = path.stem
        data = read_json(path)
        for sim in data.get("simulations", []):
            reward = float((sim.get("reward_info") or {}).get("reward", 0.0))
            term = str(sim.get("termination_reason") or "unknown")
            if reward >= 1.0 - 1e-9:
                continue
            is_system = "error" in term.lower()
            target = sys_counts if is_system else llm_counts
            bucket = f"{domain}:{term}"
            target[bucket] += 1
            kind = "system" if is_system else "llm"
            if bucket not in seen_reasons and len(
                [e for e in examples if e["kind"] == kind]
            ) < (3 if is_system else 5):
                seen_reasons.add(bucket)
                msgs = sim.get("messages") or []
                first_user = next(
                    (m.get("content") for m in msgs if m.get("role") == "user"), ""
                )
                last_agent = next(
                    (
                        m.get("tool_calls") or m.get("content")
                        for m in reversed(msgs)
                        if m.get("role") == "assistant"
                    ),
                    "",
                )
                examples.append(
                    {
                        "kind": kind,
                        "id": f"{domain}:task_{sim.get('task_id')}:trial_{sim.get('trial')}",
                        "category": domain,
                        "reason": f"termination={term}, reward={reward:.2f}",
                        "prompt": truncate(first_user),
                        "output": truncate(last_agent or "(no assistant msg)"),
                        "expected": truncate(
                            (sim.get("reward_info") or {}).get("info")
                            or "DB/Policy task completion (reward=1.0)"
                        ),
                    }
                )

    return dict(sys_counts), dict(llm_counts), examples


def primary_metric(benchmark: str, profile: str, summary: dict) -> tuple[str, float | None, str]:
    """Return (label, score_0_to_1, fraction_text) for a benchmark summary."""
    metrics = summary.get("metrics", {})
    if benchmark == "bfcl-v4":
        if "official_overall_accuracy" in metrics:
            label = "Official Overall"
            score = metrics["official_overall_accuracy"]
        elif "no_web_weighted_accuracy" in metrics:
            label = "No-Web Weighted"
            score = metrics["no_web_weighted_accuracy"]
        elif "core_weighted_accuracy" in metrics:
            label = "Core Weighted (No-Agentic)"
            score = metrics["core_weighted_accuracy"]
        elif "bfcl_aggregate_without_web_renormalized" in metrics:
            label = "No-Web Weighted (Hist.)"
            score = metrics["bfcl_aggregate_without_web_renormalized"]
        else:
            label = "Microaccuracy"
            score = metrics.get("microaccuracy")
        total = metrics.get("cases", metrics.get("scored_cases"))
        correct = metrics.get("correct")
        frac = f"{correct} / {total}" if total is not None and correct is not None else "N/A"
        return label, score, frac
    if benchmark == "api-bank":
        score = metrics.get("call_microaccuracy")
        total = metrics.get("call_samples")
        correct = metrics.get("correct_calls")
        frac = f"{correct} / {total}" if total is not None and correct is not None else "N/A"
        return "Call Accuracy", score, frac
    score = metrics.get("domain_mean_task_success_rate")
    groups = summary.get("groups", {})
    total = sum(int(g.get("simulations", 0)) for g in groups.values())
    passed = round(
        sum(float(g.get("task_success_rate", 0)) * int(g.get("simulations", 0)) for g in groups.values())
    )
    frac = f"{passed} / {total}" if total else "N/A"
    return "Domain Mean Pass^1", score, frac


def extract_run_record(run_dir: Path) -> dict:
    """Parse one run directory into a normalized report record."""
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "summary.json")
    resources = read_jsonl(run_dir / "resources.jsonl")
    metrics = summary.get("metrics", {})
    benchmark = str(config.get("benchmark", ""))
    profile = str(
        config.get("profile")
        or ("historical" if run_dir.name.startswith("history-") else "custom")
    )
    is_tau2 = benchmark == "tau2"
    prefix = "agent_" if is_tau2 and "agent_calls" in metrics else ""

    primary_label, primary_score, cases_frac = primary_metric(
        benchmark, profile, summary
    )

    if benchmark == "bfcl-v4":
        sys_errs, llm_errs, examples = extract_bfcl_failures(run_dir, summary)
    elif benchmark == "api-bank":
        sys_errs, llm_errs, examples = extract_apibank_failures(run_dir, summary)
    else:
        sys_errs, llm_errs, examples = extract_tau2_failures(run_dir, summary)

    subgroups = {}
    for name, row in summary.get("groups", {}).items():
        if benchmark == "tau2":
            acc = row.get("task_success_rate")
            count = row.get("simulations", row.get("tasks"))
            correct = round(acc * count) if acc is not None and count is not None else None
        elif benchmark == "api-bank":
            acc = row.get("accuracy")
            count = row.get("calls")
            correct = row.get("correct")
        else:
            acc = row.get("accuracy")
            count = row.get("total_count")
            correct = row.get("correct_count")
        subgroups[name] = {
            "accuracy": acc,
            "correct": correct,
            "total": count,
            "extra": {
                k: v
                for k, v in row.items()
                if k
                in {
                    "rouge_l_f1",
                    "oracle_accuracy",
                    "reference_tool_coverage",
                    "average_reward",
                }
            },
        }

    kv_raw = metrics.get("peak_kv_cache_usage_perc", metrics.get("peak_kv_cache_ratio"))
    idle_mib = next(
        (r["gpu_used_mib"] for r in resources if "gpu_used_mib" in r), None
    )

    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "benchmark": benchmark,
        "model": str(config.get("model", "unknown")),
        "profile": profile,
        "variant": config.get("variant") or "baseline",
        "status": summary.get("status", "unknown"),
        "context_window": config.get("context_window"),
        "max_tokens": config.get("max_tokens"),
        "concurrency": config.get("concurrency"),
        "user_model": config.get("user_model") if is_tau2 else None,
        "primary_label": primary_label,
        "primary_score": primary_score,
        "cases_frac": cases_frac,
        "subgroups": subgroups,
        "wall_seconds": metrics.get(
            "wall_seconds", metrics.get("total_active_wall_seconds")
        ),
        "stage_seconds": metrics.get(
            "generation_seconds", metrics.get("simulation_seconds")
        ),
        "scoring_seconds": metrics.get("scoring_seconds"),
        "calls": metrics.get(
            f"{prefix}calls", metrics.get("model_successful_requests")
        ),
        "successful_calls": metrics.get(
            f"{prefix}successful_calls", metrics.get("model_successful_requests")
        ),
        "request_errors": metrics.get(
            f"{prefix}request_errors", metrics.get("model_failed_requests", 0)
        ),
        "context_rejections": metrics.get(
            f"{prefix}context_rejections",
            metrics.get("scored_cases_with_context_rejection", 0),
        ),
        "length_stops": metrics.get(
            f"{prefix}length_stops", metrics.get("length_stop_requests", 0)
        ),
        "input_tokens": metrics.get(f"{prefix}input_tokens", metrics.get("input_tokens")),
        "output_tokens": metrics.get(
            f"{prefix}output_tokens", metrics.get("output_tokens")
        ),
        "output_tok_s": metrics.get(
            f"{prefix}output_tokens_per_second", metrics.get("output_tokens_per_second")
        ),
        "total_tok_s": metrics.get(
            f"{prefix}total_tokens_per_second", metrics.get("total_tokens_per_second")
        ),
        "latency_p95_s": metrics.get(
            f"{prefix}latency_p95_seconds",
            metrics.get("model_call_latency_p95_seconds"),
        ),
        "idle_gpu_mib": idle_mib,
        "peak_gpu_mib": metrics.get(
            "peak_gpu_used_mib", metrics.get("gpu_used_peak_mib")
        ),
        "peak_gpu_util_pct": metrics.get(
            "peak_gpu_utilization_pct", metrics.get("gpu_util_avg_pct")
        ),
        "peak_kv_pct": kv_raw * 100 if kv_raw is not None else None,
        "peak_running": metrics.get(
            "peak_num_requests_running", metrics.get("peak_requests_running")
        ),
        "peak_waiting": metrics.get(
            "peak_num_requests_waiting", metrics.get("peak_requests_waiting")
        ),
        "preemptions": metrics.get("preemptions"),
        "system_errors": sys_errs,
        "llm_errors": llm_errs,
        "failure_examples": examples,
        "resource_series": downsample_resources(resources),
    }


def render_markdown(records: list[dict], title: str) -> str:
    """Render a structured Markdown evaluation report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {title}",
        "",
        f"> 生成时间：{now} · 纳入运行数：{len(records)}",
        "",
        "## 1. 评测范围与主分总览",
        "",
        "| 模型 | 评测集 | 范围 (Profile) | 状态 | 指标口径 | 正确/总数 | 主分 | 上下文 / 上限 / 并发 | Run ID |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        cfg = f"{r['context_window'] or 'N/A'} / {r['max_tokens'] or 'N/A'} / {r['concurrency'] or 'N/A'}"
        lines.append(
            f"| `{r['model']}` | `{r['benchmark']}` | `{r['profile']}` | `{r['status']}` | "
            f"{r['primary_label']} | {r['cases_frac']} | **{fmt_pct(r['primary_score'])}** | "
            f"{cfg} | `{r['run_id']}` |"
        )

    lines.extend(
        [
            "",
            "## 2. 时间、吞吐与延迟指标",
            "",
            "| 模型 | 评测集 | Wall (s) | 生成/模拟 (s) | 评分 (s) | 输入 Tokens | 输出 Tokens | 输出 tok/s | 总 tok/s | P95 延迟 (s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in records:
        lines.append(
            f"| `{r['model']}` | `{r['benchmark']}` (`{r['profile']}`) | "
            f"{fmt_num(r['wall_seconds'], 1)} | {fmt_num(r['stage_seconds'], 1)} | "
            f"{fmt_num(r['scoring_seconds'], 1)} | {fmt_num(r['input_tokens'], 0)} | "
            f"{fmt_num(r['output_tokens'], 0)} | {fmt_num(r['output_tok_s'], 1)} | "
            f"{fmt_num(r['total_tok_s'], 1)} | {fmt_num(r['latency_p95_s'], 2)} |"
        )

    lines.extend(
        [
            "",
            "## 3. 显卡与 vLLM 服务压力",
            "",
            "| 模型 | 评测集 | 初始显存 (MiB) | 峰值显存 (MiB) | GPU 利用率峰值 | KV Cache 峰值 | 运行/排队峰值 | 抢占次数 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in records:
        rw = (
            f"{fmt_num(r['peak_running'], 0)} / {fmt_num(r['peak_waiting'], 0)}"
            if r["peak_running"] is not None
            else "N/A"
        )
        lines.append(
            f"| `{r['model']}` | `{r['benchmark']}` (`{r['profile']}`) | "
            f"{fmt_num(r['idle_gpu_mib'], 0)} | {fmt_num(r['peak_gpu_mib'], 0)} | "
            f"{fmt_num(r['peak_gpu_util_pct'], 1, '%')} | {fmt_num(r['peak_kv_pct'], 2, '%')} | "
            f"{rw} | {fmt_num(r['preemptions'], 0)} |"
        )

    lines.extend(["", "## 4. 分项得分明细", ""])
    for r in records:
        lines.append(f"### {r['model']} · `{r['benchmark']}` (`{r['profile']}` · `{r['run_id']}`)")
        lines.append("")
        lines.append("| 子类 / 域 | 正确 / 总数 | 准确率 | 补充指标 |")
        lines.append("| --- | --- | --- | --- |")
        for name, sub in r["subgroups"].items():
            extras = ", ".join(
                f"{k}={fmt_pct(v) if 'accuracy' in k or 'coverage' in k else fmt_num(v, 4)}"
                for k, v in sub["extra"].items()
                if v is not None
            )
            frac = (
                f"{sub['correct']} / {sub['total']}"
                if sub["correct"] is not None and sub["total"] is not None
                else "N/A"
            )
            lines.append(
                f"| `{name}` | {frac} | {fmt_pct(sub['accuracy'])} | {extras or '—'} |"
            )
        lines.append("")

    lines.extend(["## 5. 错误归因与典型错题（系统错误 vs 真实 LLM 错误）", ""])
    for r in records:
        lines.append(f"### {r['model']} · `{r['benchmark']}` (`{r['profile']}`)")
        lines.append("")
        sys_str = (
            ", ".join(f"`{k}`: {v}" for k, v in r["system_errors"].items()) or "无"
        )
        llm_str = (
            ", ".join(f"`{k}`: {v}" for k, v in sorted(r["llm_errors"].items(), key=lambda x: -x[1])[:8])
            or "无"
        )
        lines.append(f"- **系统 / 基础设施错误**：{sys_str}")
        lines.append(f"- **真实 LLM 能力错误**：{llm_str}")
        if r["failure_examples"]:
            lines.append("")
            lines.append("| 归因分类 | Case ID | 子类 | 错误原因 | 模型输出摘要 | 期望 / 参考 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for ex in r["failure_examples"]:
                tag = "系统/服务" if ex["kind"] == "system" else "LLM 能力"
                lines.append(
                    f"| {tag} | `{md_cell(str(ex['id']))}` | `{md_cell(str(ex['category']))}` | "
                    f"`{md_cell(str(ex['reason']))}` | `{md_cell(truncate(ex['output'], 110))}` | "
                    f"`{md_cell(truncate(ex['expected'], 90))}` |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f7f7;--card:#fff;--ink:#132a2f;--muted:#536b70;--line:#dbe6e6;--navy:#0f2b31;--teal:#0f766e;--blue:#285ca4;--amber:#b45309;--red:#b91c1c;--mono:"SFMono-Regular",Consolas,monospace;--sans:"Inter","Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 var(--sans)}
.wrap{width:min(1220px,calc(100% - 40px));margin:0 auto}
header{background:var(--navy);color:#eef8f6;padding:34px 0 30px}
header h1{margin:6px 0 8px;font-size:clamp(1.6rem,3vw,2.4rem);letter-spacing:-.02em}
.eyebrow{font:600 12px/1.4 var(--mono);color:#7fe3cb;letter-spacing:.06em;text-transform:uppercase}
header p{margin:0;color:#b8d4d2;font-size:.95rem}
main{padding:26px 0 56px}
section{margin-top:34px}
section:first-child{margin-top:0}
h2{font-size:1.25rem;margin:0 0 12px;border-left:4px solid var(--teal);padding-left:10px}
h3{font-size:1rem;margin:0 0 10px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.kpi-label{font-size:.82rem;color:var(--muted)}
.kpi-value{font:700 1.65rem/1.25 var(--mono);color:var(--teal);margin:6px 0 4px}
.kpi-sub{font:12px/1.4 var(--mono);color:var(--muted)}
.table-box{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{width:100%;border-collapse:collapse;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
thead{background:#eaf2f1;color:#2f4d52}
tbody tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;white-space:nowrap}
code,.mono{font-family:var(--mono);font-size:.85em}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font:600 11px/1.4 var(--mono)}
.b-ok{background:#e3f5ef;color:#0b6058}.b-warn{background:#fef3c7;color:#92400e}.b-err{background:#fee2e2;color:#991b1b}
.grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}
.bar-row{display:grid;grid-template-columns:170px minmax(0,1fr) 72px;gap:10px;align-items:center;margin:7px 0}
.bar-label{font:12px/1.3 var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-track{height:10px;background:#e8efef;border-radius:4px;overflow:hidden}
.bar-fill{height:100%;background:var(--teal);border-radius:4px}
.bar-val{text-align:right;font:600 12px/1 var(--mono)}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.tabs button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;border-radius:6px;font:600 12px/1.2 var(--mono);cursor:pointer}
.tabs button.active{background:var(--teal);color:#fff;border-color:var(--teal)}
canvas{width:100%;height:230px;display:block}
.case-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.case-card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.case-card pre{margin:6px 0 0;padding:8px 10px;background:#f7fafa;border:1px solid #e5eeee;border-radius:6px;white-space:pre-wrap;word-break:break-word;font:12px/1.45 var(--mono)}
.case-meta{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:6px}
footer{padding:20px 0 34px;color:var(--muted);font-size:.82rem;border-top:1px solid var(--line)}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="eyebrow">ToolCall-SFT · Baseline Evaluation Report</div>
    <h1>__TITLE__</h1>
    <p>生成时间：__NOW__ · 覆盖质量得分、时间与吞吐、显卡/KV Cache 压力及「系统错误 vs 真实 LLM 错误」归因</p>
  </div>
</header>
<main class="wrap">
  <section>
    <h2>1. 总览卡片</h2>
    <div class="kpis" id="kpi-cards"></div>
  </section>

  <section>
    <h2>2. 核心质量与子集对比</h2>
    <div class="table-box">
      <table>
        <thead><tr><th>模型</th><th>评测集</th><th>Profile</th><th>状态</th><th>主指标</th><th class="num">正确 / 总数</th><th class="num">主分</th><th>上下文 / 上限 / 并发</th></tr></thead>
        <tbody id="summary-tbody"></tbody>
      </table>
    </div>
    <div style="margin-top:16px" class="card">
      <div class="tabs" id="subgroup-tabs"></div>
      <div id="subgroup-bars"></div>
    </div>
  </section>

  <section class="grid-2">
    <div class="card">
      <h3>时间、吞吐与 P95 延迟</h3>
      <div class="table-box">
        <table>
          <thead><tr><th>模型 · 评测集</th><th class="num">Wall (s)</th><th class="num">生成/模拟 (s)</th><th class="num">输出 tok/s</th><th class="num">P95 (s)</th></tr></thead>
          <tbody id="perf-tbody"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h3>显卡显存、KV Cache 与排队压力</h3>
      <div class="table-box">
        <table>
          <thead><tr><th>模型 · 评测集</th><th class="num">峰值显存 (MiB)</th><th class="num">GPU 峰值</th><th class="num">KV 峰值</th><th class="num">运行/排队</th><th class="num">抢占</th></tr></thead>
          <tbody id="gpu-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section>
    <h2>3. GPU 显存与 KV Cache 时序曲线</h2>
    <div class="card">
      <div class="tabs" id="resource-tabs"></div>
      <canvas id="resource-canvas"></canvas>
    </div>
  </section>

  <section>
    <h2>4. 错误归因与典型错题（系统错误 vs 真实 LLM 错误）</h2>
    <div class="table-box" style="margin-bottom:14px">
      <table>
        <thead><tr><th>模型 · 评测集</th><th>系统 / 基础设施错误计数</th><th>真实 LLM 能力错误计数</th></tr></thead>
        <tbody id="err-tbody"></tbody>
      </table>
    </div>
    <div class="tabs" id="case-filters">
      <button class="active" data-filter="all">全部典型错题</button>
      <button data-filter="system">仅看系统/服务错误</button>
      <button data-filter="llm">仅看真实 LLM 错误</button>
    </div>
    <div class="case-grid" id="case-grid"></div>
  </section>
</main>
<footer><div class="wrap">ToolCall-SFT · 自动生成基线评测报告</div></footer>
<script id="report-data" type="application/json">__JSON_DATA__</script>
<script>
"use strict";
const payload = JSON.parse(document.getElementById("report-data").textContent);
const runs = payload.runs;
const pct = v => v == null ? "N/A" : (v * 100).toFixed(2) + "%";
const num = (v, d = 1) => v == null ? "N/A" : Number(v).toLocaleString("en-US", {maximumFractionDigits: d, minimumFractionDigits: d});
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

document.getElementById("kpi-cards").innerHTML = runs.map(r => `
  <div class="card">
    <div class="kpi-label">${esc(r.model)} · <code>${esc(r.benchmark)}</code> (${esc(r.profile)})</div>
    <div class="kpi-value">${pct(r.primary_score)}</div>
    <div class="kpi-sub">${esc(r.primary_label)} · ${esc(r.cases_frac)}<br>输出 ${num(r.output_tok_s, 1)} tok/s · KV峰值 ${r.peak_kv_pct == null ? "N/A" : r.peak_kv_pct.toFixed(1) + "%"}</div>
  </div>`).join("");

const badge = s => s === "completed" ? "b-ok" : s === "historical" ? "b-warn" : "b-err";
document.getElementById("summary-tbody").innerHTML = runs.map(r => `
  <tr>
    <td><code>${esc(r.model)}</code></td><td><code>${esc(r.benchmark)}</code></td><td><code>${esc(r.profile)}</code></td>
    <td><span class="badge ${badge(r.status)}">${esc(r.status)}</span></td>
    <td>${esc(r.primary_label)}</td><td class="num">${esc(r.cases_frac)}</td><td class="num"><b>${pct(r.primary_score)}</b></td>
    <td><code>${r.context_window ?? "N/A"} / ${r.max_tokens ?? "N/A"} / ${r.concurrency ?? "N/A"}</code></td>
  </tr>`).join("");

document.getElementById("perf-tbody").innerHTML = runs.map(r => `
  <tr>
    <td><code>${esc(r.model.split("/").pop())} · ${esc(r.benchmark)}</code></td>
    <td class="num">${num(r.wall_seconds, 1)}</td><td class="num">${num(r.stage_seconds, 1)}</td>
    <td class="num">${num(r.output_tok_s, 1)}</td><td class="num">${num(r.latency_p95_s, 2)}</td>
  </tr>`).join("");

document.getElementById("gpu-tbody").innerHTML = runs.map(r => `
  <tr>
    <td><code>${esc(r.model.split("/").pop())} · ${esc(r.benchmark)}</code></td>
    <td class="num">${num(r.peak_gpu_mib, 0)}</td>
    <td class="num">${r.peak_gpu_util_pct == null ? "N/A" : r.peak_gpu_util_pct.toFixed(1) + "%"}</td>
    <td class="num">${r.peak_kv_pct == null ? "N/A" : r.peak_kv_pct.toFixed(2) + "%"}</td>
    <td class="num">${r.peak_running == null ? "N/A" : Math.round(r.peak_running) + " / " + Math.round(r.peak_waiting ?? 0)}</td>
    <td class="num">${r.preemptions == null ? "N/A" : Math.round(r.preemptions)}</td>
  </tr>`).join("");

const fmtMap = obj => Object.keys(obj).length ? Object.entries(obj).map(([k,v]) => `<code>${esc(k)}</code>: <b>${v}</b>`).join(" · ") : "无";
document.getElementById("err-tbody").innerHTML = runs.map(r => `
  <tr><td><code>${esc(r.model.split("/").pop())} · ${esc(r.benchmark)} (${esc(r.profile)})</code></td>
  <td>${fmtMap(r.system_errors)}</td><td>${fmtMap(r.llm_errors)}</td></tr>`).join("");

const subTabs = document.getElementById("subgroup-tabs");
function showSubgroups(idx) {
  [...subTabs.children].forEach((b, i) => b.classList.toggle("active", i === idx));
  const r = runs[idx];
  const entries = Object.entries(r.subgroups);
  document.getElementById("subgroup-bars").innerHTML = entries.map(([k, v]) => {
    const val = v.accuracy ?? 0;
    const frac = v.correct != null && v.total != null ? ` (${v.correct}/${v.total})` : "";
    return `<div class="bar-row"><div class="bar-label" title="${esc(k)}">${esc(k)}${frac}</div><div class="bar-track"><div class="bar-fill" style="width:${(val*100).toFixed(1)}%"></div></div><div class="bar-val">${pct(v.accuracy)}</div></div>`;
  }).join("");
}
subTabs.innerHTML = runs.map((r, i) => `<button>${esc(r.model.split("/").pop())} · ${esc(r.benchmark)} (${esc(r.profile)})</button>`).join("");
[...subTabs.children].forEach((b, i) => b.addEventListener("click", () => showSubgroups(i)));
if (runs.length) showSubgroups(0);

const resTabs = document.getElementById("resource-tabs");
const canvas = document.getElementById("resource-canvas");
let activeResIdx = 0;
function drawResources(idx) {
  activeResIdx = idx;
  [...resTabs.children].forEach((b, i) => b.classList.toggle("active", i === idx));
  const series = runs[idx].resource_series || [];
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(230 * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, 230);
  if (!series.length) {
    ctx.fillStyle = "#536b70"; ctx.font = "13px sans-serif";
    ctx.fillText("该运行未记录时序资源采样", 20, 40); return;
  }
  const pad = {l: 52, r: 52, t: 20, b: 30}, w = rect.width - pad.l - pad.r, h = 230 - pad.t - pad.b;
  const maxT = Math.max(...series.map(p => p.t), 1);
  const maxGpu = Math.max(...series.map(p => p.gpu_mib || 0), 1024);
  ctx.strokeStyle = "#e3ecec"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + h * (i / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(rect.width - pad.r, y); ctx.stroke();
    ctx.fillStyle = "#285ca4"; ctx.font = "11px monospace"; ctx.textAlign = "right";
    ctx.fillText((100 - i * 25) + "%", pad.l - 6, y + 3);
    ctx.fillStyle = "#0f766e"; ctx.textAlign = "left";
    ctx.fillText((maxGpu * (1 - i / 4) / 1024).toFixed(1) + "G", rect.width - pad.r + 6, y + 3);
  }
  const plot = (key, maxY, color) => {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    let started = false;
    for (const p of series) {
      if (p[key] == null) continue;
      const x = pad.l + (p.t / maxT) * w, y = pad.t + h * (1 - p[key] / maxY);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  plot("kv_pct", 100, "#285ca4");
  plot("gpu_mib", maxGpu, "#0f766e");
  ctx.fillStyle = "#536b70"; ctx.textAlign = "center";
  ctx.fillText(`耗时 0 ~ ${maxT.toFixed(1)}s（蓝线: KV Cache % ｜ 绿线: GPU 显存 GiB）`, pad.l + w / 2, 220);
}
resTabs.innerHTML = runs.map((r, i) => `<button>${esc(r.model.split("/").pop())} · ${esc(r.benchmark)}</button>`).join("");
[...resTabs.children].forEach((b, i) => b.addEventListener("click", () => drawResources(i)));
if (runs.length) drawResources(0);
window.addEventListener("resize", () => runs.length && drawResources(activeResIdx));

const allCases = runs.flatMap(r => (r.failure_examples || []).map(ex => ({...ex, run: `${r.model.split("/").pop()} · ${r.benchmark}`})));
function renderCases(filter) {
  const list = filter === "all" ? allCases : allCases.filter(c => c.kind === filter);
  document.getElementById("case-grid").innerHTML = list.length ? list.map(c => `
    <div class="case-card">
      <div class="case-meta">
        <span><code>${esc(c.run)}</code> · <code>${esc(c.id)}</code></span>
        <span class="badge ${c.kind === "system" ? "b-err" : "b-warn"}">${c.kind === "system" ? "系统/服务错误" : "真实 LLM 错误"}</span>
      </div>
      <div class="kpi-sub"><b>原因：</b>${esc(c.reason)}</div>
      <pre><b>输入/上下文：</b>${esc(c.prompt)}\n<b>实际输出：</b>${esc(c.output)}\n<b>期望参考：</b>${esc(c.expected)}</pre>
    </div>`).join("") : '<div class="card">当前过滤条件下无失败样例。</div>';
}
document.querySelectorAll("#case-filters button").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll("#case-filters button").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  renderCases(btn.dataset.filter);
}));
renderCases("all");
</script>
</body>
</html>
"""


def render_html(records: list[dict], title: str) -> str:
    """Render a standalone interactive HTML report with charts and failure cases."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data_json = json.dumps({"runs": records}, ensure_ascii=False).replace("</", "<\\/")
    return (
        HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__NOW__", html.escape(now))
        .replace("__JSON_DATA__", data_json)
    )


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs", nargs="+", type=Path, help="Run directories to include")
    group.add_argument(
        "--auto",
        action="store_true",
        help="Auto-select latest run per (benchmark, model) under runs/eval",
    )
    cli.add_argument("--eval-root", type=Path, default=Path("runs/eval"))
    cli.add_argument("--title", default="ToolCall-SFT 基线评测报告")
    cli.add_argument("--out-md", type=Path, default=Path("reports/baseline-report.md"))
    cli.add_argument("--out-html", type=Path, default=Path("reports/baseline-report.html"))
    args = cli.parse_args()

    run_dirs = (
        discover_runs(args.eval_root.resolve())
        if args.auto
        else [p.resolve() for p in args.runs]
    )
    if not run_dirs:
        cli.exit(1, "No evaluation run directories found.\n")
    records = [extract_run_record(run_dir) for run_dir in run_dirs]

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(records, args.title), encoding="utf-8")
    args.out_html.write_text(render_html(records, args.title), encoding="utf-8")
    print(f"Markdown report: {args.out_md}")
    print(f"HTML report:     {args.out_html}")


if __name__ == "__main__":
    main()
