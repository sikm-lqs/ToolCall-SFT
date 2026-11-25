"""ToolRL-aligned GRPO reward functions with 3.4-B symmetric non-call scoring and BFCL-aligned value matching."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import itertools
import json
import math
import re
from typing import Any

from ..data.parsers import _NL_PSEUDO_CALL_RE

_THINK_BLOCK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>")
_TOOL_CALL_CLOSE_RE = re.compile(r"</tool_call>")
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_BLOCK_RE = re.compile(r"^<function=([^\s>]+)>\s*(.*?)\s*</function>$", re.DOTALL)
_PARAM_BLOCK_RE = re.compile(r"<parameter=([^\s>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


@dataclass
class ParsedCompletion:
    """Structured parse output for one Qwen3.5 completion."""

    tool_calls: list[dict[str, Any]]
    text_outside_calls: str
    has_call_intent: bool
    format_valid: bool
    has_leak: bool


def _coerce_param_value(raw_val: str, schema_type: str | None) -> Any:
    text = raw_val
    stripped = text.strip()
    if schema_type == "string":
        # Preserve raw string unless template wrapped JSON quotes
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, str):
                    return parsed
            except Exception:
                pass
        return text
    if schema_type == "integer":
        try:
            if stripped.startswith(("+", "-")) and stripped[1:].isdigit() or stripped.isdigit():
                return int(stripped)
            val = json.loads(stripped)
            if isinstance(val, int) and not isinstance(val, bool):
                return val
        except Exception:
            pass
    elif schema_type == "number":
        try:
            val = json.loads(stripped)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return val
        except Exception:
            try:
                return float(stripped)
            except Exception:
                pass
    elif schema_type == "boolean":
        low = stripped.lower()
        if low == "true":
            return True
        if low == "false":
            return False
    elif schema_type in ("array", "object"):
        try:
            val = json.loads(stripped)
            if (schema_type == "array" and isinstance(val, list)) or (
                schema_type == "object" and isinstance(val, dict)
            ):
                return val
        except Exception:
            pass

    # Fallback heuristic when schema type is absent or union
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() == "null":
        return None
    if stripped and (stripped[0] in "{[" or stripped.lstrip("-").isdigit()):
        try:
            return json.loads(stripped)
        except Exception:
            pass
    try:
        if re.fullmatch(r"-?\d+\.\d+", stripped):
            return float(stripped)
    except Exception:
        pass
    return text


def _build_schema_index(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for t in tools or []:
        fn = t.get("function") or {}
        name = str(fn.get("name") or "")
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        req = set(params.get("required") or [])
        idx[name] = {"properties": props, "required": req}
    return idx


def parse_qwen_completion(
    completion_text: str, tools: list[dict[str, Any]] | None = None
) -> ParsedCompletion:
    """Parse Qwen3.5 XML tool calls and validate format/leakage."""
    raw = str(completion_text or "")
    body = _THINK_BLOCK_RE.sub("", raw, count=1)

    n_open = len(_TOOL_CALL_OPEN_RE.findall(body))
    n_close = len(_TOOL_CALL_CLOSE_RE.findall(body))
    has_call_intent = bool(n_open or n_close or "<function=" in body or "<parameter=" in body)

    if not has_call_intent:
        leaked = bool(_NL_PSEUDO_CALL_RE.search(body))
        non_empty = bool(body.strip())
        return ParsedCompletion(
            tool_calls=[],
            text_outside_calls=body.strip(),
            has_call_intent=False,
            format_valid=(non_empty and not leaked),
            has_leak=leaked,
        )

    schema_idx = _build_schema_index(tools)
    outside = _TOOL_CALL_BLOCK_RE.sub("", body).strip()
    leaked_outside = bool(_NL_PSEUDO_CALL_RE.search(outside) or "<parameter=" in outside)

    if n_open == 0 or n_open != n_close or leaked_outside:
        return ParsedCompletion(
            tool_calls=[],
            text_outside_calls=outside,
            has_call_intent=True,
            format_valid=False,
            has_leak=True,
        )

    blocks = _TOOL_CALL_BLOCK_RE.findall(body)
    if len(blocks) != n_open:
        return ParsedCompletion(
            tool_calls=[],
            text_outside_calls=outside,
            has_call_intent=True,
            format_valid=False,
            has_leak=True,
        )

    calls: list[dict[str, Any]] = []
    for block in blocks:
        fn_match = _FUNCTION_BLOCK_RE.match(block.strip())
        if not fn_match:
            return ParsedCompletion(
                tool_calls=[],
                text_outside_calls=outside,
                has_call_intent=True,
                format_valid=False,
                has_leak=False,
            )
        fn_name = fn_match.group(1).strip()
        fn_body = fn_match.group(2)

        # Ensure nothing malformed remains outside <parameter=...>...</parameter> blocks
        leftover = _PARAM_BLOCK_RE.sub("", fn_body).strip()
        if leftover:
            return ParsedCompletion(
                tool_calls=[],
                text_outside_calls=outside,
                has_call_intent=True,
                format_valid=False,
                has_leak=False,
            )

        props = schema_idx.get(fn_name, {}).get("properties", {})
        args: dict[str, Any] = {}
        for p_key, p_raw in _PARAM_BLOCK_RE.findall(fn_body):
            key = p_key.strip()
            if not key or key in args:
                return ParsedCompletion(
                    tool_calls=[],
                    text_outside_calls=outside,
                    has_call_intent=True,
                    format_valid=False,
                    has_leak=False,
                )
            p_type = (props.get(key) or {}).get("type")
            args[key] = _coerce_param_value(p_raw, p_type)
        calls.append({"name": fn_name, "arguments": args})

    return ParsedCompletion(
        tool_calls=calls,
        text_outside_calls=outside,
        has_call_intent=True,
        format_valid=len(calls) > 0,
        has_leak=False,
    )


def compute_format_reward(parsed: ParsedCompletion) -> float:
    """Return 1.0 if structural format is valid (without tag leakage), else 0.0."""
    return 1.0 if parsed.format_valid else 0.0


def _norm_str_bfcl(s: str) -> str:
    # Align with BFCL string standardization: collapse whitespace and strip edge whitespace
    return " ".join(s.strip().split())


def bfcl_value_equal(pred: Any, gold: Any) -> bool:
    """Check value equivalence aligned with BFCL relaxed AST type/format rules."""
    if isinstance(gold, bool) or isinstance(pred, bool):
        return isinstance(pred, bool) and isinstance(gold, bool) and pred is gold

    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        if math.isnan(gold) or math.isnan(pred):
            return False
        return abs(float(pred) - float(gold)) <= 1e-6

    if isinstance(gold, str) and isinstance(pred, str):
        p_norm = _norm_str_bfcl(pred)
        g_norm = _norm_str_bfcl(gold)
        if p_norm == g_norm:
            return True
        if p_norm.casefold() == g_norm.casefold():
            return True
        # Allow minor trailing punctuation differences on natural text args
        if p_norm.rstrip(".,;").casefold() == g_norm.rstrip(".,;").casefold():
            return True
        return False

    if isinstance(gold, list) and isinstance(pred, list):
        if len(gold) != len(pred):
            return False
        return all(bfcl_value_equal(p_el, g_el) for p_el, g_el in zip(pred, gold, strict=True))

    if isinstance(gold, dict) and isinstance(pred, dict):
        if set(gold.keys()) != set(pred.keys()):
            return False
        return all(bfcl_value_equal(pred[k], gold[k]) for k in gold)

    return pred == gold


def _pair_key_and_value_score(
    pred_args: dict[str, Any],
    gold_args: dict[str, Any],
    tool_schema: dict[str, Any],
) -> tuple[float, float]:
    required = tool_schema.get("required") or set()
    # Align optional default parameters: if a gold key is optional and empty/None and omitted by pred,
    # do not penalize missing it in key Jaccard.
    eff_gold_keys = {
        k
        for k, v in gold_args.items()
        if not (k not in pred_args and k not in required and v in ("", None))
    }
    pred_keys = set(pred_args.keys())

    if not eff_gold_keys and not pred_keys:
        key_score = 1.0
    else:
        key_score = len(eff_gold_keys & pred_keys) / len(eff_gold_keys | pred_keys)

    val_matches = 0.0
    for k, g_val in gold_args.items():
        if k in pred_args:
            if bfcl_value_equal(pred_args[k], g_val):
                val_matches += 1.0
        elif k not in required and g_val in ("", None):
            val_matches += 1.0

    return key_score, val_matches


def compute_call_correct_reward(
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> float:
    """ToolRL correctness reward in [-3, 3] for call tasks."""
    if not gold_calls:
        raise ValueError("compute_call_correct_reward requires non-empty gold_calls")
    if not pred_calls:
        return -3.0

    schema_idx = _build_schema_index(tools)

    # 1. Multiset Jaccard for tool names
    g_names = Counter(str(g["name"]) for g in gold_calls)
    p_names = Counter(str(p["name"]) for p in pred_calls)
    inter = sum((g_names & p_names).values())
    union = sum((g_names | p_names).values())
    r_name = inter / union if union > 0 else 0.0

    # 2. Optimal 1-to-1 matching M for same-name calls
    g_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    p_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in gold_calls:
        g_by_name[str(g["name"])].append(g)
    for p in pred_calls:
        p_by_name[str(p["name"])].append(p)

    r_key = 0.0
    r_value = 0.0

    for name, g_list in g_by_name.items():
        p_list = p_by_name.get(name, [])
        if not p_list:
            continue
        t_schema = schema_idx.get(name, {"properties": {}, "required": set()})
        a, b = len(g_list), len(p_list)
        k_match = min(a, b)

        if a == 1 and b == 1:
            ks, vm = _pair_key_and_value_score(
                p_list[0].get("arguments") or {}, g_list[0].get("arguments") or {}, t_schema
            )
            r_key += ks
            r_value += vm
        elif max(a, b) <= 6:
            # Exact optimal assignment over same-name calls
            best_total = -1.0
            best_kv = (0.0, 0.0)
            if a <= b:
                for perm in itertools.permutations(range(b), k_match):
                    tk, tv = 0.0, 0.0
                    for gi, pi in enumerate(perm):
                        ks, vm = _pair_key_and_value_score(
                            p_list[pi].get("arguments") or {}, g_list[gi].get("arguments") or {}, t_schema
                        )
                        tk += ks
                        tv += vm
                    if tk + tv > best_total:
                        best_total = tk + tv
                        best_kv = (tk, tv)
            else:
                for perm in itertools.permutations(range(a), k_match):
                    tk, tv = 0.0, 0.0
                    for pi, gi in enumerate(perm):
                        ks, vm = _pair_key_and_value_score(
                            p_list[pi].get("arguments") or {}, g_list[gi].get("arguments") or {}, t_schema
                        )
                        tk += ks
                        tv += vm
                    if tk + tv > best_total:
                        best_total = tk + tv
                        best_kv = (tk, tv)
            r_key += best_kv[0]
            r_value += best_kv[1]
        else:
            # Greedy fallback for large identical-tool fanouts
            candidates = []
            for gi, g_item in enumerate(g_list):
                for pi, p_item in enumerate(p_list):
                    ks, vm = _pair_key_and_value_score(
                        p_item.get("arguments") or {}, g_item.get("arguments") or {}, t_schema
                    )
                    candidates.append((ks + vm, ks, vm, gi, pi))
            candidates.sort(reverse=True)
            used_g, used_p = set(), set()
            for _, ks, vm, gi, pi in candidates:
                if gi in used_g or pi in used_p:
                    continue
                used_g.add(gi)
                used_p.add(pi)
                r_key += ks
                r_value += vm

    s_max = 1 + len(gold_calls) + sum(len(g.get("arguments") or {}) for g in gold_calls)
    r_correct = 6.0 * ((r_name + r_key + r_value) / s_max) - 3.0
    return max(-3.0, min(3.0, r_correct))


def _extract_fact_tokens(text: str) -> list[str]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text)
    seen: list[str] = []
    for tok in nums + words:
        low = tok.casefold()
        if low not in seen and low not in {"true", "false", "null", "none", "status", "result", "error", "data"}:
            seen.append(low)
    return seen


def compute_non_call_correct_reward(
    parsed: ParsedCompletion,
    *,
    bucket: str,
    ground_truth: dict[str, Any],
) -> float:
    """3.4-B symmetric Non-Call correctness reward in [-3, 3]."""
    if parsed.has_call_intent or parsed.has_leak:
        return -3.0

    text = parsed.text_outside_calls.strip()
    if not text:
        return -3.0

    if bucket == "missing_param":
        missing = [str(p).lower() for p in (ground_truth.get("missing_params") or []) if p]
        low = text.lower()
        if missing and any(p in low or p.replace("_", " ") in low for p in missing):
            return 3.0
        if any(kw in low for kw in ("provide", "specify", "missing", "need", "required", "parameter")):
            return 2.5 if not missing else 2.0
        return 1.5

    if bucket == "multi_turn_post_tool_response":
        obs = str(ground_truth.get("tool_observation") or "")
        facts = _extract_fact_tokens(obs)
        low = text.casefold()
        if facts and any(f in low for f in facts[:12]):
            return 3.0
        if len(text) >= 20:
            return 2.0
        return 1.5

    # irrelevance_refusal
    return 3.0 if len(text) >= 12 else 2.5


def score_completion(
    completion_text: str,
    *,
    task_type: str,
    bucket: str,
    ground_truth: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, float | bool]:
    """Return full reward breakdown for one completion."""
    parsed = parse_qwen_completion(completion_text, tools=tools)
    r_format = compute_format_reward(parsed)

    if task_type == "call":
        gold_calls = ground_truth.get("tool_calls") or []
        r_correct = compute_call_correct_reward(parsed.tool_calls, gold_calls, tools=tools)
    else:
        r_correct = compute_non_call_correct_reward(
            parsed, bucket=bucket, ground_truth=ground_truth
        )

    return {
        "r_final": r_format + r_correct,
        "r_format": r_format,
        "r_correct": r_correct,
        "has_leak": parsed.has_leak,
        "has_call_intent": parsed.has_call_intent,
        "num_pred_calls": len(parsed.tool_calls),
    }


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
        return str(last)
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    return str(completion)


def grpo_reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    """TRL GRPOTrainer-compatible reward function returning scalar rewards in [-3, 4]."""
    task_types = kwargs["task_type"]
    buckets = kwargs["bucket"]
    raw_gts = kwargs.get("ground_truth") or kwargs["ground_truth_json"]
    raw_tools = kwargs.get("tools") or kwargs.get("tools_json") or [None] * len(completions)
    log_metric = kwargs.get("log_metric")

    finals: list[float] = []
    formats: list[float] = []
    corrects: list[float] = []
    call_corrects: list[float] = []
    non_call_corrects: list[float] = []
    leaks = 0

    for comp, ttype, bkt, raw_gt, raw_tls in zip(
        completions, task_types, buckets, raw_gts, raw_tools, strict=True
    ):
        gt = json.loads(raw_gt) if isinstance(raw_gt, str) else raw_gt
        tls = json.loads(raw_tls) if isinstance(raw_tls, str) else raw_tls
        text = _extract_completion_text(comp)
        res = score_completion(
            text, task_type=str(ttype), bucket=str(bkt), ground_truth=gt, tools=tls
        )
        finals.append(float(res["r_final"]))
        formats.append(float(res["r_format"]))
        corrects.append(float(res["r_correct"]))
        if ttype == "call":
            call_corrects.append(float(res["r_correct"]))
        else:
            non_call_corrects.append(float(res["r_correct"]))
        leaks += int(bool(res["has_leak"]))

    if callable(log_metric) and finals:
        log_metric("reward/format_mean", sum(formats) / len(formats))
        log_metric("reward/correct_mean", sum(corrects) / len(corrects))
        log_metric("reward/leak_rate", leaks / len(finals))
        if call_corrects:
            log_metric("reward/call_correct_mean", sum(call_corrects) / len(call_corrects))
        if non_call_corrects:
            log_metric("reward/non_call_correct_mean", sum(non_call_corrects) / len(non_call_corrects))

    return finals
