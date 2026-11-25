"""Raw dataset loaders and cleaners for xLAM-60k and ToolACE."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import pyarrow.parquet as pq

from .schema import (
    build_tool_name_mapping,
    normalize_toolace_parameters,
    normalize_xlam_parameters,
    validate_record,
)

TOOLACE_TOOLS_MARKER = "Here is a list of functions in JSON format that you can invoke:"
_TOOLACE_RETURN_ONLY_RE = re.compile(
    r"\s*You should only return the function call in tools call sections\.",
    re.IGNORECASE,
)
_TOOLACE_NOTE_RE = re.compile(r"(Note that the provided function is in [^\n.]+\.)")
_TOOLACE_POLICY_RE = re.compile(
    r"You are an expert in composing functions\.\s*"
    r"You are given a question and a set of possible functions\.\s*"
    r"Based on the question, you will need to make one or more function/tool calls to achieve the purpose\.\s*"
    r"(?:If none of the function can be used, point it out\.\s*)?"
    r"(?:If the given question lacks the parameters required by the function, also point it out\.\s*)?",
    re.IGNORECASE,
)
_NO_CALL_SUSPECT_RE = re.compile(
    r"(\[No Need to Ask\]|```(?:json)?\s*[\[\{]|\"name\"\s*:\s*\"|<tool_call>|<function=|"
    r"sufficient information to call|all the necessary information to call|"
    r"you have provided all the required)",
    re.IGNORECASE,
)
_NL_PSEUDO_CALL_RE = re.compile(
    r"(<tool_call>|</tool_call>|<function=|\"id\"\s*:\s*\"call_|\"arguments\"\s*:\s*\"?\{)",
    re.IGNORECASE,
)
_NO_CALL_IRRELEVANT_RE = re.compile(
    r"\b(?:do(?:es)? not|don't|doesn't)\s+(?:include|provide|offer|support|have|contain)\b"
    r"|\bnone of the (?:provided |available |listed |given )?(?:function|tool|api)s?\b"
    r"|\bno (?:suitable |available |relevant |matching |specific )?(?:function|tool|api)s?\s+(?:is|are|can|to|for)\b"
    r"|\bno function(?:s)? (?:is |are )?(?:available|provided|listed|relevant|capable)\b"
    r"|\bdifferent set of (?:functions|tools|apis)\b"
    r"|\bnot (?:achievable|supported|possible|capable) with (?:the )?(?:provided|available|given)\b",
    re.IGNORECASE,
)
_STRICT_IRR_RE = re.compile(
    r"\b(?:do(?:es)? not|don't|doesn't)\s+(?:include|provide\s+(?:any|a|the)?\s*(?:function|tool|api|capabilit|method|way|mechanism|option)|offer|support|have\s+(?:any|a)?\s*(?:function|tool|api|capabilit)|contain)\b"
    r"|\bnone of the (?:provided |available |listed |given )?(?:function|tool|api)s?\b"
    r"|\bno (?:suitable |available |relevant |matching |specific )?(?:function|tool|api)s?\s+(?:is|are|can|to|for)\b"
    r"|\bno function(?:s)? (?:is |are )?(?:available|provided|listed|relevant|capable)\b"
    r"|\bdifferent set of (?:functions|tools|apis)\b"
    r"|\bnot (?:achievable|supported|possible|capable) with (?:the )?(?:provided|available|given)\b"
    r"|\brelated to [^.\n]{1,60},\s*not\b"
    r"|\bdesigned (?:only |specifically )?(?:to|for) [^.\n]{1,60},?\s*(?:not|rather than)\b",
    re.IGNORECASE,
)
_NO_CALL_MISSING_PARAM_RE = re.compile(
    r"\b(lack(?:s|ing)?|missing|required (?:parameter|argument|field)|please provide|"
    r"need(?:s)? to (?:provide|specify)|must specify|"
    r"not provide(?:d)? (?:the |enough )?(?:parameter|argument|value|info))\b",
    re.IGNORECASE,
)
_STRICT_MP_RE = re.compile(
    r"\b(?:requires?(?: the)? (?:required )?(?:parameter|argument|field)|"
    r"lacks? the (?:required |necessary )?(?:parameter|argument|field)|"
    r"missing (?:the )?(?:required |necessary )?(?:parameter|argument|field)|"
    r"required (?:parameter|argument|field)s? (?:for|of|are|is)|"
    r"please provide (?:the |a )?(?:specific |valid |required )?(?:[a-z_]+|value|parameter|argument)|"
    r"need(?:s)? to (?:provide|specify))\b",
    re.IGNORECASE,
)


@dataclass
class ParseStats:
    """Counters tracking raw rows, kept rows, and rejection reasons."""

    raw_rows: int = 0
    kept_rows: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    buckets: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_rows": self.raw_rows,
            "kept_rows": self.kept_rows,
            "dropped": dict(sorted(self.dropped.items())),
            "buckets": dict(sorted(self.buckets.items())),
        }


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False

    for ch in text:
        if quote:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
        else:
            if ch in ("'", '"'):
                quote = ch
                buf.append(ch)
            elif ch in "([{":
                depth += 1
                buf.append(ch)
            elif ch in ")]}":
                depth -= 1
                buf.append(ch)
            elif ch == sep and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _eval_arg_literal(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except Exception:
        return json.loads(raw)


def parse_toolace_bracket_calls(
    text: str, known_names: list[str]
) -> list[dict[str, Any]]:
    """Parse ToolACE `[func1(k=v, ...), func2(...)]` syntax cleanly, including names with spaces/parens."""
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError("not_bracket_call")
    inner = stripped[1:-1].strip()
    if not inner:
        return []

    sorted_names = sorted(known_names, key=len, reverse=True)
    calls: list[dict[str, Any]] = []
    idx = 0
    length = len(inner)

    while idx < length:
        while idx < length and (inner[idx].isspace() or inner[idx] == ","):
            idx += 1
        if idx >= length:
            break

        matched_name: str | None = None
        paren_idx = -1
        for candidate in sorted_names:
            if inner.startswith(candidate, idx):
                after = idx + len(candidate)
                while after < length and inner[after].isspace():
                    after += 1
                if after < length and inner[after] == "(":
                    matched_name = candidate
                    paren_idx = after
                    break

        if matched_name is None:
            paren_idx = inner.find("(", idx)
            if paren_idx == -1:
                raise ValueError(f"missing_paren:{inner[idx : idx + 40]}")
            matched_name = inner[idx:paren_idx].strip()

        depth = 0
        quote: str | None = None
        escaped = False
        end_idx = paren_idx
        while end_idx < length:
            ch = inner[end_idx]
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
            else:
                if ch in ("'", '"'):
                    quote = ch
                elif ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                    if depth == 0 and ch == ")":
                        break
            end_idx += 1

        if end_idx >= length or inner[end_idx] != ")":
            raise ValueError(f"unclosed_call:{matched_name}")

        args_str = inner[paren_idx + 1 : end_idx].strip()
        kwargs: dict[str, Any] = {}
        if args_str:
            for item in _split_top_level(args_str, ","):
                eq = item.find("=")
                if eq == -1:
                    raise ValueError(f"non_kwarg:{item}")
                key = item[:eq].strip()
                val = _eval_arg_literal(item[eq + 1 :].strip())
                kwargs[key] = val

        calls.append({"name": matched_name, "arguments": kwargs})
        idx = end_idx + 1

    return calls


def _extract_toolace_system_and_tools(
    raw_system: str,
) -> tuple[str, list[dict[str, Any]]]:
    marker_idx = raw_system.find(TOOLACE_TOOLS_MARKER)
    if marker_idx == -1:
        raise ValueError("non_json_system_format")

    prefix = raw_system[:marker_idx]
    json_start = marker_idx + len(TOOLACE_TOOLS_MARKER)
    while json_start < len(raw_system) and raw_system[json_start].isspace():
        json_start += 1

    raw_tools, json_end = json.JSONDecoder().raw_decode(raw_system, json_start)
    if not isinstance(raw_tools, list):
        raise ValueError("tools_not_list")

    tail = raw_system[json_end:]
    clean_prefix = _TOOLACE_RETURN_ONLY_RE.sub("", prefix)
    clean_prefix = re.sub(r"\s+", " ", clean_prefix).strip()

    note_match = _TOOLACE_NOTE_RE.search(tail)
    if note_match:
        clean_prefix = f"{clean_prefix}\n{note_match.group(1)}".strip()

    return clean_prefix, raw_tools


def load_xlam_60k(
    path: Path,
    *,
    strict_validation: bool = False,
) -> tuple[list[dict[str, Any]], ParseStats]:
    """Load, deserialize, and normalize `Salesforce/xlam-function-calling-60k`."""
    stats = ParseStats()
    records: list[dict[str, Any]] = []

    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    stats.raw_rows = len(raw_rows)

    for row in raw_rows:
        try:
            raw_tools = json.loads(row["tools"])
            raw_answers = json.loads(row["answers"])
            query = str(row["query"]).strip()
        except Exception:
            stats.dropped["json_decode_error"] += 1
            continue

        if not query or not isinstance(raw_tools, list) or not isinstance(raw_answers, list) or not raw_answers:
            stats.dropped["empty_query_or_answers"] += 1
            continue

        raw_names = [str(t.get("name", "")).strip() for t in raw_tools]
        if any(not n for n in raw_names):
            stats.dropped["empty_tool_name"] += 1
            continue
        name_map = build_tool_name_mapping(raw_names)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": name_map[str(t["name"]).strip()],
                    "description": str(t.get("description") or "").strip(),
                    "parameters": normalize_xlam_parameters(t.get("parameters") or {}),
                },
            }
            for t in raw_tools
        ]

        tool_calls = []
        bad_call = False
        for ans in raw_answers:
            raw_name = str(ans.get("name", "")).strip()
            args = ans.get("arguments")
            if raw_name not in name_map or not isinstance(args, dict):
                bad_call = True
                break
            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": name_map[raw_name],
                        "arguments": args,
                    },
                }
            )
        if bad_call:
            stats.dropped["invalid_answer_call"] += 1
            continue

        is_parallel = len(tool_calls) > 1
        bucket = "xlam_parallel" if is_parallel else "xlam_single"
        record = {
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": "", "tool_calls": tool_calls},
            ],
            "tools": tools,
            "meta": {
                "source": "xlam-60k",
                "source_id": row.get("id"),
                "bucket": bucket,
                "num_turns": 1,
                "num_user_turns": 1,
                "num_tool_messages": 0,
                "num_tool_calls": len(tool_calls),
                "num_candidate_tools": len(tools),
                "has_tool_response": False,
                "is_parallel": is_parallel,
                "is_refusal": False,
            },
        }
        ok, reason = validate_record(
            record,
            strict_arg_types=strict_validation,
            reject_duplicate_tools=strict_validation,
        )
        if not ok:
            stats.dropped[reason] += 1
            continue

        stats.kept_rows += 1
        stats.buckets[bucket] += 1
        records.append(record)

    return records, stats


def strip_toolace_policy_prompt(sys_prompt: str) -> str:
    """Remove ToolACE's generic policy boilerplate while keeping real date/language constraints."""
    cleaned = _TOOLACE_POLICY_RE.sub("", sys_prompt)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
    if cleaned.endswith(".."):
        cleaned = cleaned[:-1]
    return cleaned


def classify_toolace_no_call(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    strict_boundary: bool = False,
) -> tuple[str | None, str | None]:
    """Classify a zero-call ToolACE record into clean sub-buckets or quarantine reasons."""
    if not tools:
        return None, "no_call_empty_tools"

    asst_text = "\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") == "assistant"
    )
    if _NO_CALL_SUSPECT_RE.search(asst_text):
        return None, "no_call_suspect_leak"

    names = [re.escape(t["function"]["name"]) for t in tools if t.get("function", {}).get("name")]
    if names and re.search(r"\[\s*(?:" + "|".join(names) + r")\s*\(", asst_text):
        return None, "no_call_suspect_leak"

    if strict_boundary:
        raw_names = [t["function"]["name"] for t in tools if t.get("function", {}).get("name")]
        req_params = {
            p
            for t in tools
            for p in ((t.get("function", {}).get("parameters") or {}).get("required") or [])
        }
        mentions_tool = any(
            n.lower() in asst_text.lower() or n.replace("_", " ").lower() in asst_text.lower()
            for n in raw_names
        )
        mentions_req_param = any(
            re.search(r"\b" + re.escape(p) + r"\b", asst_text)
            for p in req_params
            if len(p) >= 3
        )
        has_irr = bool(_STRICT_IRR_RE.search(asst_text))
        has_mp = bool(_STRICT_MP_RE.search(asst_text))
        if has_irr and not has_mp and not mentions_req_param:
            return "toolace_irrelevant_refusal", None
        if has_mp and (mentions_tool or mentions_req_param) and not has_irr:
            return "toolace_missing_param", None
        if has_irr and (has_mp or mentions_req_param):
            return None, "no_call_boundary_conflict"
        return None, "no_call_ambiguous"

    if _NO_CALL_IRRELEVANT_RE.search(asst_text):
        return "toolace_irrelevant_refusal", None
    if _NO_CALL_MISSING_PARAM_RE.search(asst_text):
        return "toolace_missing_param", None
    return None, "no_call_ambiguous"


def load_toolace(
    path: Path,
    *,
    strip_policy_prompt: bool = False,
    split_no_call_subtypes: bool = False,
    strict_validation: bool = False,
) -> tuple[list[dict[str, Any]], ParseStats]:
    """Load, deserialize, and normalize `mlabonne/ToolACE` parquet."""
    stats = ParseStats()
    records: list[dict[str, Any]] = []

    table = pq.ParquetFile(path).read().to_pylist()
    stats.raw_rows = len(table)

    for idx, row in enumerate(table):
        conv = row.get("conversations") or []
        if len(conv) < 3 or conv[0].get("from") != "system":
            stats.dropped["invalid_conversation_header"] += 1
            continue

        try:
            sys_prompt, raw_tools = _extract_toolace_system_and_tools(conv[0]["value"])
        except Exception as exc:
            stats.dropped[str(exc) if "non_json" in str(exc) else "system_parse_error"] += 1
            continue

        if strip_policy_prompt and sys_prompt:
            sys_prompt = strip_toolace_policy_prompt(sys_prompt)

        raw_names = [str(t.get("name", "")).strip() for t in raw_tools]
        if any(not n for n in raw_names):
            stats.dropped["empty_tool_name"] += 1
            continue
        name_map = build_tool_name_mapping(raw_names)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": name_map[str(t["name"]).strip()],
                    "description": str(t.get("description") or "").strip(),
                    "parameters": normalize_toolace_parameters(t.get("parameters")),
                },
            }
            for t in raw_tools
        ]

        messages: list[dict[str, Any]] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        total_tool_calls = 0
        has_parallel = False
        has_tool_response = False
        num_tool_messages = 0
        parse_failed = False

        for turn in conv[1:]:
            role = turn.get("from")
            value = str(turn.get("value") or "")
            if role == "user":
                messages.append({"role": "user", "content": value.strip()})
            elif role == "assistant":
                stripped = value.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    try:
                        parsed_calls = parse_toolace_bracket_calls(stripped, raw_names)
                    except Exception:
                        parse_failed = True
                        break
                    if not parsed_calls:
                        parse_failed = True
                        break
                    normalized_calls = []
                    for call in parsed_calls:
                        if call["name"] not in name_map:
                            parse_failed = True
                            break
                        normalized_calls.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": name_map[call["name"]],
                                    "arguments": call["arguments"],
                                },
                            }
                        )
                    if parse_failed:
                        break
                    total_tool_calls += len(normalized_calls)
                    if len(normalized_calls) > 1:
                        has_parallel = True
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": normalized_calls,
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": stripped})
            elif role == "tool":
                try:
                    obs_list = json.loads(value)
                    assert isinstance(obs_list, list) and obs_list
                except Exception:
                    parse_failed = True
                    break
                has_tool_response = True
                num_tool_messages += 1
                for item in obs_list:
                    if not isinstance(item, dict) or "name" not in item or "results" not in item:
                        parse_failed = True
                        break
                    obs_name = str(item["name"]).strip()
                    norm_obs_name = name_map.get(obs_name, obs_name)
                    res = item["results"]
                    obs_content = (
                        res
                        if isinstance(res, str)
                        else json.dumps(res, ensure_ascii=False)
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "name": norm_obs_name,
                            "content": obs_content,
                        }
                    )
                if parse_failed:
                    break
            else:
                parse_failed = True
                break

        if parse_failed:
            stats.dropped["turn_parse_error"] += 1
            continue

        if (split_no_call_subtypes or strict_validation) and total_tool_calls > 0:
            asst_nl = "\n".join(
                str(m.get("content") or "")
                for m in messages
                if m.get("role") == "assistant" and not m.get("tool_calls")
            )
            if asst_nl and (
                _NO_CALL_SUSPECT_RE.search(asst_nl)
                or (strict_validation and _NL_PSEUDO_CALL_RE.search(asst_nl))
            ):
                stats.dropped["suspect_assistant_text"] += 1
                continue

        is_refusal = total_tool_calls == 0
        if has_tool_response:
            bucket = "toolace_multi_turn"
        elif is_refusal:
            if split_no_call_subtypes:
                sub_bucket, drop_reason = classify_toolace_no_call(
                    messages, tools, strict_boundary=strict_validation
                )
                if drop_reason is not None or sub_bucket is None:
                    stats.dropped[drop_reason or "no_call_quarantined"] += 1
                    continue
                bucket = sub_bucket
            else:
                bucket = "toolace_no_call"
        else:
            bucket = "toolace_single_or_parallel"

        num_user_turns = sum(1 for m in messages if m["role"] == "user")
        record = {
            "messages": messages,
            "tools": tools,
            "meta": {
                "source": "toolace",
                "source_id": idx,
                "bucket": bucket,
                "num_turns": num_user_turns,
                "num_user_turns": num_user_turns,
                "num_tool_messages": num_tool_messages,
                "num_tool_calls": total_tool_calls,
                "num_candidate_tools": len(tools),
                "has_tool_response": has_tool_response,
                "is_parallel": has_parallel,
                "is_refusal": is_refusal,
            },
        }
        ok, reason = validate_record(
            record,
            strict_arg_types=strict_validation,
            reject_duplicate_tools=strict_validation,
        )
        if not ok:
            stats.dropped[reason] += 1
            continue

        stats.kept_rows += 1
        stats.buckets[bucket] += 1
        records.append(record)

    return records, stats


def load_apigen_mt_5k(
    path: Path,
    *,
    excluded_first_turns: set[str] | None = None,
    excluded_entities: set[str] | None = None,
    require_final_nl: bool = True,
    strict_validation: bool = False,
) -> tuple[list[dict[str, Any]], ParseStats]:
    """Load, filter, and normalize `Salesforce/APIGen-MT-5k` multi-turn trajectories."""
    stats = ParseStats()
    records: list[dict[str, Any]] = []

    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    stats.raw_rows = len(raw_rows)

    for idx, row in enumerate(raw_rows):
        conv = row.get("conversations") or []
        sys_prompt = str(row.get("system") or "").strip()
        try:
            raw_tools = (
                json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
            )
            assert isinstance(raw_tools, list)
        except Exception:
            stats.dropped["tools_json_error"] += 1
            continue

        # Drop trajectories calling pseudo-tool `think` to prevent hallucinated tool calls.
        if any(
            m.get("from") == "function_call" and '"think"' in str(m.get("value") or "")
            for m in conv
        ):
            has_think_call = False
            for m in conv:
                if m.get("from") == "function_call":
                    try:
                        if json.loads(m["value"]).get("name") == "think":
                            has_think_call = True
                            break
                    except Exception:
                        pass
            if has_think_call:
                stats.dropped["has_think_call"] += 1
                continue

        filtered_tools = [t for t in raw_tools if str(t.get("name", "")).strip() != "think"]
        raw_names = [str(t.get("name", "")).strip() for t in filtered_tools]
        if not raw_names or any(not n for n in raw_names):
            stats.dropped["empty_tool_name"] += 1
            continue
        name_map = build_tool_name_mapping(raw_names)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": name_map[str(t["name"]).strip()],
                    "description": str(t.get("description") or "").strip(),
                    "parameters": normalize_toolace_parameters(t.get("parameters")),
                },
            }
            for t in filtered_tools
        ]

        messages: list[dict[str, Any]] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        total_tool_calls = 0
        num_tool_messages = 0
        has_tool_response = False
        parse_failed: str | None = None
        prev_from: str | None = None

        for turn in conv:
            role = turn.get("from")
            value = str(turn.get("value") or "")
            if role == "human":
                if prev_from == "human":
                    parse_failed = "consecutive_human"
                    break
                text = value.strip()
                if not text:
                    parse_failed = "empty_human"
                    break
                messages.append({"role": "user", "content": text})
            elif role == "gpt":
                if prev_from == "gpt":
                    parse_failed = "consecutive_gpt"
                    break
                text = value.strip()
                if not text:
                    parse_failed = "empty_gpt"
                    break
                if _NL_PSEUDO_CALL_RE.search(text):
                    parse_failed = "pseudo_call_in_nl"
                    break
                messages.append({"role": "assistant", "content": text})
            elif role == "function_call":
                try:
                    call_obj = json.loads(value)
                    raw_name = str(call_obj.get("name", "")).strip()
                    args = call_obj.get("arguments")
                    if isinstance(args, str):
                        args = json.loads(args)
                    assert raw_name in name_map and isinstance(args, dict)
                except Exception:
                    parse_failed = "bad_function_call"
                    break
                total_tool_calls += 1
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": name_map[raw_name],
                                    "arguments": args,
                                },
                            }
                        ],
                    }
                )
            elif role == "observation":
                obs_text = value.strip()
                if not obs_text:
                    parse_failed = "empty_observation"
                    break
                prev_msg = messages[-1] if messages else {}
                if prev_msg.get("role") != "assistant" or not prev_msg.get("tool_calls"):
                    parse_failed = "orphan_observation"
                    break
                has_tool_response = True
                num_tool_messages += 1
                obs_name = prev_msg["tool_calls"][0]["function"]["name"]
                messages.append({"role": "tool", "name": obs_name, "content": obs_text})
            else:
                parse_failed = f"unexpected_from:{role}"
                break
            prev_from = role

        if parse_failed:
            stats.dropped[parse_failed] += 1
            continue
        if not has_tool_response or total_tool_calls == 0:
            stats.dropped["no_tool_interaction"] += 1
            continue
        if require_final_nl and (
            messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls")
        ):
            stats.dropped["missing_final_nl"] += 1
            continue

        if excluded_first_turns:
            first_user = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            norm_first = re.sub(r"\s+", " ", first_user.strip().lower())
            if norm_first in excluded_first_turns:
                stats.dropped["eval_overlap"] += 1
                continue

        if excluded_entities:
            msg_blob = json.dumps(messages, ensure_ascii=False)
            if any(ent in msg_blob for ent in excluded_entities):
                stats.dropped["tau2_task_entity_overlap"] += 1
                continue

        domain = "airline" if "airline" in sys_prompt.lower() else "retail"
        num_user_turns = sum(1 for m in messages if m["role"] == "user")
        record = {
            "messages": messages,
            "tools": tools,
            "meta": {
                "source": "apigen-mt-5k",
                "source_id": idx,
                "domain": domain,
                "bucket": "apigen_multi_turn",
                "num_turns": num_user_turns,
                "num_user_turns": num_user_turns,
                "num_tool_messages": num_tool_messages,
                "num_tool_calls": total_tool_calls,
                "num_candidate_tools": len(tools),
                "has_tool_response": True,
                "is_parallel": False,
                "is_refusal": False,
            },
        }
        ok, reason = validate_record(
            record,
            strict_arg_types=strict_validation,
            reject_duplicate_tools=strict_validation,
        )
        if not ok:
            stats.dropped[reason] += 1
            continue

        stats.kept_rows += 1
        stats.buckets["apigen_multi_turn"] += 1
        records.append(record)

    return records, stats
