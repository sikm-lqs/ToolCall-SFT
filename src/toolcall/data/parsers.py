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


def load_xlam_60k(path: Path) -> tuple[list[dict[str, Any]], ParseStats]:
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
        ok, reason = validate_record(record)
        if not ok:
            stats.dropped[reason] += 1
            continue

        stats.kept_rows += 1
        stats.buckets[bucket] += 1
        records.append(record)

    return records, stats


def load_toolace(path: Path) -> tuple[list[dict[str, Any]], ParseStats]:
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

        is_refusal = total_tool_calls == 0
        if has_tool_response:
            bucket = "toolace_multi_turn"
        elif is_refusal:
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
        ok, reason = validate_record(record)
        if not ok:
            stats.dropped[reason] += 1
            continue

        stats.kept_rows += 1
        stats.buckets[bucket] += 1
        records.append(record)

    return records, stats
