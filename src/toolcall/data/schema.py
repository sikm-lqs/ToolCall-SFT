"""Schema normalization, tool-name sanitization, validation, and deduplication."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any

_SIMPLE_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "object": "object",
    "list": "array",
    "array": "array",
    "set": "array",
    "tuple": "array",
}

_LIST_INNER_RE = re.compile(r"^(?:List|Sequence|Set)\[(.+)\]$", re.IGNORECASE)
_TUPLE_RE = re.compile(r"^Tuple\[(.+)\]$", re.IGNORECASE)
_DICT_RE = re.compile(r"^Dict(?:\[.*\])?$", re.IGNORECASE)
_OPT_SUFFIX_RE = re.compile(r",?\s*optional\b", re.IGNORECASE)
_DEFAULT_INLINE_RE = re.compile(r",?\s*default\s*=?\s*.+$", re.IGNORECASE)


def sanitize_tool_name(name: str) -> str:
    """Normalize arbitrary tool names (e.g., spaces/parens) to `[A-Za-z0-9_.-]+`."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "tool"


def build_tool_name_mapping(raw_names: list[str]) -> dict[str, str]:
    """Map raw tool names to collision-free OpenAI-safe function names."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for raw in raw_names:
        if raw in mapping:
            continue
        base = sanitize_tool_name(raw)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        mapping[raw] = candidate
    return mapping


def _map_python_type_expr(expr: str) -> dict[str, Any]:
    """Convert a Python-like xLAM type expression into a JSON Schema fragment."""
    cleaned = expr.strip()
    lower = cleaned.lower()
    if lower in _SIMPLE_TYPE_MAP:
        return {"type": _SIMPLE_TYPE_MAP[lower]}

    if _DICT_RE.match(cleaned):
        return {"type": "object"}

    if _TUPLE_RE.match(cleaned):
        return {"type": "array"}

    list_match = _LIST_INNER_RE.match(cleaned)
    if list_match:
        inner = list_match.group(1).strip()
        inner_schema = _map_python_type_expr(inner)
        if inner_schema:
            return {"type": "array", "items": inner_schema}
        return {"type": "array"}

    return {}


def normalize_xlam_parameters(raw_params: dict[str, Any]) -> dict[str, Any]:
    """Convert xLAM's flat `{param: {type, description, default}}` to JSON Schema."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, info in raw_params.items():
        if not isinstance(info, dict):
            properties[param_name] = {"type": "string", "description": str(info)}
            required.append(param_name)
            continue

        raw_type = str(info.get("type", "")).strip()
        is_optional = bool(_OPT_SUFFIX_RE.search(raw_type)) or ("default" in info)

        base_type = _OPT_SUFFIX_RE.sub("", raw_type).strip()
        base_type = _DEFAULT_INLINE_RE.sub("", base_type).strip().rstrip(",").strip()

        prop: dict[str, Any] = _map_python_type_expr(base_type)
        desc = str(info.get("description") or "").strip()
        if not prop and base_type and base_type.lower() not in {"any", "object"}:
            desc = f"{desc} (type: {base_type})".strip() if desc else f"type: {base_type}"
        if desc:
            prop["description"] = desc
        if "default" in info:
            prop["default"] = info["default"]

        properties[param_name] = prop
        if not is_optional:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _normalize_schema_type_value(value: Any) -> Any:
    if isinstance(value, str):
        return _SIMPLE_TYPE_MAP.get(value.strip().lower(), value)
    if isinstance(value, list):
        return [_normalize_schema_type_value(item) for item in value]
    return value


def normalize_json_schema_node(node: Any) -> Any:
    """Recursively normalize ToolACE schema types (`dict->object`, `int->integer`, `float->number`)."""
    if not isinstance(node, dict):
        return deepcopy(node)

    out = deepcopy(node)
    if "type" in out:
        out["type"] = _normalize_schema_type_value(out["type"])

    if isinstance(out.get("properties"), dict):
        out["properties"] = {
            prop_name: normalize_json_schema_node(prop_schema)
            for prop_name, prop_schema in out["properties"].items()
        }

    if isinstance(out.get("items"), dict):
        out["items"] = normalize_json_schema_node(out["items"])
    elif isinstance(out.get("items"), list):
        out["items"] = [normalize_json_schema_node(item) for item in out["items"]]

    if isinstance(out.get("additionalProperties"), dict):
        out["additionalProperties"] = normalize_json_schema_node(out["additionalProperties"])

    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(out.get(key), list):
            out[key] = [normalize_json_schema_node(branch) for branch in out[key]]

    if "required" in out and not isinstance(out["required"], list):
        out["required"] = list(out["required"]) if out["required"] else []

    return out


def normalize_toolace_parameters(raw_params: Any) -> dict[str, Any]:
    """Normalize a ToolACE `parameters` block into a standard JSON Schema object."""
    if not isinstance(raw_params, dict):
        return {"type": "object", "properties": {}, "required": []}
    norm = normalize_json_schema_node(raw_params)
    norm.setdefault("type", "object")
    if norm["type"] == "dict":
        norm["type"] = "object"
    norm.setdefault("properties", {})
    norm.setdefault("required", [])
    return norm


def validate_record(record: dict[str, Any]) -> tuple[bool, str]:
    """Verify structural, schema, and turn-pairing invariants for a normalized OpenAI sample."""
    tools = record.get("tools")
    messages = record.get("messages")
    if not isinstance(tools, list) or not isinstance(messages, list) or not messages:
        return False, "invalid_top_level"

    tool_map: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return False, "invalid_tool_entry"
        fn = tool.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            return False, "invalid_tool_function"
        params = fn.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            return False, "invalid_tool_parameters"
        tool_map[fn["name"]] = fn

    has_user = False
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "system":
            if idx != 0:
                return False, "non_leading_system"
        elif role == "user":
            has_user = True
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                return False, "empty_user"
        elif role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            if content is None:
                content = ""
            if not isinstance(content, str):
                return False, "non_string_assistant_content"
            if tool_calls is not None:
                if not isinstance(tool_calls, list) or not tool_calls:
                    return False, "empty_tool_calls_list"
                for call in tool_calls:
                    if not isinstance(call, dict) or call.get("type") != "function":
                        return False, "invalid_tool_call_entry"
                    fn_call = call.get("function")
                    if not isinstance(fn_call, dict):
                        return False, "invalid_tool_call_function"
                    name = fn_call.get("name")
                    args = fn_call.get("arguments")
                    if name not in tool_map:
                        return False, "unknown_tool_call"
                    if not isinstance(args, dict):
                        return False, "arguments_not_dict"
                    required = tool_map[name]["parameters"].get("required") or []
                    for req in required:
                        if req not in args:
                            return False, "missing_required_arg"
            elif not content.strip():
                return False, "empty_assistant_turn"
        elif role == "tool":
            if idx == 0:
                return False, "leading_tool_msg"
            prev = messages[idx - 1]
            if not (
                prev.get("role") == "tool"
                or (prev.get("role") == "assistant" and prev.get("tool_calls"))
            ):
                return False, "orphan_tool_msg"
            if not isinstance(msg.get("content"), str):
                return False, "non_string_tool_content"
        else:
            return False, f"unexpected_role:{role}"

    if not has_user:
        return False, "missing_user"
    if messages[-1].get("role") != "assistant":
        return False, "must_end_with_assistant"

    return True, "ok"


def dedup_signature(record: dict[str, Any]) -> str:
    """Compute canonical deduplication digest from user turns + declared tool names."""
    users = [m["content"].strip() for m in record["messages"] if m["role"] == "user"]
    tool_names = sorted(t["function"]["name"] for t in record["tools"])
    payload = json.dumps({"users": users, "tools": tool_names}, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
