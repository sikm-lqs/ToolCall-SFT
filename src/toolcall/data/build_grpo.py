"""Build strictly validated, template-verified GRPO Smoke (300) and v1 (1,500) datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from transformers import AutoTokenizer
import yaml

from .build_sft_v1 import _deduplicate, _sha256_file, _write_jsonl
from .build_sft_v2_curated import (
    _load_eval_user_turns,
    _load_tau2_excluded_turns,
    _norm_text,
)
from .build_sft_v3 import _load_historical_val_keys
from .parsers import _NL_PSEUDO_CALL_RE, load_toolace, load_xlam_60k
from .schema import validate_record

ROOT = Path(__file__).resolve().parents[3]
_FORBIDDEN_SYS_RE = re.compile(
    r"(If none of the function can be used|tools call sections|"
    r"return the function call|\[FuncName|<function=)",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "about", "after", "again", "against", "all", "also", "and", "any", "are",
        "based", "because", "been", "before", "being", "between", "both", "but",
        "can", "could", "data", "does", "done", "each", "find", "for", "from",
        "function", "get", "give", "given", "have", "help", "here", "how",
        "info", "information", "into", "its", "just", "list", "make", "more",
        "most", "need", "new", "not", "now", "only", "other", "our", "out",
        "over", "please", "provide", "provided", "query", "request", "return",
        "returns", "should", "show", "some", "such", "than", "that", "the",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "through", "tool", "tools", "under", "use", "used", "user", "using",
        "value", "values", "very", "want", "was", "were", "what", "when",
        "where", "which", "while", "who", "will", "with", "would", "your",
    }
)


def _content_tokens(text: str) -> set[str]:
    parts = re.findall(r"[a-zA-Z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower())
    return {p for p in parts if len(p) >= 4 and p not in _STOPWORDS}


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for m in messages:
        role = str(m["role"])
        if role == "system":
            text = str(m.get("content") or "").strip()
            if _FORBIDDEN_SYS_RE.search(text):
                raise AssertionError(f"Forbidden system instruction found: {text[:120]}")
            if text:
                cleaned.append({"role": "system", "content": text})
        else:
            cleaned.append(copy.deepcopy(m))
    return cleaned


def _render_grpo_prompt(
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, int]:
    rendered = tokenizer.apply_chat_template(
        prompt_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
        raise AssertionError("Rendered GRPO prompt missing Qwen3.5 non-thinking assistant prefix")
    if tools and "<tools>" not in rendered:
        raise AssertionError("Rendered GRPO prompt missing <tools> schema block")
    ids = tokenizer(text=[rendered])["input_ids"][0]
    return rendered, len(ids)


def _verify_token_equivalence(
    tokenizer: Any, prompt: str, prompt_messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> None:
    ids_text = tokenizer(text=[prompt])["input_ids"][0]
    tpl_out = tokenizer.apply_chat_template(
        prompt_messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids_tpl = list(tpl_out["input_ids"] if isinstance(tpl_out, dict) or hasattr(tpl_out, "input_ids") else tpl_out)
    if ids_text != ids_tpl:
        raise AssertionError("Tokenization mismatch between pre-rendered prompt text and apply_chat_template")


def _tool_calls_to_gt(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tc in tool_calls:
        fn = tc["function"] if "function" in tc else tc
        out.append({"name": str(fn["name"]), "arguments": copy.deepcopy(fn["arguments"])})
    return out


def _group_key_for_record(rec: dict[str, Any]) -> tuple[str, str]:
    src = str(rec["meta"]["source"])
    sid = str(rec["meta"]["source_id"])
    return (src, sid)


def _user_turn_keys(messages: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {("user", _norm_text(str(m.get("content") or ""))) for m in messages if m.get("role") == "user"}


def _make_grpo_record(
    *,
    tokenizer: Any,
    max_prompt_tokens: int,
    prompt_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    task_type: str,
    bucket: str,
    ground_truth: dict[str, Any],
    source: str,
    source_id: Any,
    group_key: str,
    eval_turns: set[str],
    tau2_turns: set[str],
) -> dict[str, Any] | None:
    if not tools:
        return None
    if any(t["function"]["name"] == "think" for t in tools):
        return None

    clean_msgs = _clean_messages(prompt_messages)
    for m in clean_msgs:
        if m["role"] == "user":
            norm_u = _norm_text(str(m.get("content") or ""))
            if norm_u in eval_turns or norm_u in tau2_turns:
                return None

    prompt_str, n_tok = _render_grpo_prompt(tokenizer, clean_msgs, tools)
    if n_tok > max_prompt_tokens:
        return None

    return {
        "prompt": prompt_str,
        "prompt_messages": clean_msgs,
        "tools": copy.deepcopy(tools),
        "task_type": task_type,
        "bucket": bucket,
        "ground_truth": ground_truth,
        "meta": {
            "source": source,
            "source_id": str(source_id),
            "group_key": group_key,
            "num_candidate_tools": len(tools),
            "num_prompt_tokens": n_tok,
        },
    }


def _slice_toolace_multi_turn(
    mt_records: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int,
    eval_turns: set[str],
    tau2_turns: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    continues: list[dict[str, Any]] = []
    post_resps: list[dict[str, Any]] = []

    for rec in mt_records:
        msgs = rec["messages"]
        tools = rec["tools"]
        src_id = rec["meta"]["source_id"]
        gkey = f"toolace:{src_id}"

        cont_added = False
        post_added = False
        for idx, m in enumerate(msgs):
            if m["role"] != "assistant" or idx == 0 or msgs[idx - 1]["role"] != "tool":
                continue
            prefix = msgs[:idx]
            obs_text = str(msgs[idx - 1].get("content") or "")
            if m.get("tool_calls") and not cont_added:
                item = _make_grpo_record(
                    tokenizer=tokenizer,
                    max_prompt_tokens=max_prompt_tokens,
                    prompt_messages=prefix,
                    tools=tools,
                    task_type="call",
                    bucket="multi_turn_continue",
                    ground_truth={
                        "tool_calls": _tool_calls_to_gt(m["tool_calls"]),
                        "reference_text": "",
                        "missing_params": [],
                        "tool_observation": obs_text,
                    },
                    source="toolace_multi_turn",
                    source_id=f"{src_id}:step_{idx}",
                    group_key=gkey,
                    eval_turns=eval_turns,
                    tau2_turns=tau2_turns,
                )
                if item is not None:
                    continues.append(item)
                    cont_added = True
            elif not m.get("tool_calls") and not post_added:
                nl = str(m.get("content") or "").strip()
                if not nl or _NL_PSEUDO_CALL_RE.search(nl):
                    continue
                item = _make_grpo_record(
                    tokenizer=tokenizer,
                    max_prompt_tokens=max_prompt_tokens,
                    prompt_messages=prefix,
                    tools=tools,
                    task_type="non_call",
                    bucket="multi_turn_post_tool_response",
                    ground_truth={
                        "tool_calls": [],
                        "reference_text": nl,
                        "missing_params": [],
                        "tool_observation": obs_text,
                    },
                    source="toolace_multi_turn",
                    source_id=f"{src_id}:step_{idx}",
                    group_key=gkey,
                    eval_turns=eval_turns,
                    tau2_turns=tau2_turns,
                )
                if item is not None:
                    post_resps.append(item)
                    post_added = True
    return continues, post_resps


def _build_miss_func_recover_pool(
    call_records: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int,
    eval_turns: set[str],
    tau2_turns: set[str],
    rng: random.Random,
    limit: int = 400,
) -> list[dict[str, Any]]:
    candidates = [
        r for r in call_records
        if len(r["messages"]) == 2
        and r["messages"][0]["role"] == "user"
        and r["messages"][1]["role"] == "assistant"
        and r["messages"][1].get("tool_calls")
    ]
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    out: list[dict[str, Any]] = []
    for rec in shuffled:
        if len(out) >= limit:
            break
        user_q = str(rec["messages"][0]["content"])
        tcalls = rec["messages"][1]["tool_calls"]
        called_names = sorted({tc["function"]["name"] for tc in tcalls})
        missing_str = ", ".join(called_names)
        prompt_msgs = [
            {"role": "user", "content": user_q},
            {
                "role": "assistant",
                "content": (
                    f"I cannot complete this request right now because the required tool ({missing_str}) "
                    "is not available in the current toolset. Please enable or provide it so I can proceed."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"I have enabled the {missing_str} tool(s) now. "
                    "Please go ahead and call the appropriate tool to complete my original request."
                ),
            },
        ]
        src = rec["meta"]["source"]
        sid = rec["meta"]["source_id"]
        item = _make_grpo_record(
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            prompt_messages=prompt_msgs,
            tools=rec["tools"],
            task_type="call",
            bucket="multi_turn_miss_func_recover",
            ground_truth={
                "tool_calls": _tool_calls_to_gt(tcalls),
                "reference_text": "",
                "missing_params": [],
                "tool_observation": "",
            },
            source=f"{src}:miss_func_recover",
            source_id=f"{sid}:recover",
            group_key=f"{src}:{sid}",
            eval_turns=eval_turns,
            tau2_turns=tau2_turns,
        )
        if item is not None:
            out.append(item)
    return out


def _build_adversarial_1tool_irrelevance_pool(
    single_records: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int,
    eval_turns: set[str],
    tau2_turns: set[str],
    rng: random.Random,
    limit: int = 200,
) -> list[dict[str, Any]]:
    shuffled = [r for r in single_records if len(r["tools"]) >= 2 and len(r["messages"]) == 2]
    rng.shuffle(shuffled)
    out: list[dict[str, Any]] = []
    for rec in shuffled:
        if len(out) >= limit:
            break
        tcalls = rec["messages"][1].get("tool_calls") or []
        if not tcalls:
            continue
        called_names = {tc["function"]["name"] for tc in tcalls}
        user_q = str(rec["messages"][0]["content"])
        q_tokens = _content_tokens(user_q)
        gold_tokens = set()
        for t in rec["tools"]:
            if t["function"]["name"] in called_names:
                gold_tokens |= _content_tokens(t["function"]["name"] + " " + t["function"].get("description", ""))

        disjoint_tools = []
        for t in rec["tools"]:
            fn = t["function"]
            if fn["name"] in called_names:
                continue
            cand_tokens = _content_tokens(fn["name"] + " " + fn.get("description", ""))
            if cand_tokens and not (cand_tokens & q_tokens) and not (cand_tokens & gold_tokens):
                disjoint_tools.append(t)

        if not disjoint_tools:
            continue
        chosen = rng.choice(disjoint_tools)
        src = rec["meta"]["source"]
        sid = rec["meta"]["source_id"]
        item = _make_grpo_record(
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            prompt_messages=[{"role": "user", "content": user_q}],
            tools=[chosen],
            task_type="non_call",
            bucket="irrelevance_refusal",
            ground_truth={
                "tool_calls": [],
                "reference_text": "",
                "missing_params": [],
                "tool_observation": "",
            },
            source=f"{src}:adv_1tool_irrelevance",
            source_id=f"{sid}:adv_irr",
            group_key=f"{src}:{sid}",
            eval_turns=eval_turns,
            tau2_turns=tau2_turns,
        )
        if item is not None:
            out.append(item)
    return out


def _build_missing_param_records(
    mp_records: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int,
    eval_turns: set[str],
    tau2_turns: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in mp_records:
        msgs = rec["messages"]
        assts = [m for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")]
        if not assts:
            continue
        nl = str(assts[-1].get("content") or "").strip()
        if not nl or _NL_PSEUDO_CALL_RE.search(nl):
            continue
        # Last assistant message is target; preceding messages form prompt
        last_idx = len(msgs) - 1
        while last_idx >= 0 and msgs[last_idx]["role"] != "assistant":
            last_idx -= 1
        if last_idx <= 0:
            continue
        req_params = sorted(
            {
                p
                for t in rec["tools"]
                for p in (t["function"].get("parameters", {}).get("required") or [])
                if p.lower() in nl.lower()
            }
        )
        sid = rec["meta"]["source_id"]
        item = _make_grpo_record(
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            prompt_messages=msgs[:last_idx],
            tools=rec["tools"],
            task_type="non_call",
            bucket="missing_param",
            ground_truth={
                "tool_calls": [],
                "reference_text": nl,
                "missing_params": req_params,
                "tool_observation": "",
            },
            source="toolace_missing_param",
            source_id=sid,
            group_key=f"toolace:{sid}",
            eval_turns=eval_turns,
            tau2_turns=tau2_turns,
        )
        if item is not None:
            out.append(item)
    return out


def _build_irrelevant_refusal_records(
    irr_records: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int,
    eval_turns: set[str],
    tau2_turns: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in irr_records:
        msgs = rec["messages"]
        assts = [m for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")]
        if not assts:
            continue
        nl = str(assts[-1].get("content") or "").strip()
        if not nl or _NL_PSEUDO_CALL_RE.search(nl):
            continue
        last_idx = len(msgs) - 1
        while last_idx >= 0 and msgs[last_idx]["role"] != "assistant":
            last_idx -= 1
        if last_idx <= 0:
            continue
        sid = rec["meta"]["source_id"]
        item = _make_grpo_record(
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            prompt_messages=msgs[:last_idx],
            tools=rec["tools"],
            task_type="non_call",
            bucket="irrelevance_refusal",
            ground_truth={
                "tool_calls": [],
                "reference_text": nl,
                "missing_params": [],
                "tool_observation": "",
            },
            source="toolace_irrelevant_refusal",
            source_id=sid,
            group_key=f"toolace:{sid}",
            eval_turns=eval_turns,
            tau2_turns=tau2_turns,
        )
        if item is not None:
            out.append(item)
    return out


def _build_single_or_parallel_call_records(
    records: list[dict[str, Any]],
    bucket: str,
    tokenizer: Any,
    max_prompt_tokens: int,
    eval_turns: set[str],
    tau2_turns: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        msgs = rec["messages"]
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant" or not msgs[-1].get("tool_calls"):
            continue
        src = rec["meta"]["source"]
        sid = rec["meta"]["source_id"]
        item = _make_grpo_record(
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            prompt_messages=msgs[:-1],
            tools=rec["tools"],
            task_type="call",
            bucket=bucket,
            ground_truth={
                "tool_calls": _tool_calls_to_gt(msgs[-1]["tool_calls"]),
                "reference_text": "",
                "missing_params": [],
                "tool_observation": "",
            },
            source=src,
            source_id=sid,
            group_key=f"{src}:{sid}",
            eval_turns=eval_turns,
            tau2_turns=tau2_turns,
        )
        if item is not None:
            out.append(item)
    return out


def _pick_unique_by_group(
    pool: list[dict[str, Any]],
    n: int,
    used_groups: set[str],
    used_user_turns: set[str],
    rng: random.Random,
    *,
    min_one_tool_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    items = list(pool)
    rng.shuffle(items)

    def _can_take(item: dict[str, Any]) -> bool:
        if item["meta"]["group_key"] in used_groups:
            return False
        for m in item["prompt_messages"]:
            if m["role"] == "user" and _norm_text(str(m.get("content") or "")) in used_user_turns:
                return False
        return True

    def _take(item: dict[str, Any], chosen: list[dict[str, Any]]) -> None:
        chosen.append(item)
        used_groups.add(item["meta"]["group_key"])
        for m in item["prompt_messages"]:
            if m["role"] == "user":
                used_user_turns.add(_norm_text(str(m.get("content") or "")))

    chosen: list[dict[str, Any]] = []
    min_one = int(round(n * min_one_tool_ratio))
    if min_one > 0:
        for item in items:
            if len(chosen) >= min_one:
                break
            if item["meta"]["num_candidate_tools"] == 1 and _can_take(item):
                _take(item, chosen)

    for item in items:
        if len(chosen) >= n:
            break
        if _can_take(item):
            _take(item, chosen)

    if len(chosen) < n:
        raise ValueError(f"Insufficient unique candidates: needed {n}, got {len(chosen)}")
    return chosen


def _load_excluded_val_sets(paths: list[Path]) -> tuple[set[str], set[str]]:
    group_keys: set[str] = set()
    user_turns: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                src = rec.get("meta", {}).get("source")
                sid = rec.get("meta", {}).get("source_id")
                if src and sid is not None:
                    group_keys.add(f"{src}:{sid}")
                for m in rec.get("messages") or rec.get("prompt_messages") or []:
                    if m.get("role") == "user":
                        user_turns.add(_norm_text(str(m.get("content") or "")))
    return group_keys, user_turns


def _write_grpo_bundle(
    *,
    out_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    tokenizer: Any,
    eval_turns: set[str],
    tau2_turns: set[str],
    excl_groups: set[str],
    excl_turns: set[str],
    cfg: dict[str, Any],
    split_tag: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in (("train", train_rows), ("val", val_rows)):
        for i, row in enumerate(rows):
            row["meta"]["split"] = split_name
            row["meta"]["sample_id"] = f"{split_tag}_{split_name}_{i:05d}"
            _verify_token_equivalence(tokenizer, row["prompt"], row["prompt_messages"], row["tools"])

    train_groups = {r["meta"]["group_key"] for r in train_rows}
    val_groups = {r["meta"]["group_key"] for r in val_rows}
    if train_groups & val_groups:
        raise AssertionError(f"Train/val group overlap: {len(train_groups & val_groups)}")
    if (train_groups | val_groups) & excl_groups:
        raise AssertionError("Overlap with excluded validation group_keys")

    for r in train_rows + val_rows:
        for m in r["prompt_messages"]:
            if m["role"] == "user":
                ut = _norm_text(str(m.get("content") or ""))
                if ut in eval_turns or ut in tau2_turns or ut in excl_turns:
                    raise AssertionError(f"User turn overlap in {r['meta']['sample_id']}")

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    ledger_path = out_dir / "audit_ledger.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    ledger_rows = [
        {
            "sample_id": r["meta"]["sample_id"],
            "split": r["meta"]["split"],
            "task_type": r["task_type"],
            "bucket": r["bucket"],
            "source": r["meta"]["source"],
            "source_id": r["meta"]["source_id"],
            "group_key": r["meta"]["group_key"],
            "num_candidate_tools": r["meta"]["num_candidate_tools"],
            "num_prompt_tokens": r["meta"]["num_prompt_tokens"],
            "prompt_sha256": hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest(),
        }
        for r in train_rows + val_rows
    ]
    _write_jsonl(ledger_path, ledger_rows)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    seen_b: set[str] = set()
    for r in train_rows:
        b = r["bucket"]
        if b in seen_b:
            continue
        seen_b.add(b)
        (samples_dir / f"{b}.txt").write_text(
            f"# sample_id: {r['meta']['sample_id']}\n"
            f"# task_type: {r['task_type']} | bucket: {b} | source: {r['meta']['source']}\n"
            f"# num_prompt_tokens: {r['meta']['num_prompt_tokens']} | num_tools: {r['meta']['num_candidate_tools']}\n"
            f"# ground_truth: {json.dumps(r['ground_truth'], ensure_ascii=False)}\n"
            + "=" * 80 + "\n"
            + r["prompt"],
            encoding="utf-8",
        )

    def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        toks = [r["meta"]["num_prompt_tokens"] for r in rows]
        return {
            "count": len(rows),
            "buckets": dict(Counter(r["bucket"] for r in rows)),
            "task_types": dict(Counter(r["task_type"] for r in rows)),
            "one_tool_rows": sum(1 for r in rows if r["meta"]["num_candidate_tools"] == 1),
            "prompt_tokens_total": sum(toks),
            "prompt_tokens_avg": round(sum(toks) / max(1, len(rows)), 2),
            "prompt_tokens_max": max(toks) if toks else 0,
        }

    manifest = {
        "version": split_tag,
        "seed": cfg["seed"],
        "max_prompt_tokens": cfg["max_prompt_tokens"],
        "outputs": {
            "train.jsonl": {"sha256": _sha256_file(train_path), **_stats(train_rows)},
            "val.jsonl": {"sha256": _sha256_file(val_path), **_stats(val_rows)},
            "audit_ledger.jsonl": {"sha256": _sha256_file(ledger_path), "count": len(ledger_rows)},
        },
        "invariants": {
            "train_val_group_overlap": 0,
            "excluded_val_overlap": 0,
            "eval_user_turn_overlap": 0,
            "token_equivalence_verified": True,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_grpo_pools(
    cfg: dict[str, Any],
    tokenizer: Any,
    *,
    adv_irr_limit: int = 250,
    miss_func_limit: int = 500,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    set[str],
    set[str],
    set[str],
    set[str],
]:
    """Load, clean, deduplicate, and slice all 7 GRPO candidate pools."""
    seed = int(cfg["seed"])
    rng = random.Random(seed)
    max_prompt_tokens = int(cfg["max_prompt_tokens"])

    toolace_cfg = cfg["sources"]["toolace"]
    xlam_cfg = cfg["sources"]["xlam_60k"]
    toolace_recs, _ = load_toolace(
        ROOT / toolace_cfg["raw_path"],
        strip_policy_prompt=bool(toolace_cfg.get("strip_policy_prompt", True)),
        split_no_call_subtypes=bool(toolace_cfg.get("split_no_call_subtypes", True)),
        strict_validation=bool(toolace_cfg.get("strict_validation", True)),
    )
    for rec in toolace_recs:
        if rec["meta"]["bucket"] == "toolace_single_or_parallel":
            rec["meta"]["bucket"] = (
                "toolace_parallel" if rec["meta"]["is_parallel"] else "toolace_single"
            )
    xlam_recs, _ = load_xlam_60k(
        ROOT / xlam_cfg["raw_path"],
        strict_validation=bool(xlam_cfg.get("strict_validation", True)),
    )

    seen_sigs: set[str] = set()
    toolace_dedup, _ = _deduplicate(toolace_recs, seen_sigs)
    xlam_dedup, _ = _deduplicate(xlam_recs, seen_sigs)

    eval_turns = _load_eval_user_turns([ROOT / d for d in cfg.get("eval_inspection_dirs", [])])
    tau2_turns = _load_tau2_excluded_turns(ROOT / cfg["tau2_domains_dir"])
    excl_groups, excl_turns = _load_excluded_val_sets(
        [ROOT / p for p in cfg.get("exclude_val_paths", [])]
    )

    ta_by_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in toolace_dedup:
        ta_by_b[r["meta"]["bucket"]].append(r)
    xl_by_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in xlam_dedup:
        xl_by_b[r["meta"]["bucket"]].append(r)

    cont_pool, post_resp_pool = _slice_toolace_multi_turn(
        ta_by_b["toolace_multi_turn"], tokenizer, max_prompt_tokens, eval_turns, tau2_turns
    )
    single_pool = _build_single_or_parallel_call_records(
        ta_by_b["toolace_single"] + xl_by_b["xlam_single"],
        "single_turn_call",
        tokenizer,
        max_prompt_tokens,
        eval_turns,
        tau2_turns,
    )
    parallel_pool = _build_single_or_parallel_call_records(
        ta_by_b["toolace_parallel"] + xl_by_b["xlam_parallel"],
        "parallel_call",
        tokenizer,
        max_prompt_tokens,
        eval_turns,
        tau2_turns,
    )
    mp_pool = _build_missing_param_records(
        ta_by_b["toolace_missing_param"], tokenizer, max_prompt_tokens, eval_turns, tau2_turns
    )
    irr_nat_pool = _build_irrelevant_refusal_records(
        ta_by_b["toolace_irrelevant_refusal"], tokenizer, max_prompt_tokens, eval_turns, tau2_turns
    )
    irr_adv_pool = _build_adversarial_1tool_irrelevance_pool(
        ta_by_b["toolace_single"] + xl_by_b["xlam_single"],
        tokenizer,
        max_prompt_tokens,
        eval_turns,
        tau2_turns,
        rng,
        limit=adv_irr_limit,
    )
    irr_pool = irr_nat_pool + irr_adv_pool
    miss_func_pool = _build_miss_func_recover_pool(
        ta_by_b["toolace_single"] + xl_by_b["xlam_single"],
        tokenizer,
        max_prompt_tokens,
        eval_turns,
        tau2_turns,
        rng,
        limit=miss_func_limit,
    )

    pools = {
        "single_turn_call": single_pool,
        "multi_turn_continue": cont_pool,
        "multi_turn_miss_func_recover": miss_func_pool,
        "multi_turn_post_tool_response": post_resp_pool,
        "parallel_call": parallel_pool,
        "irrelevance_refusal": irr_pool,
        "missing_param": mp_pool,
    }
    return pools, eval_turns, tau2_turns, excl_groups, excl_turns


def build_grpo(
    config_path: Path, tokenizer_override: Path | None = None
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    rng = random.Random(seed)

    tok_dir = (
        tokenizer_override
        if tokenizer_override is not None
        else Path(cfg["tokenizer"]["snapshot"]).expanduser()
    )
    if not tok_dir.is_absolute():
        tok_dir = (ROOT / tok_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir, local_files_only=True, trust_remote_code=False
    )

    pools, eval_turns, tau2_turns, excl_groups, excl_turns = prepare_grpo_pools(
        cfg, tokenizer, adv_irr_limit=250, miss_func_limit=500
    )

    # Build v1 FIRST (val held out first, isolated from excluded vals, then train)
    v1_cfg = cfg["outputs"]["v1"]
    used_groups = set(excl_groups)
    used_turns = set(excl_turns)

    v1_val: list[dict[str, Any]] = []
    for b, cnt in v1_cfg["val_counts"].items():
        ratio = 0.45 if b == "single_turn_call" else 0.0
        v1_val.extend(_pick_unique_by_group(pools[b], int(cnt), used_groups, used_turns, rng, min_one_tool_ratio=ratio))

    v1_train: list[dict[str, Any]] = []
    v1_train_by_b: dict[str, list[dict[str, Any]]] = {}
    for b, cnt in v1_cfg["train_counts"].items():
        ratio = 0.45 if b == "single_turn_call" else 0.0
        picked = _pick_unique_by_group(pools[b], int(cnt), used_groups, used_turns, rng, min_one_tool_ratio=ratio)
        v1_train_by_b[b] = picked
        v1_train.extend(picked)

    rng.shuffle(v1_train)
    rng.shuffle(v1_val)

    v1_manifest = _write_grpo_bundle(
        out_dir=ROOT / v1_cfg["output_dir"],
        train_rows=v1_train,
        val_rows=v1_val,
        tokenizer=tokenizer,
        eval_turns=eval_turns,
        tau2_turns=tau2_turns,
        excl_groups=excl_groups,
        excl_turns=excl_turns,
        cfg=cfg,
        split_tag="grpo_v1",
    )

    # Smoke is sampled cleanly from v1 train & v1 val (preserving 45%+ 1-tool share in single_turn_call)
    smoke_cfg = cfg["outputs"]["smoke"]
    v1_val_by_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in v1_val:
        v1_val_by_b[r["bucket"]].append(r)

    smoke_train: list[dict[str, Any]] = []
    for b, cnt in smoke_cfg["train_counts"].items():
        items = copy.deepcopy(v1_train_by_b[b])
        if b == "single_turn_call":
            one_t = [x for x in items if x["meta"]["num_candidate_tools"] == 1]
            multi_t = [x for x in items if x["meta"]["num_candidate_tools"] > 1]
            n1 = int(round(int(cnt) * 0.45))
            smoke_train.extend(one_t[:n1] + multi_t[: int(cnt) - n1])
        else:
            smoke_train.extend(items[: int(cnt)])

    smoke_val: list[dict[str, Any]] = []
    for b, cnt in smoke_cfg["val_counts"].items():
        smoke_val.extend(copy.deepcopy(v1_val_by_b[b][: int(cnt)]))

    rng.shuffle(smoke_train)
    rng.shuffle(smoke_val)

    smoke_manifest = _write_grpo_bundle(
        out_dir=ROOT / smoke_cfg["output_dir"],
        train_rows=smoke_train,
        val_rows=smoke_val,
        tokenizer=tokenizer,
        eval_turns=eval_turns,
        tau2_turns=tau2_turns,
        excl_groups=excl_groups,
        excl_turns=excl_turns,
        cfg=cfg,
        split_tag="grpo_smoke",
    )

    return {"smoke": smoke_manifest, "v1": v1_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/grpo_v1.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=None)
    args = parser.parse_args()
    res = build_grpo(args.config, tokenizer_override=args.tokenizer)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
