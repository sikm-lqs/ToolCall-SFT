"""Build `SFT-v2-curated` (~5.5k high-quality, multi-turn enriched, anti-leakage dataset)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any
import unicodedata

from transformers import AutoTokenizer
import yaml

from .build_sft_v1 import (
    _deduplicate,
    _filter_by_length,
    _length_summary,
    _sha256_file,
    _write_jsonl,
)
from .build_sft_v1r1 import _render_bucket_samples
from .parsers import load_apigen_mt_5k, load_toolace, load_xlam_60k
from .schema import dedup_signature, validate_record
from .smoltalk import DistractorToolIndex, load_smoltalk_irrelevant_records
from .template import patch_qwen35_chat_template

ROOT = Path(__file__).resolve().parents[3]

# Filter out SmolTalk prompts asking for real-time / external info so attaching
# API tools cannot look plausibly answerable by a missing or matched tool.
_LIVE_QUERY_RE = re.compile(
    r"\b(?:current weather|weather forecast|weather today|today's weather|"
    r"current temperature|current stock price|exchange rate today|latest news|"
    r"current time in|what time is it in)\b",
    re.IGNORECASE,
)


def _norm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _task_key(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _norm_text(msg["content"])
        for msg in record["messages"]
        if msg["role"] == "user"
    )


def _group_keys(record: dict[str, Any]) -> set[tuple[Any, ...]]:
    meta = record["meta"]
    keys: set[tuple[Any, ...]] = {
        ("messages", _task_key(record)),
        ("signature", dedup_signature(record)),
    }
    if meta.get("source_id") is not None:
        keys.add(("source", meta["source"], str(meta["source_id"])))
    return keys


def _load_tau2_excluded_turns(tau2_domains_dir: Path) -> set[str]:
    excluded: set[str] = set()
    if not tau2_domains_dir.exists():
        return excluded
    for domain in ("retail", "airline"):
        tasks_file = tau2_domains_dir / domain / "tasks.json"
        if not tasks_file.exists():
            continue
        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
        for task in tasks:
            inst = (task.get("user_scenario") or {}).get("instructions")
            if isinstance(inst, dict):
                for val in inst.values():
                    if isinstance(val, str) and val.strip():
                        excluded.add(_norm_text(val))
            elif isinstance(inst, str) and inst.strip():
                excluded.add(_norm_text(inst))
    return excluded


def _load_eval_user_turns(eval_dirs: list[Path]) -> set[str]:
    turns: set[str] = set()
    for d in eval_dirs:
        if not d.exists():
            continue
        for json_path in sorted(d.glob("*.json")):
            if json_path.name == "index.json":
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            req_msgs = (payload.get("request") or {}).get("messages") or []
            for m in req_msgs:
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    turns.add(_norm_text(m["content"]))
            case = payload.get("case") or {}
            q = case.get("question")
            if isinstance(q, list):
                for group in q:
                    if isinstance(group, list):
                        for m in group:
                            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                                turns.add(_norm_text(str(m["content"])))
    return turns


def _load_preapproved_smoltalk(clean_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_task: set[tuple[str, ...]] = set()
    if not clean_dir.exists():
        return records
    for fname in ("val.jsonl", "smoke_v1r1.jsonl", "train.jsonl"):
        fpath = clean_dir / fname
        if not fpath.exists():
            continue
        with fpath.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["meta"]["bucket"] != "smoltalk_irrelevant_tools":
                    continue
                tk = _task_key(rec)
                if tk in seen_task:
                    continue
                seen_task.add(tk)
                rec["meta"]["curated_origin"] = "v1r1_human_approved"
                records.append(rec)
    return records


def _filter_smoltalk_quality(
    records: list[dict[str, Any]], min_assistant_chars: int
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        users = [m["content"] for m in rec["messages"] if m["role"] == "user"]
        assts = [m["content"] for m in rec["messages"] if m["role"] == "assistant"]
        if any(_LIVE_QUERY_RE.search(u) for u in users):
            dropped += 1
            continue
        if any(len(a.strip()) < min_assistant_chars for a in assts):
            dropped += 1
            continue
        kept.append(rec)
    return kept, dropped


def _split_summary(recs: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket = Counter(r["meta"]["bucket"] for r in recs)
    by_source = Counter(r["meta"]["source"] for r in recs)
    toks = [r["meta"]["num_tokens"] for r in recs]
    asst = [r["meta"]["num_assistant_tokens"] for r in recs]
    parallel_cnt = sum(1 for r in recs if r["meta"]["is_parallel"])
    multi_turn_cnt = sum(1 for r in recs if r["meta"]["has_tool_response"])
    refusal_cnt = sum(1 for r in recs if r["meta"]["is_refusal"])
    irrelevance_cnt = sum(
        1
        for r in recs
        if r["meta"]["bucket"] in ("toolace_irrelevant_refusal", "smoltalk_irrelevant_tools")
    )
    multi_cand_cnt = sum(1 for r in recs if r["meta"]["num_candidate_tools"] >= 2)
    return {
        "total_rows": len(recs),
        "by_source": dict(sorted(by_source.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "traits": {
            "parallel": parallel_cnt,
            "has_tool_response": multi_turn_cnt,
            "no_call_refusal_or_clarify": refusal_cnt,
            "pure_irrelevance_with_tools": irrelevance_cnt,
            "multi_candidate_tools": multi_cand_cnt,
        },
        "total_tokens": _length_summary(toks),
        "assistant_tokens": _length_summary(asst),
    }


def _sample_balanced_apigen(
    pool: list[dict[str, Any]], count: int, rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample balanced Airline + Retail trajectories and return `(chosen, remaining)`."""
    airline = [r for r in pool if r["meta"].get("domain") == "airline"]
    retail = [r for r in pool if r["meta"].get("domain") != "airline"]
    rng.shuffle(airline)
    rng.shuffle(retail)
    half = count // 2
    chosen = airline[:half] + retail[: count - half]
    used_ids = {r["meta"]["source_id"] for r in chosen}
    remaining = [r for r in pool if r["meta"]["source_id"] not in used_ids]
    return chosen, remaining


def build_sft_v2_curated(
    config_path: Path, tokenizer_override: Path | None = None
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    rng = random.Random(seed)
    max_seq_length = int(cfg["max_seq_length"])

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
    raw_template = (tok_dir / "chat_template.jinja").read_text(encoding="utf-8")
    patched_template = patch_qwen35_chat_template(raw_template)

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    toolace_cfg = cfg["sources"]["toolace"]
    xlam_cfg = cfg["sources"]["xlam_60k"]
    apigen_cfg = cfg["sources"]["apigen_mt_5k"]
    smol_cfg = cfg["sources"]["smoltalk_irrelevant"]

    toolace_path = ROOT / toolace_cfg["raw_path"]
    xlam_path = ROOT / xlam_cfg["raw_path"]
    apigen_path = ROOT / apigen_cfg["raw_path"]
    smol_dir = ROOT / smol_cfg["raw_dir"]

    # 1. Load & clean ToolACE (split single_or_parallel explicitly into parallel & single)
    toolace_raw_recs, toolace_parse_stats = load_toolace(
        toolace_path,
        strip_policy_prompt=bool(toolace_cfg.get("strip_policy_prompt", True)),
        split_no_call_subtypes=bool(toolace_cfg.get("split_no_call_subtypes", True)),
    )
    for rec in toolace_raw_recs:
        if rec["meta"]["bucket"] == "toolace_single_or_parallel":
            rec["meta"]["bucket"] = (
                "toolace_parallel" if rec["meta"]["is_parallel"] else "toolace_single"
            )

    # 2. Load xLAM-60k (require >=2 candidate tools for xlam_single)
    xlam_raw_recs, xlam_parse_stats = load_xlam_60k(xlam_path)
    if xlam_cfg.get("require_multi_candidate_single", True):
        xlam_raw_recs = [
            r
            for r in xlam_raw_recs
            if r["meta"]["bucket"] != "xlam_single" or r["meta"]["num_candidate_tools"] >= 2
        ]

    # 3. Load APIGen-MT-5k with strict quality & tau2-bench exclusion
    tau2_excluded = _load_tau2_excluded_turns(ROOT / apigen_cfg["tau2_domains_dir"])
    apigen_raw_recs, apigen_parse_stats = load_apigen_mt_5k(
        apigen_path,
        excluded_first_turns=tau2_excluded,
        require_final_nl=bool(apigen_cfg.get("require_final_nl", True)),
    )

    priority = {
        "apigen_multi_turn": 0,
        "toolace_multi_turn": 1,
        "toolace_irrelevant_refusal": 2,
        "toolace_missing_param": 3,
        "toolace_parallel": 4,
        "toolace_single": 5,
    }
    core_recs = sorted(
        apigen_raw_recs + toolace_raw_recs,
        key=lambda r: priority[r["meta"]["bucket"]],
    )

    seen_sigs: set[str] = set()
    core_dedup, core_dups = _deduplicate(core_recs, seen_sigs)
    xlam_dedup, xlam_dups = _deduplicate(xlam_raw_recs, seen_sigs)

    # 4. Build SmolTalk irrelevant tools (start from pre-approved, top up with strict semantic guard)
    distractor_index = DistractorToolIndex(core_dedup + xlam_dedup)
    preapproved_smol = _load_preapproved_smoltalk(ROOT / smol_cfg["approved_clean_dir"])
    min_asst_chars = int(smol_cfg.get("min_assistant_chars", 12))
    preapproved_smol, preapproved_dropped = _filter_smoltalk_quality(
        preapproved_smol, min_asst_chars
    )

    total_smol_needed: dict[str, int] = {}
    for sub in set(smol_cfg["train_subset_targets"]) | set(smol_cfg["val_subset_targets"]):
        total_smol_needed[sub] = int(smol_cfg["train_subset_targets"].get(sub, 0)) + int(
            smol_cfg["val_subset_targets"].get(sub, 0)
        )

    approved_by_sub: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_task_keys = {_task_key(r) for r in preapproved_smol}
    for rec in preapproved_smol:
        sub = rec["meta"]["source"].split(":", 1)[1]
        approved_by_sub[sub].append(rec)

    topup_targets: dict[str, int] = {}
    for sub, need_n in sorted(total_smol_needed.items()):
        have_n = len(approved_by_sub.get(sub, []))
        if have_n < need_n:
            # Oversample 2x to survive live-query / min-char / dedup filtering
            topup_targets[sub] = (need_n - have_n) * 2 + 40

    smol_topup_recs, smol_parse_stats = (
        load_smoltalk_irrelevant_records(
            smoltalk_dir=smol_dir,
            subset_targets=topup_targets,
            distractor_index=distractor_index,
            seed=seed,
            max_total_chars=int(smol_cfg.get("max_total_chars", 3000)),
        )
        if topup_targets
        else ([], {})
    )
    smol_topup_recs, topup_quality_dropped = _filter_smoltalk_quality(
        smol_topup_recs, min_asst_chars
    )
    for r in smol_topup_recs:
        r["meta"]["curated_origin"] = "v2_strict_guarded"

    smol_combined: list[dict[str, Any]] = []
    for sub, need_n in sorted(total_smol_needed.items()):
        app = list(approved_by_sub.get(sub, []))
        rng.shuffle(app)
        chosen = app[:need_n]
        if len(chosen) < need_n:
            extra = [
                r
                for r in smol_topup_recs
                if r["meta"]["source"] == f"smoltalk:{sub}"
                and _task_key(r) not in seen_task_keys
            ]
            rng.shuffle(extra)
            for r in extra:
                tk = _task_key(r)
                if tk in seen_task_keys:
                    continue
                seen_task_keys.add(tk)
                chosen.append(r)
                if len(chosen) == need_n:
                    break
        smol_combined.extend(chosen)

    smol_dedup, smol_dups = _deduplicate(smol_combined, seen_sigs)

    # 5. Tokenize & filter by length (apigen_mt uses tighter 7500 ceiling for clean packing)
    buckets_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in core_dedup + xlam_dedup + smol_dedup:
        buckets_raw[rec["meta"]["bucket"]].append(rec)

    apigen_max_len = int(apigen_cfg.get("max_seq_length", max_seq_length))
    buckets: dict[str, list[dict[str, Any]]] = {}
    filter_stats: dict[str, Any] = {}
    for bucket_name, items in sorted(buckets_raw.items()):
        limit = apigen_max_len if bucket_name == "apigen_multi_turn" else max_seq_length
        filtered, b_stats = _filter_by_length(
            items, tokenizer, patched_template, limit, verify_every=100
        )
        buckets[bucket_name] = filtered
        filter_stats[bucket_name] = b_stats

    # 6. Hold out stratified zero-leakage validation split FIRST with 3-key task-group isolation
    val_counts_cfg: dict[str, int] = {
        **{k: int(v) for k, v in toolace_cfg["val_counts"].items()},
        **{k: int(v) for k, v in xlam_cfg["val_counts"].items()},
        "apigen_multi_turn": int(apigen_cfg["val_count"]),
    }

    val_records: list[dict[str, Any]] = []
    heldout_keys: set[tuple[Any, ...]] = set()
    train_pool: dict[str, list[dict[str, Any]]] = {}

    for bucket_name in sorted(buckets):
        items = list(buckets[bucket_name])
        if bucket_name == "apigen_multi_turn":
            val_slice, rest = _sample_balanced_apigen(
                items, val_counts_cfg["apigen_multi_turn"], rng
            )
        elif bucket_name == "smoltalk_irrelevant_tools":
            by_sub: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rec in items:
                sub = rec["meta"]["source"].split(":", 1)[1]
                by_sub[sub].append(rec)
            val_slice = []
            rest = []
            for sub, n_val_sub in sorted(smol_cfg["val_subset_targets"].items()):
                sub_items = list(by_sub.get(sub, []))
                rng.shuffle(sub_items)
                val_slice.extend(sub_items[: int(n_val_sub)])
                rest.extend(sub_items[int(n_val_sub) :])
        else:
            rng.shuffle(items)
            n_val = val_counts_cfg[bucket_name]
            val_slice = items[:n_val]
            rest = items[n_val:]

        for idx, rec in enumerate(val_slice):
            copy_rec = json.loads(json.dumps(rec, ensure_ascii=False))
            copy_rec["meta"]["split"] = "val"
            copy_rec["meta"]["sample_id"] = f"sft_v2_val_{bucket_name}_{idx:05d}"
            val_records.append(copy_rec)
            heldout_keys.update(_group_keys(copy_rec))
        train_pool[bucket_name] = [r for r in rest if not (_group_keys(r) & heldout_keys)]

    rng.shuffle(val_records)

    # 7. Sample exact curated train mix (zero upsampling!)
    train_counts_cfg: dict[str, int] = {
        **{k: int(v) for k, v in toolace_cfg["train_counts"].items()},
        **{k: int(v) for k, v in xlam_cfg["train_counts"].items()},
        "apigen_multi_turn": int(apigen_cfg["train_count"]),
    }

    train_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for bucket_name in sorted(train_pool):
        pool = list(train_pool[bucket_name])
        if bucket_name == "apigen_multi_turn":
            chosen, _ = _sample_balanced_apigen(
                pool, train_counts_cfg["apigen_multi_turn"], rng
            )
        elif bucket_name == "smoltalk_irrelevant_tools":
            by_sub_t: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rec in pool:
                sub = rec["meta"]["source"].split(":", 1)[1]
                by_sub_t[sub].append(rec)
            chosen = []
            for sub, n_tr_sub in sorted(smol_cfg["train_subset_targets"].items()):
                sub_items = list(by_sub_t.get(sub, []))
                rng.shuffle(sub_items)
                chosen.extend(sub_items[: int(n_tr_sub)])
        else:
            rng.shuffle(pool)
            chosen = pool[: train_counts_cfg[bucket_name]]

        train_by_bucket[bucket_name] = [
            json.loads(json.dumps(r, ensure_ascii=False)) for r in chosen
        ]

    train_records: list[dict[str, Any]] = []
    for bucket_name in sorted(train_by_bucket):
        for idx, rec in enumerate(train_by_bucket[bucket_name]):
            rec["meta"]["split"] = "train"
            rec["meta"]["sample_id"] = f"sft_v2_{bucket_name}_{idx:05d}"
            train_records.append(rec)
    rng.shuffle(train_records)

    # 8. Strict post-build audit: val leakage == 0 and eval-inspection overlap == 0
    val_keys_check = set()
    for r in val_records:
        val_keys_check.update(_group_keys(r))
    train_val_leakage = sum(1 for r in train_records if _group_keys(r) & val_keys_check)
    if train_val_leakage:
        raise AssertionError(f"Detected {train_val_leakage} train/val leakage items!")

    eval_turns = _load_eval_user_turns(
        [ROOT / p for p in cfg.get("eval_inspection_dirs", [])]
    ) | tau2_excluded
    eval_overlap = sum(
        1
        for r in train_records + val_records
        if any(_norm_text(m["content"]) in eval_turns for m in r["messages"] if m["role"] == "user")
    )

    # 9. Write artifacts
    tpl_path = out_dir / "qwen35_chat_template_train.jinja"
    tpl_path.write_text(patched_template, encoding="utf-8")

    train_sha = _write_jsonl(out_dir / "train.jsonl", train_records)
    val_sha = _write_jsonl(out_dir / "val.jsonl", val_records)
    _render_bucket_samples(
        out_dir / "samples", train_by_bucket, tokenizer, patched_template
    )

    manifest = {
        "version": cfg["version"],
        "seed": seed,
        "max_seq_length": max_seq_length,
        "apigen_max_seq_length": apigen_max_len,
        "tokenizer_model_id": cfg["tokenizer"]["model_id"],
        "distractor_tool_pool_size": len(distractor_index.tools),
        "raw_sources": {
            "toolace": {
                "path": str(toolace_cfg["raw_path"]),
                "sha256": _sha256_file(toolace_path),
                "parse_stats": toolace_parse_stats.to_dict(),
            },
            "xlam_60k": {
                "path": str(xlam_cfg["raw_path"]),
                "sha256": _sha256_file(xlam_path),
                "parse_stats": xlam_parse_stats.to_dict(),
                "dedup_dropped": xlam_dups,
            },
            "apigen_mt_5k": {
                "path": str(apigen_cfg["raw_path"]),
                "sha256": _sha256_file(apigen_path),
                "tau2_excluded_turns_count": len(tau2_excluded),
                "parse_stats": apigen_parse_stats.to_dict(),
            },
            "smoltalk_irrelevant": {
                "preapproved_loaded": len(preapproved_smol),
                "preapproved_quality_dropped": preapproved_dropped,
                "topup_subsets": smol_parse_stats,
                "topup_quality_dropped": topup_quality_dropped,
                "dedup_dropped": smol_dups,
            },
        },
        "core_dedup_dropped": core_dups,
        "clean_pool_after_dedup_and_length_filter": {
            bucket: len(items) for bucket, items in sorted(buckets.items())
        },
        "length_filter_stats": filter_stats,
        "leakage_audit": {
            "train_val_group_overlap": train_val_leakage,
            "eval_user_turn_overlap": eval_overlap,
        },
        "outputs": {
            "train.jsonl": {"sha256": train_sha, **_split_summary(train_records)},
            "val.jsonl": {"sha256": val_sha, **_split_summary(val_records)},
            "qwen35_chat_template_train.jinja": {"sha256": _sha256_file(tpl_path)},
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data/sft_v2_curated.yaml",
        help="Path to SFT-v2-curated dataset config YAML",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Optional override path to local Qwen3.5 tokenizer snapshot",
    )
    args = parser.parse_args()
    manifest = build_sft_v2_curated(args.config, args.tokenizer)
    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
