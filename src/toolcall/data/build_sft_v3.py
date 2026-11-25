"""Build `SFT-v3` (~4.6k strictly validated, zero-leakage, balanced tool-calling dataset)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer
import yaml

from .build_sft_v1 import (
    _deduplicate,
    _filter_by_length,
    _sha256_file,
    _write_jsonl,
)
from .build_sft_v1r1 import _render_bucket_samples
from .build_sft_v2_curated import (
    _group_keys,
    _load_eval_user_turns,
    _load_tau2_excluded_turns,
    _norm_text,
    _sample_balanced_apigen,
    _split_summary,
)
from .parsers import _NL_PSEUDO_CALL_RE, load_apigen_mt_5k, load_toolace, load_xlam_60k
from .schema import validate_record
from .template import patch_qwen35_chat_template

ROOT = Path(__file__).resolve().parents[3]


def _review_digest(record: dict[str, Any]) -> str:
    payload = json.dumps(
        [record["messages"], record["tools"]], ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_tau2_task_entities(tau2_domains_dir: Path) -> set[str]:
    entities: set[str] = set()
    if not tau2_domains_dir.exists():
        return entities
    for domain in ("retail", "airline"):
        tasks_file = tau2_domains_dir / domain / "tasks.json"
        if not tasks_file.exists():
            continue
        txt = tasks_file.read_text(encoding="utf-8")
        entities.update(re.findall(r"#W\d{6,}", txt))
        entities.update(re.findall(r"[a-z]+_[a-z]+_\d{4}", txt))
        entities.update(re.findall(r"[a-z]+\.[a-z]+\d+@[a-z.]+", txt))
        if domain == "airline":
            entities.update(re.findall(r"\b[A-Z0-9]{6}\b", txt))
    return entities


def _load_historical_val_keys(hist_val_path: Path) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    if not hist_val_path.exists():
        return keys
    with hist_val_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            keys.update(_group_keys(rec))
    return keys


def _load_allowlisted_smoltalk(
    review_path: Path,
    approved_clean_dir: Path,
    smoltalk_dir: Path,
    tool_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    approved_entries = review["approved"]

    # Index pre-approved records from sft_v1r1_clean by review_digest
    pre_by_sha: dict[str, dict[str, Any]] = {}
    for fname in ("val.jsonl", "smoke_v1r1.jsonl", "train.jsonl"):
        fpath = approved_clean_dir / fname
        if not fpath.exists():
            continue
        with fpath.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["meta"]["bucket"] == "smoltalk_irrelevant_tools":
                    pre_by_sha.setdefault(_review_digest(rec), rec)

    # Index cleaned tool schemas by exact schema SHA-256 (with name fallback)
    tools_by_sha: dict[str, dict[str, Any]] = {}
    tools_by_name: dict[str, dict[str, Any]] = {}
    for rec in tool_candidates:
        for t in rec.get("tools") or []:
            t_sha = hashlib.sha256(
                json.dumps(t, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            tools_by_sha.setdefault(t_sha, t)
            tools_by_name.setdefault(t["function"]["name"], t)

    # Lazy-load needed raw SmolTalk subsets by row index
    raw_cache: dict[str, list[dict[str, Any]]] = {}

    def get_raw_row(subset_name: str, row_idx: int) -> dict[str, Any]:
        if subset_name not in raw_cache:
            pf = sorted((smoltalk_dir / subset_name).glob("train-*.parquet"))
            rows: list[dict[str, Any]] = []
            for p in pf:
                rows.extend(pq.ParquetFile(p).read().to_pylist())
            raw_cache[subset_name] = rows
        return raw_cache[subset_name][row_idx]

    records: list[dict[str, Any]] = []
    for entry in approved_entries:
        sha = entry["sha256"]
        if sha in pre_by_sha:
            rec = json.loads(json.dumps(pre_by_sha[sha], ensure_ascii=False))
            rec["meta"]["curated_origin"] = entry["origin"]
            records.append(rec)
            continue

        subset_name = entry["source"].split(":", 1)[1]
        row_idx = int(str(entry["source_id"]).split(":", 1)[1])
        raw_row = get_raw_row(subset_name, row_idx)
        clean_msgs = [
            {"role": str(m["role"]), "content": str(m["content"]).strip()}
            for m in raw_row["messages"]
        ]
        tool_hashes = entry.get("tool_hashes") or []
        if tool_hashes:
            tools = [
                json.loads(json.dumps(tools_by_sha[th], ensure_ascii=False))
                for th in tool_hashes
            ]
        else:
            tools = [
                json.loads(json.dumps(tools_by_name[n], ensure_ascii=False))
                for n in entry["tool_names"]
            ]
        rec = {
            "messages": clean_msgs,
            "tools": tools,
            "meta": {
                "source": entry["source"],
                "source_id": entry["source_id"],
                "bucket": "smoltalk_irrelevant_tools",
                "curated_origin": entry["origin"],
                "num_turns": sum(1 for m in clean_msgs if m["role"] == "user"),
                "num_user_turns": sum(1 for m in clean_msgs if m["role"] == "user"),
                "num_tool_messages": 0,
                "num_tool_calls": 0,
                "num_candidate_tools": len(tools),
                "has_tool_response": False,
                "is_parallel": False,
                "is_refusal": True,
            },
        }
        actual_sha = _review_digest(rec)
        if actual_sha != sha:
            raise AssertionError(
                f"SHA mismatch reconstructing allowlisted SmolTalk {entry['source_id']}: {actual_sha} != {sha}"
            )
        records.append(rec)

    return records


def _assert_final_invariants(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    hist_val_keys: set[tuple[Any, ...]],
    eval_turns: set[str],
    tau2_entities: set[str],
) -> dict[str, int]:
    val_keys = set()
    for r in val_records:
        val_keys.update(_group_keys(r))

    train_val_overlap = sum(1 for r in train_records if _group_keys(r) & val_keys)
    if train_val_overlap:
        raise AssertionError(f"train/val 3-key overlap detected: {train_val_overlap}")

    hist_overlap = sum(1 for r in train_records if _group_keys(r) & hist_val_keys)
    if hist_overlap:
        raise AssertionError(f"train overlaps historical val set: {hist_overlap}")

    eval_turn_overlap = 0
    tau2_entity_overlap = 0
    for rec in train_records + val_records:
        ok, reason = validate_record(
            rec, strict_arg_types=True, reject_duplicate_tools=True
        )
        if not ok:
            raise AssertionError(
                f"Strict validation failed for {rec['meta'].get('sample_id')}: {reason}"
            )
        if not rec["tools"]:
            raise AssertionError(f"Empty tools in {rec['meta'].get('sample_id')}")
        if any(t["function"]["name"] == "think" for t in rec["tools"]):
            raise AssertionError(f"Residual `think` tool in {rec['meta'].get('sample_id')}")

        sys_msg = rec["messages"][0]
        if sys_msg["role"] == "system" and "If none of the function can be used" in sys_msg["content"]:
            raise AssertionError(f"ToolACE policy leaked in {rec['meta'].get('sample_id')}")

        for m in rec["messages"]:
            if m["role"] == "assistant" and not m.get("tool_calls"):
                content = str(m.get("content") or "")
                if _NL_PSEUDO_CALL_RE.search(content):
                    raise AssertionError(
                        f"Pseudo tool call leaked in NL turn of {rec['meta'].get('sample_id')}"
                    )
            if m["role"] == "user" and _norm_text(m["content"]) in eval_turns:
                eval_turn_overlap += 1

        if rec["meta"]["bucket"] == "apigen_multi_turn":
            blob = json.dumps(rec["messages"], ensure_ascii=False)
            if any(ent in blob for ent in tau2_entities):
                tau2_entity_overlap += 1

    if eval_turn_overlap or tau2_entity_overlap:
        raise AssertionError(
            f"Eval contamination detected: eval_turn_overlap={eval_turn_overlap}, "
            f"tau2_entity_overlap={tau2_entity_overlap}"
        )

    return {
        "train_val_group_overlap": 0,
        "train_historical_val_overlap": 0,
        "eval_user_turn_overlap": 0,
        "tau2_task_entity_overlap": 0,
    }


def build_sft_v3(
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

    # 1. Load with strict validation & zero-leakage filters
    toolace_raw_recs, toolace_parse_stats = load_toolace(
        toolace_path,
        strip_policy_prompt=bool(toolace_cfg.get("strip_policy_prompt", True)),
        split_no_call_subtypes=bool(toolace_cfg.get("split_no_call_subtypes", True)),
        strict_validation=bool(toolace_cfg.get("strict_validation", True)),
    )
    for rec in toolace_raw_recs:
        if rec["meta"]["bucket"] == "toolace_single_or_parallel":
            rec["meta"]["bucket"] = (
                "toolace_parallel" if rec["meta"]["is_parallel"] else "toolace_single"
            )

    xlam_raw_recs, xlam_parse_stats = load_xlam_60k(
        xlam_path,
        strict_validation=bool(xlam_cfg.get("strict_validation", True)),
    )
    if xlam_cfg.get("require_multi_candidate_single", True):
        xlam_raw_recs = [
            r
            for r in xlam_raw_recs
            if r["meta"]["bucket"] != "xlam_single" or r["meta"]["num_candidate_tools"] >= 2
        ]

    tau2_excluded_turns = _load_tau2_excluded_turns(ROOT / apigen_cfg["tau2_domains_dir"])
    tau2_entities = _load_tau2_task_entities(ROOT / apigen_cfg["tau2_domains_dir"])
    apigen_raw_recs, apigen_parse_stats = load_apigen_mt_5k(
        apigen_path,
        excluded_first_turns=tau2_excluded_turns,
        excluded_entities=tau2_entities,
        require_final_nl=bool(apigen_cfg.get("require_final_nl", True)),
        strict_validation=bool(apigen_cfg.get("strict_validation", True)),
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

    # 2. Load 100% SHA-256 allowlisted SmolTalk irrelevance records
    smol_raw_recs = _load_allowlisted_smoltalk(
        review_path=ROOT / smol_cfg["review_path"],
        approved_clean_dir=ROOT / smol_cfg["approved_clean_dir"],
        smoltalk_dir=smol_dir,
        tool_candidates=core_dedup + xlam_dedup,
    )
    smol_dedup, smol_dups = _deduplicate(smol_raw_recs, seen_sigs)

    # 3. Tokenize & filter by length
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

    # 4. Isolate historical val keys from training pools & hold out new val FIRST
    hist_val_keys = _load_historical_val_keys(ROOT / cfg["historical_val_path"])
    val_counts_cfg: dict[str, int] = {
        **{k: int(v) for k, v in toolace_cfg["val_counts"].items()},
        **{k: int(v) for k, v in xlam_cfg["val_counts"].items()},
        "apigen_multi_turn": int(apigen_cfg["val_count"]),
        "smoltalk_irrelevant_tools": int(smol_cfg["val_count"]),
    }

    val_records: list[dict[str, Any]] = []
    heldout_keys: set[tuple[Any, ...]] = set(hist_val_keys)
    train_pool: dict[str, list[dict[str, Any]]] = {}

    for bucket_name in sorted(buckets):
        items = list(buckets[bucket_name])
        if bucket_name == "apigen_multi_turn":
            val_slice, rest = _sample_balanced_apigen(
                items, val_counts_cfg["apigen_multi_turn"], rng
            )
        elif bucket_name == "smoltalk_irrelevant_tools":
            # Stratify SmolTalk val across sources proportionally
            n_val = val_counts_cfg["smoltalk_irrelevant_tools"]
            by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rec in items:
                by_src[rec["meta"]["source"]].append(rec)
            val_slice = []
            rest = []
            total_smol = len(items)
            for src_name in sorted(by_src):
                src_items = list(by_src[src_name])
                rng.shuffle(src_items)
                take = max(1, round(n_val * len(src_items) / max(1, total_smol)))
                val_slice.extend(src_items[:take])
                rest.extend(src_items[take:])
            if len(val_slice) > n_val:
                rest.extend(val_slice[n_val:])
                val_slice = val_slice[:n_val]
            elif len(val_slice) < n_val:
                need = n_val - len(val_slice)
                val_slice.extend(rest[:need])
                rest = rest[need:]
        else:
            # Prefer picking val from items NOT already in hist_val_keys if plenty,
            # or let val include clean items while strictly excluding hist_val_keys from train.
            non_hist = [r for r in items if not (_group_keys(r) & hist_val_keys)]
            in_hist = [r for r in items if _group_keys(r) & hist_val_keys]
            rng.shuffle(non_hist)
            rng.shuffle(in_hist)
            ordered = non_hist + in_hist
            n_val = val_counts_cfg[bucket_name]
            val_slice = ordered[:n_val]
            rest = ordered[n_val:]

        for idx, rec in enumerate(val_slice):
            copy_rec = json.loads(json.dumps(rec, ensure_ascii=False))
            copy_rec["meta"]["split"] = "val"
            copy_rec["meta"]["sample_id"] = f"sft_v3_val_{bucket_name}_{idx:05d}"
            val_records.append(copy_rec)
            heldout_keys.update(_group_keys(copy_rec))

        train_pool[bucket_name] = [r for r in rest if not (_group_keys(r) & heldout_keys)]

    rng.shuffle(val_records)

    # 5. Sample exact curated train mix (zero upsampling!)
    train_counts_cfg: dict[str, int] = {
        **{k: int(v) for k, v in toolace_cfg["train_counts"].items()},
        **{k: int(v) for k, v in xlam_cfg["train_counts"].items()},
        "apigen_multi_turn": int(apigen_cfg["train_count"]),
        "smoltalk_irrelevant_tools": int(smol_cfg["train_count"]),
    }

    train_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for bucket_name in sorted(train_pool):
        pool = list(train_pool[bucket_name])
        target_n = train_counts_cfg[bucket_name]
        if len(pool) < target_n:
            raise RuntimeError(
                f"Insufficient clean records in {bucket_name}: have {len(pool)}, need {target_n}"
            )
        if bucket_name == "apigen_multi_turn":
            chosen, _ = _sample_balanced_apigen(pool, target_n, rng)
        elif bucket_name == "toolace_multi_turn":
            # Prioritize multi-user-turn NL-ending trajectories first, then remaining clean ones
            tier1 = [
                r
                for r in pool
                if r["meta"]["num_user_turns"] >= 2 and not r["messages"][-1].get("tool_calls")
            ]
            tier2 = [
                r
                for r in pool
                if not (
                    r["meta"]["num_user_turns"] >= 2 and not r["messages"][-1].get("tool_calls")
                )
            ]
            rng.shuffle(tier1)
            rng.shuffle(tier2)
            chosen = (tier1 + tier2)[:target_n]
        else:
            rng.shuffle(pool)
            chosen = pool[:target_n]

        train_by_bucket[bucket_name] = [
            json.loads(json.dumps(r, ensure_ascii=False)) for r in chosen
        ]

    train_records: list[dict[str, Any]] = []
    for bucket_name in sorted(train_by_bucket):
        for idx, rec in enumerate(train_by_bucket[bucket_name]):
            rec["meta"]["split"] = "train"
            rec["meta"]["sample_id"] = f"sft_v3_{bucket_name}_{idx:05d}"
            train_records.append(rec)
    rng.shuffle(train_records)

    # 6. Hard post-build invariant checks (blocks build on any violation!)
    eval_turns = _load_eval_user_turns(
        [ROOT / p for p in cfg.get("eval_inspection_dirs", [])]
    ) | tau2_excluded_turns
    leakage_report = _assert_final_invariants(
        train_records, val_records, hist_val_keys, eval_turns, tau2_entities
    )

    # 7. Write outputs + traceable audit ledger
    tpl_path = out_dir / "qwen35_chat_template_train.jinja"
    tpl_path.write_text(patched_template, encoding="utf-8")

    train_sha = _write_jsonl(out_dir / "train.jsonl", train_records)
    val_sha = _write_jsonl(out_dir / "val.jsonl", val_records)
    _render_bucket_samples(
        out_dir / "samples", train_by_bucket, tokenizer, patched_template
    )

    ledger_rows = [
        {
            "sample_id": r["meta"]["sample_id"],
            "split": r["meta"]["split"],
            "bucket": r["meta"]["bucket"],
            "source": r["meta"]["source"],
            "source_id": r["meta"].get("source_id"),
            "curated_origin": r["meta"].get("curated_origin"),
            "domain": r["meta"].get("domain"),
            "tool_names": [t["function"]["name"] for t in r["tools"]],
            "num_tokens": r["meta"]["num_tokens"],
            "num_assistant_tokens": r["meta"]["num_assistant_tokens"],
            "content_sha256": _review_digest(r),
        }
        for r in train_records + val_records
    ]
    ledger_sha = _write_jsonl(out_dir / "audit_ledger.jsonl", ledger_rows)

    manifest = {
        "version": cfg["version"],
        "seed": seed,
        "max_seq_length": max_seq_length,
        "apigen_max_seq_length": apigen_max_len,
        "tokenizer_model_id": cfg["tokenizer"]["model_id"],
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
                "tau2_task_entities_count": len(tau2_entities),
                "parse_stats": apigen_parse_stats.to_dict(),
            },
            "smoltalk_irrelevant": {
                "review_path": str(smol_cfg["review_path"]),
                "review_sha256": _sha256_file(ROOT / smol_cfg["review_path"]),
                "allowlisted_loaded": len(smol_raw_recs),
                "dedup_dropped": smol_dups,
            },
        },
        "core_dedup_dropped": core_dups,
        "clean_pool_after_dedup_and_length_filter": {
            bucket: len(items) for bucket, items in sorted(buckets.items())
        },
        "length_filter_stats": filter_stats,
        "leakage_audit": leakage_report,
        "outputs": {
            "train.jsonl": {"sha256": train_sha, **_split_summary(train_records)},
            "val.jsonl": {"sha256": val_sha, **_split_summary(val_records)},
            "audit_ledger.jsonl": {"sha256": ledger_sha, "total_rows": len(ledger_rows)},
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
        default=ROOT / "configs/data/sft_v3.yaml",
        help="Path to SFT-v3 dataset config YAML",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Optional override path to local Qwen3.5 tokenizer snapshot",
    )
    args = parser.parse_args()
    manifest = build_sft_v3(args.config, args.tokenizer)
    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
