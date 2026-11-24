"""Build the curated, deduplicated, Qwen3.5-verified `SFT-v1` dataset (`xlam-60k + toolace`)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import random
from typing import Any

from transformers import AutoTokenizer
import yaml

from .parsers import load_toolace, load_xlam_60k
from .schema import dedup_signature
from .template import (
    format_masked_inspection,
    patch_qwen35_chat_template,
    render_and_mask,
)

ROOT = Path(__file__).resolve().parents[3]


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, round((len(sorted_vals) - 1) * q)))
    return sorted_vals[idx]


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0}
    s = sorted(lengths)
    return {
        "count": len(s),
        "min": s[0],
        "p50": _percentile(s, 0.50),
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "max": s[-1],
        "mean": round(sum(s) / len(s), 1),
    }


def _deduplicate(
    records: list[dict[str, Any]],
    seen: set[str],
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dups = 0
    for rec in records:
        sig = dedup_signature(rec)
        if sig in seen:
            dups += 1
            continue
        seen.add(sig)
        kept.append(rec)
    return kept, dups


def _filter_by_length(
    records: list[dict[str, Any]],
    tokenizer: Any,
    patched_template: str,
    max_seq_length: int,
    verify_every: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped_too_long = 0
    dropped_mask_err = 0
    token_lengths: list[int] = []
    asst_lengths: list[int] = []

    for i, rec in enumerate(records):
        try:
            rendered = render_and_mask(
                tokenizer,
                rec,
                patched_template,
                verify_equivalence=(i % verify_every == 0),
            )
        except Exception:
            dropped_mask_err += 1
            continue

        num_toks = rendered["num_tokens"]
        if num_toks > max_seq_length:
            dropped_too_long += 1
            continue

        rec["meta"]["num_tokens"] = num_toks
        rec["meta"]["num_assistant_tokens"] = rendered["num_assistant_tokens"]
        token_lengths.append(num_toks)
        asst_lengths.append(rendered["num_assistant_tokens"])
        kept.append(rec)

    stats = {
        "dropped_too_long": dropped_too_long,
        "dropped_mask_err": dropped_mask_err,
        "total_tokens": _length_summary(token_lengths),
        "assistant_tokens": _length_summary(asst_lengths),
    }
    return kept, stats


def _sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return file_digest(fh, "sha256").hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return _sha256_file(path)


def build_sft_v1(config_path: Path, tokenizer_override: Path | None = None) -> dict[str, Any]:
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

    # 1. Load raw sources
    toolace_path = ROOT / cfg["sources"]["toolace"]["raw_path"]
    xlam_path = ROOT / cfg["sources"]["xlam_60k"]["raw_path"]

    toolace_raw_recs, toolace_parse_stats = load_toolace(toolace_path)
    xlam_raw_recs, xlam_parse_stats = load_xlam_60k(xlam_path)

    # Priority order during dedup: multi_turn > no_call > single_or_parallel > xlam
    priority = {
        "toolace_multi_turn": 0,
        "toolace_no_call": 1,
        "toolace_single_or_parallel": 2,
    }
    toolace_raw_recs.sort(key=lambda r: priority[r["meta"]["bucket"]])

    seen_sigs: set[str] = set()
    toolace_dedup, toolace_dups = _deduplicate(toolace_raw_recs, seen_sigs)
    xlam_dedup, xlam_dups = _deduplicate(xlam_raw_recs, seen_sigs)

    # 2. Group by bucket and run exact Qwen3.5 tokenization + length/mask check
    buckets_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in toolace_dedup + xlam_dedup:
        buckets_raw[rec["meta"]["bucket"]].append(rec)

    # For xlam buckets, sample slightly before full tokenization? No, tokenizing all ~70k is fast
    # (~15-20s) and yields exact pool statistics in manifest.json.
    buckets: dict[str, list[dict[str, Any]]] = {}
    filter_stats: dict[str, Any] = {}
    for bucket_name, items in sorted(buckets_raw.items()):
        filtered, b_stats = _filter_by_length(
            items, tokenizer, patched_template, max_seq_length, verify_every=100
        )
        buckets[bucket_name] = filtered
        filter_stats[bucket_name] = b_stats

    # 3. Hold out zero-leakage stratified validation split BEFORE any upsampling
    val_counts_cfg = {
        **cfg["sources"]["toolace"]["val_counts"],
        **cfg["sources"]["xlam_60k"]["val_counts"],
    }
    val_records: list[dict[str, Any]] = []
    train_pool: dict[str, list[dict[str, Any]]] = {}

    for bucket_name, items in buckets.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = int(val_counts_cfg.get(bucket_name, 0))
        val_slice = shuffled[:n_val]
        rest = shuffled[n_val:]
        for idx, rec in enumerate(val_slice):
            rec = json.loads(json.dumps(rec, ensure_ascii=False))
            rec["meta"]["split"] = "val"
            rec["meta"]["sample_id"] = f"sft_v1_val_{bucket_name}_{idx:05d}"
            val_records.append(rec)
        train_pool[bucket_name] = rest

    rng.shuffle(val_records)

    # 4. Construct SFT-v1 training mix
    upsample = int(cfg["sources"]["toolace"]["multi_turn_upsample"])
    train_by_bucket: dict[str, list[dict[str, Any]]] = {}

    # 4a. ToolACE multi-turn (keep all remaining, upsample 3x)
    mt_items = train_pool["toolace_multi_turn"]
    mt_expanded: list[dict[str, Any]] = []
    for rep in range(upsample):
        for rec in mt_items:
            copy_rec = json.loads(json.dumps(rec, ensure_ascii=False))
            copy_rec["meta"]["upsample_copy"] = rep
            mt_expanded.append(copy_rec)
    train_by_bucket["toolace_multi_turn"] = mt_expanded

    # 4b. ToolACE no-call (keep all remaining)
    train_by_bucket["toolace_no_call"] = [
        json.loads(json.dumps(r, ensure_ascii=False))
        for r in train_pool["toolace_no_call"]
    ]

    # 4c. ToolACE single/parallel (keep all remaining)
    train_by_bucket["toolace_single_or_parallel"] = [
        json.loads(json.dumps(r, ensure_ascii=False))
        for r in train_pool["toolace_single_or_parallel"]
    ]

    # 4d. xLAM parallel (sample target)
    xlam_par_target = int(cfg["sources"]["xlam_60k"]["parallel_sample"])
    xlam_par_pool = list(train_pool["xlam_parallel"])
    rng.shuffle(xlam_par_pool)
    train_by_bucket["xlam_parallel"] = [
        json.loads(json.dumps(r, ensure_ascii=False))
        for r in xlam_par_pool[:xlam_par_target]
    ]

    # 4e. xLAM single (prioritize multi-candidate tool lists len(tools) >= 2)
    xlam_sin_target = int(cfg["sources"]["xlam_60k"]["single_sample"])
    xlam_sin_pool = list(train_pool["xlam_single"])
    if cfg["sources"]["xlam_60k"].get("prefer_multi_candidate_single", True):
        multi_cand = [r for r in xlam_sin_pool if r["meta"]["num_candidate_tools"] >= 2]
        one_cand = [r for r in xlam_sin_pool if r["meta"]["num_candidate_tools"] < 2]
        rng.shuffle(multi_cand)
        rng.shuffle(one_cand)
        chosen_sin = (multi_cand + one_cand)[:xlam_sin_target]
    else:
        rng.shuffle(xlam_sin_pool)
        chosen_sin = xlam_sin_pool[:xlam_sin_target]
    train_by_bucket["xlam_single"] = [
        json.loads(json.dumps(r, ensure_ascii=False)) for r in chosen_sin
    ]

    # Combine & deterministically shuffle train records
    train_records: list[dict[str, Any]] = []
    for bucket_name in sorted(train_by_bucket):
        for idx, rec in enumerate(train_by_bucket[bucket_name]):
            rec["meta"]["split"] = "train"
            rec["meta"]["sample_id"] = f"sft_v1_{bucket_name}_{idx:05d}"
            train_records.append(rec)
    rng.shuffle(train_records)

    # 5. Construct V0 smoke subset (unique samples drawn from train_pool without duplicates)
    smoke_cfg = cfg["smoke_counts"]
    smoke_records: list[dict[str, Any]] = []
    for bucket_name, target_n in sorted(smoke_cfg.items()):
        unique_pool = list(train_pool[bucket_name])
        rng.shuffle(unique_pool)
        for idx, rec in enumerate(unique_pool[: int(target_n)]):
            copy_rec = json.loads(json.dumps(rec, ensure_ascii=False))
            copy_rec["meta"]["split"] = "smoke_v0"
            copy_rec["meta"]["sample_id"] = f"sft_v0_{bucket_name}_{idx:05d}"
            smoke_records.append(copy_rec)
    rng.shuffle(smoke_records)

    # 6. Write outputs + patched template + human-readable samples
    tpl_path = out_dir / "qwen35_chat_template_train.jinja"
    tpl_path.write_text(patched_template, encoding="utf-8")

    train_sha = _write_jsonl(out_dir / "train.jsonl", train_records)
    val_sha = _write_jsonl(out_dir / "val.jsonl", val_records)
    smoke_sha = _write_jsonl(out_dir / "smoke_v0.jsonl", smoke_records)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for bucket_name in sorted(train_by_bucket):
        sample_rec = next(
            (
                r
                for r in train_by_bucket[bucket_name]
                if (bucket_name != "toolace_single_or_parallel" or r["meta"]["is_parallel"])
            ),
            train_by_bucket[bucket_name][0],
        )
        rendered = render_and_mask(
            tokenizer, sample_rec, patched_template, verify_equivalence=True
        )
        text = (
            f"# Bucket: {bucket_name}\n"
            f"# Sample ID: {sample_rec['meta']['sample_id']}\n"
            f"# Total tokens: {rendered['num_tokens']} | Assistant tokens: {rendered['num_assistant_tokens']}\n\n"
            + format_masked_inspection(
                tokenizer, rendered["input_ids"], rendered["assistant_masks"]
            )
        )
        (samples_dir / f"{bucket_name}.txt").write_text(text, encoding="utf-8")

    # 7. Compute final split distributions & save manifest
    def split_summary(recs: list[dict[str, Any]]) -> dict[str, Any]:
        by_bucket = Counter(r["meta"]["bucket"] for r in recs)
        by_source = Counter(r["meta"]["source"] for r in recs)
        toks = [r["meta"]["num_tokens"] for r in recs]
        asst = [r["meta"]["num_assistant_tokens"] for r in recs]
        parallel_cnt = sum(1 for r in recs if r["meta"]["is_parallel"])
        multi_turn_cnt = sum(1 for r in recs if r["meta"]["has_tool_response"])
        refusal_cnt = sum(1 for r in recs if r["meta"]["is_refusal"])
        multi_cand_cnt = sum(1 for r in recs if r["meta"]["num_candidate_tools"] >= 2)
        return {
            "total_rows": len(recs),
            "by_source": dict(sorted(by_source.items())),
            "by_bucket": dict(sorted(by_bucket.items())),
            "traits": {
                "parallel": parallel_cnt,
                "has_tool_response": multi_turn_cnt,
                "no_call_refusal_or_clarify": refusal_cnt,
                "multi_candidate_tools": multi_cand_cnt,
            },
            "total_tokens": _length_summary(toks),
            "assistant_tokens": _length_summary(asst),
        }

    manifest = {
        "version": cfg["version"],
        "seed": seed,
        "max_seq_length": max_seq_length,
        "tokenizer_model_id": cfg["tokenizer"]["model_id"],
        "raw_sources": {
            "toolace": {
                "path": str(cfg["sources"]["toolace"]["raw_path"]),
                "sha256": _sha256_file(toolace_path),
                "parse_stats": toolace_parse_stats.to_dict(),
                "dedup_dropped": toolace_dups,
            },
            "xlam_60k": {
                "path": str(cfg["sources"]["xlam_60k"]["raw_path"]),
                "sha256": _sha256_file(xlam_path),
                "parse_stats": xlam_parse_stats.to_dict(),
                "dedup_dropped": xlam_dups,
            },
        },
        "clean_pool_after_dedup_and_length_filter": {
            bucket: len(items) for bucket, items in sorted(buckets.items())
        },
        "length_filter_stats": filter_stats,
        "upsample": {"toolace_multi_turn": upsample},
        "outputs": {
            "train.jsonl": {"sha256": train_sha, **split_summary(train_records)},
            "val.jsonl": {"sha256": val_sha, **split_summary(val_records)},
            "smoke_v0.jsonl": {"sha256": smoke_sha, **split_summary(smoke_records)},
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
        default=ROOT / "configs/data/sft_v1.yaml",
        help="Path to SFT-v1 dataset config YAML",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Optional override path to local Qwen3.5 tokenizer snapshot",
    )
    args = parser.parse_args()
    manifest = build_sft_v1(args.config, args.tokenizer)
    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
