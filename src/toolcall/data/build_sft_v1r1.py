"""Build `SFT-v1r1` (`xlam-60k + cleaned toolace + smoltalk_irrelevant_tools`)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

from transformers import AutoTokenizer
import yaml

from .build_sft_v1 import (
    _deduplicate,
    _filter_by_length,
    _length_summary,
    _sha256_file,
    _write_jsonl,
)
from .parsers import load_toolace, load_xlam_60k
from .smoltalk import DistractorToolIndex, load_smoltalk_irrelevant_records
from .template import (
    format_masked_inspection,
    patch_qwen35_chat_template,
    render_and_mask,
)

ROOT = Path(__file__).resolve().parents[3]


def _render_bucket_samples(
    samples_dir: Path,
    train_by_bucket: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    patched_template: str,
) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    for bucket_name in sorted(train_by_bucket):
        items = train_by_bucket[bucket_name]
        if not items:
            continue
        sample_rec = next(
            (
                r
                for r in items
                if (bucket_name != "toolace_single_or_parallel" or r["meta"]["is_parallel"])
            ),
            items[0],
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


def build_sft_v1r1(config_path: Path, tokenizer_override: Path | None = None) -> dict[str, Any]:
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

    # 1. Load & clean ToolACE + xLAM-60k
    toolace_cfg = cfg["sources"]["toolace"]
    xlam_cfg = cfg["sources"]["xlam_60k"]
    smol_cfg = cfg["sources"]["smoltalk_irrelevant"]

    toolace_path = ROOT / toolace_cfg["raw_path"]
    xlam_path = ROOT / xlam_cfg["raw_path"]
    smol_dir = ROOT / smol_cfg["raw_dir"]

    toolace_raw_recs, toolace_parse_stats = load_toolace(
        toolace_path,
        strip_policy_prompt=bool(toolace_cfg.get("strip_policy_prompt", True)),
        split_no_call_subtypes=bool(toolace_cfg.get("split_no_call_subtypes", True)),
    )
    xlam_raw_recs, xlam_parse_stats = load_xlam_60k(xlam_path)

    priority = {
        "toolace_multi_turn": 0,
        "toolace_irrelevant_refusal": 1,
        "toolace_missing_param": 2,
        "toolace_single_or_parallel": 3,
    }
    toolace_raw_recs.sort(key=lambda r: priority[r["meta"]["bucket"]])

    seen_sigs: set[str] = set()
    toolace_dedup, toolace_dups = _deduplicate(toolace_raw_recs, seen_sigs)
    xlam_dedup, xlam_dups = _deduplicate(xlam_raw_recs, seen_sigs)

    # 2. Build DistractorToolIndex from cleaned ToolACE + xLAM tools & generate SmolTalk irrelevance
    distractor_index = DistractorToolIndex(toolace_dedup + xlam_dedup)
    smol_raw_recs, smol_parse_stats = load_smoltalk_irrelevant_records(
        smoltalk_dir=smol_dir,
        subset_targets={k: int(v) for k, v in smol_cfg["pool_targets"].items()},
        distractor_index=distractor_index,
        seed=seed,
        max_total_chars=int(smol_cfg.get("max_total_chars", 3500)),
    )
    smol_dedup, smol_dups = _deduplicate(smol_raw_recs, seen_sigs)

    # 3. Group by bucket and run exact Qwen3.5 tokenization + length/mask check
    buckets_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in toolace_dedup + xlam_dedup + smol_dedup:
        buckets_raw[rec["meta"]["bucket"]].append(rec)

    buckets: dict[str, list[dict[str, Any]]] = {}
    filter_stats: dict[str, Any] = {}
    for bucket_name, items in sorted(buckets_raw.items()):
        filtered, b_stats = _filter_by_length(
            items, tokenizer, patched_template, max_seq_length, verify_every=100
        )
        buckets[bucket_name] = filtered
        filter_stats[bucket_name] = b_stats

    # 4. Hold out zero-leakage stratified validation split BEFORE any upsampling/sampling
    val_counts_cfg: dict[str, int] = {
        **{k: int(v) for k, v in toolace_cfg["val_counts"].items()},
        **{k: int(v) for k, v in xlam_cfg["val_counts"].items()},
        "smoltalk_irrelevant_tools": int(smol_cfg["val_count"]),
    }

    val_records: list[dict[str, Any]] = []
    train_pool: dict[str, list[dict[str, Any]]] = {}

    for bucket_name, items in buckets.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = int(val_counts_cfg.get(bucket_name, 0))
        if bucket_name == "smoltalk_irrelevant_tools":
            # Stratify SmolTalk val across subsets evenly
            by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rec in shuffled:
                by_src[rec["meta"]["source"]].append(rec)
            per_src = max(1, n_val // max(1, len(by_src)))
            val_slice = []
            rest = []
            for src_name in sorted(by_src):
                src_items = by_src[src_name]
                val_slice.extend(src_items[:per_src])
                rest.extend(src_items[per_src:])
            if len(val_slice) < n_val:
                need = n_val - len(val_slice)
                val_slice.extend(rest[:need])
                rest = rest[need:]
            val_slice = val_slice[:n_val]
        else:
            val_slice = shuffled[:n_val]
            rest = shuffled[n_val:]

        for idx, rec in enumerate(val_slice):
            rec = json.loads(json.dumps(rec, ensure_ascii=False))
            rec["meta"]["split"] = "val"
            rec["meta"]["sample_id"] = f"sft_v1r1_val_{bucket_name}_{idx:05d}"
            val_records.append(rec)
        train_pool[bucket_name] = rest

    rng.shuffle(val_records)

    # 5. Construct SFT-v1r1 training mix
    upsample = int(toolace_cfg["multi_turn_upsample"])
    train_by_bucket: dict[str, list[dict[str, Any]]] = {}

    # 5a. ToolACE multi-turn (3x upsample)
    mt_expanded: list[dict[str, Any]] = []
    for rep in range(upsample):
        for rec in train_pool["toolace_multi_turn"]:
            copy_rec = json.loads(json.dumps(rec, ensure_ascii=False))
            copy_rec["meta"]["upsample_copy"] = rep
            mt_expanded.append(copy_rec)
    train_by_bucket["toolace_multi_turn"] = mt_expanded

    # 5b. ToolACE missing_param & irrelevant_refusal (keep all remaining)
    for bname in ("toolace_missing_param", "toolace_irrelevant_refusal"):
        train_by_bucket[bname] = [
            json.loads(json.dumps(r, ensure_ascii=False)) for r in train_pool[bname]
        ]

    # 5c. ToolACE single_or_parallel (sample target)
    sp_target = int(toolace_cfg["single_or_parallel_sample"])
    sp_pool = list(train_pool["toolace_single_or_parallel"])
    rng.shuffle(sp_pool)
    train_by_bucket["toolace_single_or_parallel"] = [
        json.loads(json.dumps(r, ensure_ascii=False)) for r in sp_pool[:sp_target]
    ]

    # 5d. xLAM parallel (sample target)
    xlam_par_target = int(xlam_cfg["parallel_sample"])
    xlam_par_pool = list(train_pool["xlam_parallel"])
    rng.shuffle(xlam_par_pool)
    train_by_bucket["xlam_parallel"] = [
        json.loads(json.dumps(r, ensure_ascii=False)) for r in xlam_par_pool[:xlam_par_target]
    ]

    # 5e. xLAM single (prefer multi-candidate tools)
    xlam_sin_target = int(xlam_cfg["single_sample"])
    xlam_sin_pool = list(train_pool["xlam_single"])
    if xlam_cfg.get("prefer_multi_candidate_single", True):
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

    # 5f. SmolTalk irrelevant tools (stratified sample across subsets)
    smol_target = int(smol_cfg["train_sample"])
    smol_by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in train_pool["smoltalk_irrelevant_tools"]:
        smol_by_src[rec["meta"]["source"]].append(rec)
    chosen_smol: list[dict[str, Any]] = []
    # Match pool_targets proportional share
    pool_weights = {
        f"smoltalk:{k}": int(v) for k, v in smol_cfg["pool_targets"].items()
    }
    total_w = max(1, sum(pool_weights.values()))
    for src_name in sorted(smol_by_src):
        src_pool = list(smol_by_src[src_name])
        rng.shuffle(src_pool)
        take = round(smol_target * pool_weights.get(src_name, 0) / total_w)
        chosen_smol.extend(src_pool[:take])
    if len(chosen_smol) < smol_target:
        used_ids = {r["meta"]["source_id"] for r in chosen_smol}
        leftovers = [
            r
            for r in train_pool["smoltalk_irrelevant_tools"]
            if r["meta"]["source_id"] not in used_ids
        ]
        rng.shuffle(leftovers)
        chosen_smol.extend(leftovers[: smol_target - len(chosen_smol)])
    chosen_smol = chosen_smol[:smol_target]
    train_by_bucket["smoltalk_irrelevant_tools"] = [
        json.loads(json.dumps(r, ensure_ascii=False)) for r in chosen_smol
    ]

    train_records: list[dict[str, Any]] = []
    for bucket_name in sorted(train_by_bucket):
        for idx, rec in enumerate(train_by_bucket[bucket_name]):
            rec["meta"]["split"] = "train"
            rec["meta"]["sample_id"] = f"sft_v1r1_{bucket_name}_{idx:05d}"
            train_records.append(rec)
    rng.shuffle(train_records)

    # 6. Construct smoke subset (unique samples drawn from train_pool without duplicates)
    smoke_file = str(cfg.get("smoke_file", "smoke_v1r1.jsonl"))
    smoke_cfg = cfg["smoke_counts"]
    smoke_records: list[dict[str, Any]] = []
    for bucket_name, target_n in sorted(smoke_cfg.items()):
        target_int = int(target_n)
        if bucket_name == "smoltalk_irrelevant_tools":
            # Stratify smoke smoltalk cleanly as well
            smol_smoke_by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rec in train_pool[bucket_name]:
                smol_smoke_by_src[rec["meta"]["source"]].append(rec)
            chosen_smoke_smol: list[dict[str, Any]] = []
            for src_name in sorted(smol_smoke_by_src):
                src_pool = list(smol_smoke_by_src[src_name])
                rng.shuffle(src_pool)
                take = round(target_int * pool_weights.get(src_name, 0) / total_w)
                chosen_smoke_smol.extend(src_pool[:take])
            if len(chosen_smoke_smol) < target_int:
                used_ids = {r["meta"]["source_id"] for r in chosen_smoke_smol}
                leftovers = [
                    r for r in train_pool[bucket_name] if r["meta"]["source_id"] not in used_ids
                ]
                rng.shuffle(leftovers)
                chosen_smoke_smol.extend(leftovers[: target_int - len(chosen_smoke_smol)])
            slice_recs = chosen_smoke_smol[:target_int]
        elif bucket_name == "xlam_single" and xlam_cfg.get("prefer_multi_candidate_single", True):
            multi_cand = [r for r in train_pool[bucket_name] if r["meta"]["num_candidate_tools"] >= 2]
            one_cand = [r for r in train_pool[bucket_name] if r["meta"]["num_candidate_tools"] < 2]
            rng.shuffle(multi_cand)
            rng.shuffle(one_cand)
            slice_recs = (multi_cand + one_cand)[:target_int]
        else:
            unique_pool = list(train_pool[bucket_name])
            rng.shuffle(unique_pool)
            slice_recs = unique_pool[:target_int]

        for idx, rec in enumerate(slice_recs):
            copy_rec = json.loads(json.dumps(rec, ensure_ascii=False))
            copy_rec["meta"]["split"] = "smoke_v1r1"
            copy_rec["meta"]["sample_id"] = f"sft_v1r1_smoke_{bucket_name}_{idx:05d}"
            smoke_records.append(copy_rec)
    rng.shuffle(smoke_records)

    # 7. Write outputs + patched template + samples + manifest
    tpl_path = out_dir / "qwen35_chat_template_train.jinja"
    tpl_path.write_text(patched_template, encoding="utf-8")

    train_sha = _write_jsonl(out_dir / "train.jsonl", train_records)
    val_sha = _write_jsonl(out_dir / "val.jsonl", val_records)
    smoke_sha = _write_jsonl(out_dir / smoke_file, smoke_records)

    _render_bucket_samples(
        out_dir / "samples", train_by_bucket, tokenizer, patched_template
    )

    manifest = {
        "version": cfg["version"],
        "seed": seed,
        "max_seq_length": max_seq_length,
        "tokenizer_model_id": cfg["tokenizer"]["model_id"],
        "distractor_tool_pool_size": len(distractor_index.tools),
        "raw_sources": {
            "toolace": {
                "path": str(toolace_cfg["raw_path"]),
                "sha256": _sha256_file(toolace_path),
                "strip_policy_prompt": bool(toolace_cfg.get("strip_policy_prompt", True)),
                "split_no_call_subtypes": bool(toolace_cfg.get("split_no_call_subtypes", True)),
                "parse_stats": toolace_parse_stats.to_dict(),
                "dedup_dropped": toolace_dups,
            },
            "xlam_60k": {
                "path": str(xlam_cfg["raw_path"]),
                "sha256": _sha256_file(xlam_path),
                "parse_stats": xlam_parse_stats.to_dict(),
                "dedup_dropped": xlam_dups,
            },
            "smoltalk_irrelevant": {
                "dir": str(smol_cfg["raw_dir"]),
                "subsets": smol_parse_stats,
                "dedup_dropped": smol_dups,
            },
        },
        "clean_pool_after_dedup_and_length_filter": {
            bucket: len(items) for bucket, items in sorted(buckets.items())
        },
        "length_filter_stats": filter_stats,
        "upsample": {"toolace_multi_turn": upsample},
        "outputs": {
            "train.jsonl": {"sha256": train_sha, **_split_summary(train_records)},
            "val.jsonl": {"sha256": val_sha, **_split_summary(val_records)},
            smoke_file: {"sha256": smoke_sha, **_split_summary(smoke_records)},
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
        default=ROOT / "configs/data/sft_v1r1.yaml",
        help="Path to SFT-v1r1 dataset config YAML",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Optional override path to local Qwen3.5 tokenizer snapshot",
    )
    args = parser.parse_args()
    manifest = build_sft_v1r1(args.config, args.tokenizer)
    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
