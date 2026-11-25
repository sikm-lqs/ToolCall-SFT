"""Freeze a quality-first subset of an existing SFT-v1r1 snapshot without relabeling."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unicodedata

from transformers import AutoTokenizer

from .build_sft_v1 import _sha256_file, _write_jsonl
from .build_sft_v1r1 import ROOT, _render_bucket_samples, _split_summary
from .schema import dedup_signature, validate_record
from ..training.preflight import verify_data_manifest

FILES = {"train": "train.jsonl", "val": "val.jsonl", "smoke": "smoke_v1r1.jsonl"}
TEMPLATE = "qwen35_chat_template_train.jinja"
SYNTHETIC = "smoltalk_irrelevant_tools"


def review_digest(record):
    """Bind approval to the exact prompts, answers, and ordered tool schemas."""
    data = json.dumps(
        [record["messages"], record["tools"]], ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def task_key(record):
    """Group exact normalized user turns independently of candidate tools."""
    return tuple(
        " ".join(unicodedata.normalize("NFKC", msg["content"]).casefold().split())
        for msg in record["messages"]
        if msg["role"] == "user"
    )


def group_keys(record):
    meta = record["meta"]
    keys = {("messages", task_key(record)), ("signature", dedup_signature(record))}
    if meta.get("source_id") is not None:
        keys.add(("source", meta["source"], str(meta["source_id"])))
    return keys


def filter_splits(splits, approved, rejected):
    """Strictly allowlist synthetic negatives; give held-out task groups priority."""
    eligible = {split: [] for split in FILES}
    excluded = []

    def drop(split, row, reason, digest=None):
        excluded.append(
            {
                "input_split": split,
                "sample_id": row["meta"]["sample_id"],
                "bucket": row["meta"]["bucket"],
                "reason": reason,
                "content_sha256": digest or review_digest(row),
            }
        )

    for split in FILES:
        for row in splits[split]:
            if row["meta"]["bucket"] == SYNTHETIC:
                digest = review_digest(row)
                if digest in rejected:
                    drop(split, row, rejected[digest], digest)
                    continue
                if digest not in approved:
                    drop(split, row, "synthetic_unreviewed", digest)
                    continue
            eligible[split].append(row)

    heldout = set()
    val = []
    for row in eligible["val"]:
        keys = group_keys(row)
        if heldout & keys:
            drop("val", row, "heldout_duplicate")
            continue
        heldout.update(keys)
        val.append(row)
    eligible["val"] = val
    for split in ("train", "smoke"):
        keep = []
        for row in eligible[split]:
            if group_keys(row) & heldout:
                drop(split, row, "heldout_task_group")
            else:
                keep.append(row)
        eligible[split] = keep
    return eligible, excluded


def freeze(source, output, review_path, tokenizer_path):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    provenance = verify_data_manifest(source, [*FILES.values(), TEMPLATE])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["input_manifest_sha256"] != _sha256_file(source / "manifest.json"):
        raise ValueError("Review list does not match the frozen input manifest")
    approved = {r["sha256"] for r in review["approved"]}
    rejected = {r["sha256"]: r["reason"] for r in review["rejected"]}
    if approved & rejected.keys():
        raise ValueError("Conflicting reviews for the same content")

    splits = {}
    for split, filename in FILES.items():
        with (source / filename).open(encoding="utf-8") as stream:
            splits[split] = [json.loads(line) for line in stream]
    seen = {
        review_digest(row)
        for rows in splits.values()
        for row in rows
        if row["meta"]["bucket"] == SYNTHETIC
    }
    if (approved | rejected.keys()) - seen:
        raise ValueError("Review list references absent or changed synthetic records")

    cleaned, excluded = filter_splits(splits, approved, rejected)
    limits = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for rows in cleaned.values():
        for row in rows:
            ok, reason = validate_record(row)
            meta = row["meta"]
            if (
                not ok
                or not row["tools"]
                or not (
                    0
                    < meta["num_assistant_tokens"]
                    <= meta["num_tokens"]
                    <= limits["max_seq_length"]
                )
            ):
                raise ValueError(
                    f"Invalid retained record {meta['sample_id']}: {reason}"
                )
            if meta["bucket"] == SYNTHETIC and any(
                m.get("tool_calls") for m in row["messages"]
            ):
                raise ValueError(
                    f"Synthetic negative includes tool calls: {meta['sample_id']}"
                )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        shutil.copyfile(source / TEMPLATE, stage / TEMPLATE)
        outputs = {TEMPLATE: {"sha256": provenance[TEMPLATE]}}
        for split, rows in cleaned.items():
            filename = FILES[split]
            outputs[filename] = {
                "sha256": _write_jsonl(stage / filename, rows),
                **_split_summary(rows),
            }
        audit_file = "exclusions.jsonl"
        outputs[audit_file] = {
            "sha256": _write_jsonl(stage / audit_file, excluded),
            "total_rows": len(excluded),
        }
        train_buckets = defaultdict(list)
        for row in cleaned["train"]:
            train_buckets[row["meta"]["bucket"]].append(row)
        _render_bucket_samples(
            stage / "samples",
            train_buckets,
            tokenizer,
            (stage / TEMPLATE).read_text(encoding="utf-8"),
        )
        counts = {
            split: dict(
                sorted(
                    Counter(
                        r["reason"] for r in excluded if r["input_split"] == split
                    ).items()
                )
            )
            for split in FILES
        }
        manifest = {
            "version": "sft_v1r1_clean",
            "source_version": limits["version"],
            "input_manifest_sha256": review["input_manifest_sha256"],
            "input_files_sha256": provenance,
            "review_file_sha256": _sha256_file(review_path),
            "max_seq_length": limits["max_seq_length"],
            "tokenizer_model_id": limits["tokenizer_model_id"],
            "grouping": ["normalized_user_turns", "dedup_signature", "source_and_id"],
            "excluded_by_split": counts,
            "outputs": outputs,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        verify_data_manifest(stage, list(outputs))
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/sft_v1r1")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/sft_v1r1_clean"
    )
    parser.add_argument(
        "--review", type=Path, default=ROOT / "configs/data/sft_v1r1_review.json"
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path.home()
        / ".cache/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master",
    )
    args = parser.parse_args()
    manifest = freeze(args.source, args.output, args.review, args.tokenizer)
    print(json.dumps(manifest["excluded_by_split"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
