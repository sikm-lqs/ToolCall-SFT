"""Build large difficulty-mining GRPO candidate pool from cleaned ToolACE + xLAM-60k."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from transformers import AutoTokenizer
import yaml

from .build_grpo import _pick_unique_by_group, _verify_token_equivalence, prepare_grpo_pools
from .build_sft_v1 import _sha256_file, _write_jsonl
from .build_sft_v2_curated import _norm_text

ROOT = Path(__file__).resolve().parents[3]


def _pick_up_to_unique(
    pool: list[dict[str, Any]],
    limit: int | None,
    used_groups: set[str],
    used_user_turns: set[str],
    rng: random.Random,
    *,
    min_one_tool_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    if limit is not None:
        return _pick_unique_by_group(
            pool,
            int(limit),
            used_groups,
            used_user_turns,
            rng,
            min_one_tool_ratio=min_one_tool_ratio,
        )
    items = list(pool)
    rng.shuffle(items)
    chosen: list[dict[str, Any]] = []
    for item in items:
        gkey = item["meta"]["group_key"]
        if gkey in used_groups:
            continue
        turns = [
            _norm_text(str(m.get("content") or ""))
            for m in item["prompt_messages"]
            if m.get("role") == "user"
        ]
        if any(t in used_user_turns for t in turns):
            continue
        chosen.append(item)
        used_groups.add(gkey)
        used_user_turns.update(turns)
    return chosen


def build_rollout_pool(
    config_path: Path, tokenizer_override: Path | None = None
) -> dict[str, Any]:
    """Build and SHA-256-verify the rollout mining candidate JSONL and manifest."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rng = random.Random(int(cfg["seed"]))

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

    limits = cfg.get("pool_limits") or {}
    pools, _, _, excl_groups, excl_turns = prepare_grpo_pools(
        cfg,
        tokenizer,
        adv_irr_limit=int(limits.get("adv_irr_limit", 800)),
        miss_func_limit=int(limits.get("miss_func_limit", 1000)),
    )

    used_groups = set(excl_groups)
    used_turns = set(excl_turns)
    for raw_excl in cfg.get("exclude_pool_paths", []):
        ep = ROOT / raw_excl
        if ep.exists():
            with ep.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    gkey = rec.get("meta", {}).get("group_key")
                    if gkey:
                        used_groups.add(str(gkey))
                    for m in rec.get("prompt_messages") or []:
                        if m.get("role") == "user":
                            used_turns.add(_norm_text(str(m.get("content") or "")))

    ratio = float(cfg.get("single_turn_min_one_tool_ratio", 0.45))

    rows: list[dict[str, Any]] = []
    for bucket, target in cfg["counts"].items():
        picked = _pick_up_to_unique(
            pools[bucket],
            target,
            used_groups,
            used_turns,
            rng,
            min_one_tool_ratio=ratio if bucket == "single_turn_call" else 0.0,
        )
        rows.extend(picked)

    rng.shuffle(rows)
    out_dir = (ROOT / cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    id_prefix = str(cfg.get("sample_id_prefix", "grpo_pool"))
    for idx, row in enumerate(rows):
        row["meta"]["split"] = str(cfg.get("split_tag", "rollout_pool"))
        row["meta"]["sample_id"] = f"{id_prefix}_{idx:05d}"
        _verify_token_equivalence(tokenizer, row["prompt"], row["prompt_messages"], row["tools"])

    cand_path = out_dir / "candidates.jsonl"
    _write_jsonl(cand_path, rows)

    toks = [r["meta"]["num_prompt_tokens"] for r in rows]
    manifest = {
        "version": cfg["version"],
        "seed": int(cfg["seed"]),
        "max_prompt_tokens": int(cfg["max_prompt_tokens"]),
        "outputs": {
            "candidates.jsonl": {
                "sha256": _sha256_file(cand_path),
                "count": len(rows),
                "buckets": dict(Counter(r["bucket"] for r in rows)),
                "task_types": dict(Counter(r["task_type"] for r in rows)),
                "one_tool_rows": sum(1 for r in rows if r["meta"]["num_candidate_tools"] == 1),
                "prompt_tokens_total": sum(toks),
                "prompt_tokens_avg": round(sum(toks) / max(1, len(toks)), 2),
                "prompt_tokens_max": max(toks) if toks else 0,
            }
        },
        "pool_raw_sizes": {k: len(v) for k, v in pools.items()},
        "invariants": {
            "excluded_val_overlap": 0,
            "unique_group_keys": len({r["meta"]["group_key"] for r in rows}),
            "token_equivalence_verified": True,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/data/grpo_rollout_pool.yaml")
    )
    parser.add_argument("--tokenizer", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_rollout_pool(args.config, tokenizer_override=args.tokenizer)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
