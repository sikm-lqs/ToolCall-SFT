"""Build difficulty-mined GRPO v2 train (1,500) and dual-tier val (190) datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
from typing import Any

from transformers import AutoTokenizer
import yaml

from .build_grpo import _load_excluded_val_sets, _write_grpo_bundle
from .build_sft_v2_curated import (
    _load_eval_user_turns,
    _load_tau2_excluded_turns,
    _norm_text,
)

ROOT = Path(__file__).resolve().parents[3]


def _user_turns_of(row: dict[str, Any]) -> list[str]:
    return [
        _norm_text(str(m.get("content") or ""))
        for m in row.get("prompt_messages") or []
        if m.get("role") == "user"
    ]


def _can_take(
    row: dict[str, Any], used_groups: set[str], used_turns: set[str]
) -> bool:
    if row["meta"]["group_key"] in used_groups:
        return False
    return not any(t in used_turns for t in _user_turns_of(row))


def _take(row: dict[str, Any], used_groups: set[str], used_turns: set[str]) -> None:
    used_groups.add(row["meta"]["group_key"])
    used_turns.update(_user_turns_of(row))


def select_with_one_tool_floor(
    pool: list[dict[str, Any]],
    n: int,
    used_groups: set[str],
    used_turns: set[str],
    rng: random.Random,
    *,
    min_one_tool_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    """Select n unique-group items while optionally preserving a 1-tool minimum ratio."""
    if n == 0:
        return []
    items = list(pool)
    rng.shuffle(items)
    chosen: list[dict[str, Any]] = []
    min_one = int(round(n * min_one_tool_ratio))
    if min_one > 0:
        for item in items:
            if len(chosen) >= min_one:
                break
            if item["meta"]["num_candidate_tools"] == 1 and _can_take(
                item, used_groups, used_turns
            ):
                _take(item, used_groups, used_turns)
                chosen.append(item)
    for item in items:
        if len(chosen) >= n:
            break
        if _can_take(item, used_groups, used_turns):
            _take(item, used_groups, used_turns)
            chosen.append(item)
    if len(chosen) < n:
        raise ValueError(f"Insufficient candidates: needed {n}, got {len(chosen)}")
    return chosen


def select_zero_pass_items(
    zero_pool: list[dict[str, Any]],
    n: int,
    used_groups: set[str],
    used_turns: set[str],
    rng: random.Random,
    *,
    std_gt0_share: float = 0.75,
    min_one_tool_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    """Pick 0/4 prompts prioritizing reward_std > 0 while keeping a share of std == 0."""
    if n == 0:
        return []
    avail = [r for r in zero_pool if _can_take(r, used_groups, used_turns)]
    avail_std = [r for r in avail if float(r["meta"]["mining"]["reward_std"]) > 0.0]
    avail_zero = [r for r in avail if float(r["meta"]["mining"]["reward_std"]) == 0.0]

    target_std = min(len(avail_std), int(round(n * std_gt0_share)))
    target_zero = min(len(avail_zero), n - target_std)
    # Backfill if either sub-pool is smaller than its target share
    if target_std + target_zero < n:
        rem = n - (target_std + target_zero)
        extra_std = min(len(avail_std) - target_std, rem)
        target_std += extra_std
        rem -= extra_std
        target_zero += min(len(avail_zero) - target_zero, rem)

    chosen = select_with_one_tool_floor(
        avail_std,
        target_std,
        used_groups,
        used_turns,
        rng,
        min_one_tool_ratio=min_one_tool_ratio,
    ) + select_with_one_tool_floor(
        avail_zero,
        target_zero,
        used_groups,
        used_turns,
        rng,
        min_one_tool_ratio=min_one_tool_ratio,
    )
    if len(chosen) < n:
        raise ValueError(f"Insufficient 0/4 candidates: needed {n}, got {len(chosen)}")
    return chosen


def load_merged_mined_records(
    rollout_runs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Join candidate JSONL rows with their scored rollout metrics by sample_id."""
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for run in rollout_runs:
        cand_path = (ROOT / run["candidates"]).resolve()
        scored_path = (ROOT / run["scored"]).resolve()
        scored_by_id: dict[str, dict[str, Any]] = {}
        with scored_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                scored_by_id[str(rec["sample_id"])] = rec

        with cand_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                cand = json.loads(line)
                sid = str(cand["meta"]["sample_id"])
                if sid in seen_ids:
                    raise ValueError(f"Duplicate sample_id across rollout runs: {sid}")
                seen_ids.add(sid)
                sc = scored_by_id[sid]
                row = copy.deepcopy(cand)
                row["meta"]["mining"] = {
                    "pass_count": int(sc["pass_count"]),
                    "pass_rate": float(sc["pass_rate"]),
                    "reward_mean": float(sc["reward_mean"]),
                    "reward_std": float(sc["reward_std"]),
                    "leak_count": int(sc["leak_count"]),
                }
                merged.append(row)
    return merged


def _summarize_mining(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_hist = Counter(r["meta"]["mining"]["pass_count"] for r in rows)
    std_gt0 = sum(1 for r in rows if float(r["meta"]["mining"]["reward_std"]) > 0.0)
    tiers = Counter(r["meta"].get("val_tier", "train") for r in rows)
    by_b: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "mid": 0, "zero": 0, "full": 0, "std_gt0": 0})
    for r in rows:
        b = r["bucket"]
        pc = int(r["meta"]["mining"]["pass_count"])
        by_b[b]["total"] += 1
        if 1 <= pc <= 3:
            by_b[b]["mid"] += 1
        elif pc == 0:
            by_b[b]["zero"] += 1
        else:
            by_b[b]["full"] += 1
        if float(r["meta"]["mining"]["reward_std"]) > 0.0:
            by_b[b]["std_gt0"] += 1
    return {
        "count": len(rows),
        "pass_hist": {str(k): pass_hist.get(k, 0) for k in range(5)},
        "reward_std_gt0_count": std_gt0,
        "reward_std_gt0_rate": round(std_gt0 / max(1, len(rows)), 4),
        "tiers": dict(tiers),
        "by_bucket": dict(sorted(by_b.items())),
    }


def build_grpo_v2(
    config_path: Path, tokenizer_override: Path | None = None
) -> dict[str, Any]:
    """Select difficulty-filtered Train and dual-tier Val splits and verify all invariants."""
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

    records = load_merged_mined_records(cfg["sources"]["rollout_runs"])
    eval_turns = _load_eval_user_turns([ROOT / d for d in cfg.get("eval_inspection_dirs", [])])
    tau2_turns = _load_tau2_excluded_turns(ROOT / cfg["tau2_domains_dir"])
    excl_groups, excl_turns = _load_excluded_val_sets(
        [ROOT / p for p in cfg.get("exclude_val_paths", [])]
    )

    used_groups = set(excl_groups)
    used_turns = set(excl_turns)

    # Partition by bucket and difficulty tier
    pools: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"mid": [], "zero": [], "full": []}
    )
    for r in records:
        pc = int(r["meta"]["mining"]["pass_count"])
        tier = "mid" if 1 <= pc <= 3 else ("zero" if pc == 0 else "full")
        pools[r["bucket"]][tier].append(r)

    std_share = float(cfg.get("zero_pass_std_gt0_share", 0.75))
    one_tool_ratio = float(cfg.get("single_turn_min_one_tool_ratio", 0.45))

    # 1. Hold out Val FIRST (mined sensitivity set + 4/4 stability anchors)
    val_rows: list[dict[str, Any]] = []
    for bucket, quota in cfg["val"]["mined"].items():
        ratio = one_tool_ratio if bucket == "single_turn_call" else 0.0
        mids = select_with_one_tool_floor(
            pools[bucket]["mid"], int(quota["mid"]), used_groups, used_turns, rng, min_one_tool_ratio=ratio
        )
        for x in mids:
            x["meta"]["val_tier"] = "mined_mid"
        zeros = select_zero_pass_items(
            pools[bucket]["zero"],
            int(quota["zero"]),
            used_groups,
            used_turns,
            rng,
            std_gt0_share=std_share,
            min_one_tool_ratio=ratio,
        )
        for x in zeros:
            x["meta"]["val_tier"] = "mined_zero"
        val_rows.extend(mids + zeros)

    for bucket, cnt in cfg["val"]["anchors_4_4"].items():
        ratio = one_tool_ratio if bucket == "single_turn_call" else 0.0
        anchors = select_with_one_tool_floor(
            pools[bucket]["full"], int(cnt), used_groups, used_turns, rng, min_one_tool_ratio=ratio
        )
        for x in anchors:
            x["meta"]["val_tier"] = "anchor_full"
        val_rows.extend(anchors)

    # 2. Select Train strictly disjoint from Val & historical Vals
    train_rows: list[dict[str, Any]] = []
    for bucket, quota in cfg["train"].items():
        ratio = one_tool_ratio if bucket == "single_turn_call" else 0.0
        mids = select_with_one_tool_floor(
            pools[bucket]["mid"], int(quota["mid"]), used_groups, used_turns, rng, min_one_tool_ratio=ratio
        )
        zeros = select_zero_pass_items(
            pools[bucket]["zero"],
            int(quota["zero"]),
            used_groups,
            used_turns,
            rng,
            std_gt0_share=std_share,
            min_one_tool_ratio=ratio,
        )
        train_rows.extend(mids + zeros)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    out_dir = (ROOT / cfg["output_dir"]).resolve()
    manifest = _write_grpo_bundle(
        out_dir=out_dir,
        train_rows=train_rows,
        val_rows=val_rows,
        tokenizer=tokenizer,
        eval_turns=eval_turns,
        tau2_turns=tau2_turns,
        excl_groups=excl_groups,
        excl_turns=excl_turns,
        cfg=cfg,
        split_tag="grpo_v2",
    )
    manifest["mining_stats"] = {
        "train": _summarize_mining(train_rows),
        "val": _summarize_mining(val_rows),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    smoke_cfg = cfg.get("smoke")
    if smoke_cfg:
        smoke_train = _sample_stratified_subset(
            train_rows, smoke_cfg["train"], rng, one_tool_ratio=one_tool_ratio
        )
        smoke_val = _sample_stratified_val_subset(
            val_rows, smoke_cfg["val"], rng, one_tool_ratio=one_tool_ratio
        )
        rng.shuffle(smoke_train)
        rng.shuffle(smoke_val)
        smoke_dir = (ROOT / smoke_cfg["output_dir"]).resolve()
        smoke_manifest = _write_grpo_bundle(
            out_dir=smoke_dir,
            train_rows=smoke_train,
            val_rows=smoke_val,
            tokenizer=tokenizer,
            eval_turns=eval_turns,
            tau2_turns=tau2_turns,
            excl_groups=excl_groups,
            excl_turns=excl_turns,
            cfg=cfg,
            split_tag="grpo_v2_smoke",
        )
        smoke_manifest["mining_stats"] = {
            "train": _summarize_mining(smoke_train),
            "val": _summarize_mining(smoke_val),
        }
        (smoke_dir / "manifest.json").write_text(
            json.dumps(smoke_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest["smoke"] = smoke_manifest
    return manifest


def _sample_stratified_subset(
    rows: list[dict[str, Any]],
    quotas: dict[str, dict[str, int]],
    rng: random.Random,
    *,
    one_tool_ratio: float,
) -> list[dict[str, Any]]:
    by_bt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        pc = int(r["meta"]["mining"]["pass_count"])
        tier = "mid" if 1 <= pc <= 3 else ("zero" if pc == 0 else "full")
        by_bt[(r["bucket"], tier)].append(copy.deepcopy(r))
    out: list[dict[str, Any]] = []
    used_g: set[str] = set()
    used_t: set[str] = set()
    for bucket, q in quotas.items():
        ratio = one_tool_ratio if bucket == "single_turn_call" else 0.0
        out.extend(
            select_with_one_tool_floor(
                by_bt[(bucket, "mid")], int(q["mid"]), used_g, used_t, rng, min_one_tool_ratio=ratio
            )
        )
        out.extend(
            select_zero_pass_items(
                by_bt[(bucket, "zero")],
                int(q["zero"]),
                used_g,
                used_t,
                rng,
                std_gt0_share=0.75,
                min_one_tool_ratio=ratio,
            )
        )
    return out


def _sample_stratified_val_subset(
    val_rows: list[dict[str, Any]],
    val_cfg: dict[str, Any],
    rng: random.Random,
    *,
    one_tool_ratio: float,
) -> list[dict[str, Any]]:
    mined_part = _sample_stratified_subset(
        [r for r in val_rows if r["meta"].get("val_tier") != "anchor_full"],
        val_cfg["mined"],
        rng,
        one_tool_ratio=one_tool_ratio,
    )
    by_b_anc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in val_rows:
        if r["meta"].get("val_tier") == "anchor_full":
            by_b_anc[r["bucket"]].append(copy.deepcopy(r))
    anc_part: list[dict[str, Any]] = []
    used_g = {r["meta"]["group_key"] for r in mined_part}
    used_t = {t for r in mined_part for t in _user_turns_of(r)}
    for bucket, cnt in val_cfg["anchors_4_4"].items():
        ratio = one_tool_ratio if bucket == "single_turn_call" else 0.0
        anc_part.extend(
            select_with_one_tool_floor(
                by_b_anc[bucket], int(cnt), used_g, used_t, rng, min_one_tool_ratio=ratio
            )
        )
    return mined_part + anc_part


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/grpo_v2.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_grpo_v2(args.config, tokenizer_override=args.tokenizer)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
