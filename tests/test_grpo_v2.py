"""Tests for GRPO v2 difficulty-mined selection and train/val isolation."""

from __future__ import annotations

import json
from pathlib import Path
import random
import unittest

import yaml

from toolcall.data.build_grpo_v2 import select_with_one_tool_floor, select_zero_pass_items

ROOT = Path(__file__).resolve().parents[1]


def _make_dummy(idx: int, *, pass_count: int, reward_std: float, one_tool: bool) -> dict:
    return {
        "prompt": f"p{idx}",
        "prompt_messages": [{"role": "user", "content": f"user query {idx}"}],
        "tools": [{"type": "function", "function": {"name": "fn"}}] * (1 if one_tool else 2),
        "task_type": "call",
        "bucket": "single_turn_call",
        "ground_truth": {"tool_calls": [], "reference_text": "", "missing_params": [], "tool_observation": ""},
        "meta": {
            "sample_id": f"s{idx}",
            "group_key": f"xlam:{idx}",
            "num_candidate_tools": 1 if one_tool else 2,
            "mining": {
                "pass_count": pass_count,
                "pass_rate": pass_count / 4.0,
                "reward_mean": 1.0,
                "reward_std": reward_std,
                "leak_count": 0,
            },
        },
    }


class GrpoV2BuilderTests(unittest.TestCase):
    def test_config_quotas_sum_cleanly(self) -> None:
        cfg = yaml.safe_load((ROOT / "configs/data/grpo_v2.yaml").read_text(encoding="utf-8"))
        train_total = sum(v["mid"] + v["zero"] for v in cfg["train"].values())
        val_mined = sum(v["mid"] + v["zero"] for v in cfg["val"]["mined"].values())
        val_anchors = sum(cfg["val"]["anchors_4_4"].values())
        self.assertEqual(train_total, 1500)
        self.assertEqual(val_mined, 150)
        self.assertEqual(val_anchors, 40)

    def test_zero_pass_prioritizes_reward_std_gt0_and_retains_std0(self) -> None:
        pool = [_make_dummy(i, pass_count=0, reward_std=1.2, one_tool=(i % 2 == 0)) for i in range(10)] + [
            _make_dummy(100 + i, pass_count=0, reward_std=0.0, one_tool=(i % 2 == 0)) for i in range(10)
        ]
        used_g: set[str] = set()
        used_t: set[str] = set()
        picked = select_zero_pass_items(
            pool,
            8,
            used_g,
            used_t,
            random.Random(42),
            std_gt0_share=0.75,
            min_one_tool_ratio=0.5,
        )
        self.assertEqual(len(picked), 8)
        n_std = sum(1 for r in picked if r["meta"]["mining"]["reward_std"] > 0)
        n_zero = sum(1 for r in picked if r["meta"]["mining"]["reward_std"] == 0)
        self.assertEqual(n_std, 6)
        self.assertEqual(n_zero, 2)
        self.assertGreaterEqual(sum(1 for r in picked if r["meta"]["num_candidate_tools"] == 1), 4)

    def test_frozen_grpo_v2_if_present(self) -> None:
        man_path = ROOT / "data/processed/grpo_v2/manifest.json"
        if not man_path.exists():
            self.skipTest("grpo_v2 not built yet")
        m = json.loads(man_path.read_text(encoding="utf-8"))
        self.assertEqual(m["outputs"]["train.jsonl"]["count"], 1500)
        self.assertEqual(m["outputs"]["val.jsonl"]["count"], 190)
        self.assertEqual(m["invariants"]["train_val_group_overlap"], 0)
        self.assertEqual(m["invariants"]["eval_user_turn_overlap"], 0)
        self.assertTrue(m["invariants"]["token_equivalence_verified"])

    def test_frozen_grpo_v2_smoke_if_present(self) -> None:
        man_path = ROOT / "data/processed/grpo_v2_smoke/manifest.json"
        if not man_path.exists():
            self.skipTest("grpo_v2_smoke not built yet")
        m = json.loads(man_path.read_text(encoding="utf-8"))
        self.assertEqual(m["outputs"]["train.jsonl"]["count"], 240)
        self.assertEqual(m["outputs"]["val.jsonl"]["count"], 60)
        self.assertEqual(m["invariants"]["train_val_group_overlap"], 0)
        self.assertTrue(m["invariants"]["token_equivalence_verified"])


if __name__ == "__main__":
    unittest.main()
