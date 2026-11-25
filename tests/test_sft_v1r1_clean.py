"""Regression tests for the SFT-v1r1 quality-first freeze."""

import unittest

from toolcall.data.clean_sft_v1r1 import filter_splits, review_digest


def row(
    prompt, name, split="train", bucket="toolace_single_or_parallel", source="fixture"
):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "An answer."},
        ],
        "tools": [{"function": {"name": name, "description": name}}],
        "meta": {
            "sample_id": f"{split}-{name}",
            "source": source,
            "source_id": f"{split}-{name}",
            "bucket": bucket,
        },
    }


SMOL = "smoltalk_irrelevant_tools"


class CleanSftV1r1Tests(unittest.TestCase):
    def test_heldout_group_is_removed_even_if_tools_or_normalization_differ(self):
        val = row("What   is ４?", "one", "val")
        train = row("what is 4?", "two")
        smoke = row(" What is 4? ", "three", "smoke")
        good = row("Other request", "four")
        result, excluded = filter_splits(
            {"train": [train, good], "val": [val], "smoke": [smoke]}, set(), {}
        )
        self.assertEqual(result["val"], [val])
        self.assertEqual(result["train"], [good])
        self.assertEqual(result["smoke"], [])
        self.assertEqual({r["reason"] for r in excluded}, {"heldout_task_group"})

    def test_unreviewed_and_rejected_synthetic_rows_are_never_admitted(self):
        keep = row("Calculate 10+2.", "book_flight", bucket=SMOL)
        changed = row("Calculate 10+2.", "book_flight", "smoke", bucket=SMOL)
        changed["tools"][0]["function"]["description"] = "General problem solver"
        bad = row("Weather in NYC today?", "forecast", "smoke", bucket=SMOL)
        files = {"train": [keep], "val": [], "smoke": [changed, bad]}
        result, excluded = filter_splits(
            files, {review_digest(keep)}, {review_digest(bad): "applicable_tool"}
        )
        self.assertEqual(result["train"], [keep])
        self.assertEqual(result["smoke"], [])
        self.assertEqual(
            {r["reason"] for r in excluded},
            {"synthetic_unreviewed", "applicable_tool"},
        )

    def test_shared_source_id_is_not_independent_with_different_prompts(self):
        val = row("held-out request", "one", "val")
        train = row("different request", "two")
        train["meta"]["source_id"] = val["meta"]["source_id"]
        result, excluded = filter_splits(
            {"train": [train], "val": [val], "smoke": []}, set(), {}
        )
        self.assertEqual(result["train"], [])
        self.assertEqual(excluded[0]["reason"], "heldout_task_group")
