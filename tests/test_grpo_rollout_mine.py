"""Tests for GRPO rollout difficulty scoring, summary stats, and candidate pool config."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from toolcall.training.rollout_mine import score_prompt_rollouts, summarize_scored_rollouts

ROOT = Path(__file__).resolve().parents[1]


class GrpoRolloutMineTests(unittest.TestCase):
    def test_rollout_pool_config_parses(self) -> None:
        cfg = yaml.safe_load((ROOT / "configs/data/grpo_rollout_pool.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["version"], "grpo_rollout_pool_v1")
        self.assertIn("single_turn_call", cfg["counts"])
        self.assertIn("multi_turn_continue", cfg["counts"])
        self.assertIn("missing_param", cfg["counts"])

    def test_score_and_summarize_call_and_non_call(self) -> None:
        call_row = {
            "task_type": "call",
            "bucket": "single_turn_call",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "ground_truth": {
                "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
                "reference_text": "",
                "missing_params": [],
                "tool_observation": "",
            },
            "meta": {
                "sample_id": "s0",
                "group_key": "xlam:0",
                "source": "xlam-60k",
                "num_candidate_tools": 1,
                "num_prompt_tokens": 120,
            },
        }
        good = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
        wrong_val = "<tool_call>\n<function=get_weather>\n<parameter=city>\nLondon\n</parameter>\n</function>\n</tool_call>"
        no_call = "I cannot check weather right now."
        scored_call = score_prompt_rollouts(call_row, [good, good, wrong_val, no_call], [20, 20, 20, 8])
        self.assertEqual(scored_call["pass_count"], 2)
        self.assertAlmostEqual(scored_call["pass_rate"], 0.5)
        self.assertGreater(scored_call["reward_std"], 0.0)

        non_call_row = {
            "task_type": "non_call",
            "bucket": "irrelevance_refusal",
            "tools": call_row["tools"],
            "ground_truth": {"tool_calls": [], "reference_text": "", "missing_params": [], "tool_observation": ""},
            "meta": {**call_row["meta"], "sample_id": "s1", "group_key": "toolace:1"},
        }
        scored_nc = score_prompt_rollouts(
            non_call_row,
            [
                "Sorry, none of the available tools can book flights.",
                "Unfortunately, the weather tool cannot help with flight booking.",
                good,
                "<tool_call>broken",
            ],
            [12, 14, 20, 6],
        )
        self.assertEqual(scored_nc["pass_count"], 2)
        self.assertEqual(scored_nc["r_corrects"][2], -3.0)
        self.assertEqual(scored_nc["leak_count"], 1)

        summary = summarize_scored_rollouts([scored_call, scored_nc], wall_seconds=2.5, n_rollouts=4)
        self.assertEqual(summary["total_prompts"], 2)
        self.assertEqual(summary["total_completions"], 8)
        self.assertEqual(summary["overall"]["mid_25_75_count"], 2)
        self.assertEqual(summary["overall"]["reward_std_gt0_count"], 2)


if __name__ == "__main__":
    unittest.main()
