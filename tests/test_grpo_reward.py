"""Unit tests for ToolRL + 3.4-B GRPO reward scoring and parser invariants."""

from __future__ import annotations

import unittest

from toolcall.training.grpo_reward import (
    grpo_reward_func,
    parse_qwen_completion,
    score_completion,
)

TOOLS_FIXTURE = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get city weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "unit": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "No-arg health ping.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class GrpoRewardTests(unittest.TestCase):
    def test_exact_single_call_max_reward(self) -> None:
        comp = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        gt = {"tool_calls": [{"name": "get_weather", "arguments": {"city": "paris", "days": 3.0}}]}
        res = score_completion(
            comp,
            task_type="call",
            bucket="single_turn_call",
            ground_truth=gt,
            tools=TOOLS_FIXTURE,
        )
        self.assertEqual(res["r_format"], 1.0)
        self.assertAlmostEqual(res["r_correct"], 3.0, places=6)
        self.assertAlmostEqual(res["r_final"], 4.0, places=6)

    def test_no_arg_function_gets_full_reward(self) -> None:
        comp = "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        gt = {"tool_calls": [{"name": "ping", "arguments": {}}]}
        res = score_completion(
            comp,
            task_type="call",
            bucket="single_turn_call",
            ground_truth=gt,
            tools=TOOLS_FIXTURE,
        )
        self.assertEqual(res["r_format"], 1.0)
        self.assertAlmostEqual(res["r_correct"], 3.0, places=6)
        self.assertAlmostEqual(res["r_final"], 4.0, places=6)

    def test_parallel_call_order_invariant_and_partial_match(self) -> None:
        comp = (
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>"
        )
        gt = {
            "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "Seoul"}},
                {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            ]
        }
        res = score_completion(
            comp,
            task_type="call",
            bucket="parallel_call",
            ground_truth=gt,
            tools=TOOLS_FIXTURE,
        )
        self.assertAlmostEqual(res["r_final"], 4.0, places=6)

        # Wrong value -> name(1) + key(1) + val(0) out of S_max=3 => 6*(2/3)-3 = +1.0
        comp_wrong_val = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n</tool_call>"
        gt_one = {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}
        res_wv = score_completion(
            comp_wrong_val,
            task_type="call",
            bucket="single_turn_call",
            ground_truth=gt_one,
            tools=TOOLS_FIXTURE,
        )
        self.assertEqual(res_wv["r_format"], 1.0)
        self.assertAlmostEqual(res_wv["r_correct"], 1.0, places=6)

    def test_zero_call_and_unclosed_format_on_call_task(self) -> None:
        gt = {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}
        zero = score_completion(
            "I don't want to call it.",
            task_type="call",
            bucket="single_turn_call",
            ground_truth=gt,
            tools=TOOLS_FIXTURE,
        )
        self.assertEqual(zero["r_format"], 1.0)
        self.assertAlmostEqual(zero["r_correct"], -3.0, places=6)
        self.assertAlmostEqual(zero["r_final"], -2.0, places=6)

        broken = score_completion(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis",
            task_type="call",
            bucket="single_turn_call",
            ground_truth=gt,
            tools=TOOLS_FIXTURE,
        )
        self.assertEqual(broken["r_format"], 0.0)
        self.assertAlmostEqual(broken["r_correct"], -3.0, places=6)
        self.assertAlmostEqual(broken["r_final"], -3.0, places=6)

    def test_symmetric_non_call_rewards(self) -> None:
        irr_good = score_completion(
            "The available tools cannot solve a triangle area problem.",
            task_type="non_call",
            bucket="irrelevance_refusal",
            ground_truth={"tool_calls": []},
            tools=TOOLS_FIXTURE,
        )
        self.assertAlmostEqual(irr_good["r_final"], 4.0, places=6)

        irr_bad_call = score_completion(
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
            task_type="non_call",
            bucket="irrelevance_refusal",
            ground_truth={"tool_calls": []},
            tools=TOOLS_FIXTURE,
        )
        self.assertEqual(irr_bad_call["r_format"], 1.0)
        self.assertAlmostEqual(irr_bad_call["r_correct"], -3.0, places=6)
        self.assertAlmostEqual(irr_bad_call["r_final"], -2.0, places=6)

        leak = score_completion(
            "Here is the answer: <tool_call><function=ping></function>",
            task_type="non_call",
            bucket="multi_turn_post_tool_response",
            ground_truth={"tool_calls": [], "tool_observation": '{"temp": 21}'},
            tools=TOOLS_FIXTURE,
        )
        self.assertEqual(leak["r_format"], 0.0)
        self.assertAlmostEqual(leak["r_correct"], -3.0, places=6)
        self.assertAlmostEqual(leak["r_final"], -3.0, places=6)

        mp_good = score_completion(
            "Could you please specify the city you want the weather for?",
            task_type="non_call",
            bucket="missing_param",
            ground_truth={"tool_calls": [], "missing_params": ["city"]},
            tools=TOOLS_FIXTURE,
        )
        self.assertAlmostEqual(mp_good["r_final"], 4.0, places=6)

        post_good = score_completion(
            "The current temperature is 21 degrees.",
            task_type="non_call",
            bucket="multi_turn_post_tool_response",
            ground_truth={"tool_calls": [], "tool_observation": '{"temperature_c": 21}'},
            tools=TOOLS_FIXTURE,
        )
        self.assertAlmostEqual(post_good["r_final"], 4.0, places=6)

    def test_trl_batch_reward_func(self) -> None:
        logged = {}
        rewards = grpo_reward_func(
            [
                "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
                "I cannot perform this task with the given tools.",
            ],
            task_type=["call", "non_call"],
            bucket=["single_turn_call", "irrelevance_refusal"],
            ground_truth=[{"tool_calls": [{"name": "ping", "arguments": {}}]}, {"tool_calls": []}],
            tools=[TOOLS_FIXTURE, TOOLS_FIXTURE],
            log_metric=lambda k, v: logged.__setitem__(k, v),
        )
        self.assertEqual(rewards, [4.0, 4.0])
        self.assertEqual(logged["reward/format_mean"], 1.0)
        self.assertEqual(logged["reward/leak_rate"], 0.0)

    def test_trl_batch_reward_func_with_json_strings(self) -> None:
        import json

        rewards = grpo_reward_func(
            ["<tool_call>\n<function=ping>\n</function>\n</tool_call>"],
            task_type=["call"],
            bucket=["single_turn_call"],
            ground_truth_json=[json.dumps({"tool_calls": [{"name": "ping", "arguments": {}}]})],
            tools_json=[json.dumps(TOOLS_FIXTURE)],
        )
        self.assertEqual(rewards, [4.0])


if __name__ == "__main__":
    unittest.main()
