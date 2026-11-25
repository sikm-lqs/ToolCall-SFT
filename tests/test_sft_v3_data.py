"""Unit tests for SFT-v3 strict schema, pseudo-call, boundary, and anti-leakage checks."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolcall.data.parsers import (  # noqa: E402
    classify_toolace_no_call,
    load_apigen_mt_5k,
)
from toolcall.data.schema import validate_record  # noqa: E402
from toolcall.data.smoltalk import DistractorToolIndex  # noqa: E402

_BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "calc_metric",
        "description": "Compute metric",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["count"],
        },
    },
}


class SftV3StrictValidationTests(unittest.TestCase):
    def test_strict_arg_types_and_duplicate_tool_names_are_rejected(self) -> None:
        bad_float_for_int = {
            "tools": [_BASE_TOOL],
            "messages": [
                {"role": "user", "content": "Run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "calc_metric", "arguments": {"count": 1.5}}}
                    ],
                },
            ],
        }
        ok, reason = validate_record(bad_float_for_int, strict_arg_types=True)
        self.assertFalse(ok)
        self.assertEqual(reason, "arg_type_mismatch:integer")

        bad_str_for_array = {
            "tools": [_BASE_TOOL],
            "messages": [
                {"role": "user", "content": "Run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "calc_metric",
                                "arguments": {"count": 2, "tags": "list(range(3))"},
                            },
                        }
                    ],
                },
            ],
        }
        ok, reason = validate_record(bad_str_for_array, strict_arg_types=True)
        self.assertFalse(ok)
        self.assertEqual(reason, "arg_type_mismatch:array")

        dup_tools = {
            "tools": [_BASE_TOOL, _BASE_TOOL],
            "messages": [
                {"role": "user", "content": "Run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "calc_metric", "arguments": {"count": 2}}}
                    ],
                },
            ],
        }
        ok, reason = validate_record(dup_tools, reject_duplicate_tools=True)
        self.assertFalse(ok)
        self.assertEqual(reason, "duplicate_tool_name")

    def test_strict_toolace_boundary_fixes_cross_contamination(self) -> None:
        encode_tool = [
            {
                "type": "function",
                "function": {
                    "name": "encode_image",
                    "description": "Encode image",
                    "parameters": {
                        "type": "object",
                        "properties": {"image_path": {"type": "string"}, "output_format": {"type": "string"}},
                        "required": ["image_path", "output_format"],
                    },
                },
            }
        ]
        # Case like 00275: mentions required args & encode_image -> missing_param, NOT irrelevance
        self.assertEqual(
            classify_toolace_no_call(
                [
                    {
                        "role": "assistant",
                        "content": "The query does not provide the specific paths. The required arguments for `encode_image` are `image_path` and `output_format`.",
                    }
                ],
                encode_tool,
                strict_boundary=True,
            ),
            ("toolace_missing_param", None),
        )
        # Case like 00074: starts with boilerplate, then says unrelated -> cleanly reclassified as irrelevance
        yt_tools = [
            {
                "type": "function",
                "function": {
                    "name": "Search_Jobs",
                    "description": "Search jobs",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                },
            }
        ]
        self.assertEqual(
            classify_toolace_no_call(
                [
                    {
                        "role": "assistant",
                        "content": "The given question lacks the parameters required by the available functions. The functions provided are related to job searches, not YouTube channel information.",
                    }
                ],
                yt_tools,
                strict_boundary=True,
            ),
            ("toolace_irrelevant_refusal", None),
        )
        # True mixed conflict (both irrelevance AND missing specific required param) -> quarantined
        self.assertEqual(
            classify_toolace_no_call(
                [
                    {
                        "role": "assistant",
                        "content": "Search_Jobs lacks the required parameter query, and none of the provided functions support YouTube.",
                    }
                ],
                yt_tools,
                strict_boundary=True,
            ),
            (None, "no_call_boundary_conflict"),
        )

    def test_apigen_drops_nl_pseudo_calls_and_tau2_entities(self) -> None:
        tools = [
            {"name": "get_order", "description": "Get order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
        ]
        rows = [
            {
                "system": "# Retail agent policy",
                "tools": json.dumps(tools),
                "conversations": [
                    {"from": "human", "value": "Check #W111"},
                    {"from": "function_call", "value": json.dumps({"name": "get_order", "arguments": {"order_id": "#W111"}})},
                    {"from": "observation", "value": "Error"},
                    {"from": "gpt", "value": 'Retrying:\n[{"id":"call_123","function":{"name":"get_order"}}]'},
                ],
            },
            {
                "system": "# Retail agent policy",
                "tools": json.dumps(tools),
                "conversations": [
                    {"from": "human", "value": "Check #W9300146"},
                    {"from": "function_call", "value": json.dumps({"name": "get_order", "arguments": {"order_id": "#W9300146"}})},
                    {"from": "observation", "value": '{"status": "pending"}'},
                    {"from": "gpt", "value": "Order #W9300146 is pending."},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "apigen.json"
            p.write_text(json.dumps(rows), encoding="utf-8")
            recs, stats = load_apigen_mt_5k(
                p, excluded_entities={"#W9300146"}, strict_validation=True
            )
            self.assertEqual(len(recs), 0)
            self.assertEqual(stats.dropped["pseudo_call_in_nl"], 1)
            self.assertEqual(stats.dropped["tau2_task_entity_overlap"], 1)

    def test_camelcase_distractor_exclusion_blocks_solve_math_and_ai_answer(self) -> None:
        recs = [
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "solveMathProblem",
                            "description": "Solve math",
                            "parameters": {"type": "object", "properties": {}, "required": []},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Get_AI_Answer",
                            "description": "Answer anything",
                            "parameters": {"type": "object", "properties": {}, "required": []},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "library_catalog_lookup",
                            "description": "Library books",
                            "parameters": {"type": "object", "properties": {}, "required": []},
                        },
                    },
                ]
            }
        ]
        idx = DistractorToolIndex(recs)
        names = {t["function"]["name"] for t in idx.tools}
        self.assertEqual(names, {"library_catalog_lookup"})
        chosen = idx.sample_distractors("Solve 2 + 2", k=1, rng=random.Random(42))
        self.assertEqual([t["function"]["name"] for t in chosen], ["library_catalog_lookup"])


if __name__ == "__main__":
    unittest.main()
