"""Unit and integration tests for SFT-v2-curated APIGen-MT cleaning and topic guards."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolcall.data.parsers import load_apigen_mt_5k  # noqa: E402
from toolcall.data.smoltalk import DistractorToolIndex  # noqa: E402
from toolcall.data.template import patch_qwen35_chat_template, render_and_mask  # noqa: E402

QWEN35_SNAPSHOT = Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"


class ApigenMtLoaderTests(unittest.TestCase):
    def test_load_apigen_filters_think_parses_string_args_and_excludes_overlap(self) -> None:
        tools = [
            {"name": "think", "description": "Internal scratchpad", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]}},
            {"name": "get_order", "description": "Get order status", "parameters": {"type": "dict", "properties": {"order_id": {"type": "str"}}, "required": ["order_id"]}},
        ]
        rows = [
            # 1. Clean trajectory with double-encoded arguments and unused `think` schema
            {
                "system": "# Retail agent policy\nBe polite.",
                "tools": json.dumps(tools),
                "conversations": [
                    {"from": "human", "value": "Check order #101 please."},
                    {"from": "function_call", "value": json.dumps({"name": "get_order", "arguments": '{"order_id": "#101"}'})},
                    {"from": "observation", "value": '{"status": "shipped"}'},
                    {"from": "gpt", "value": "Order #101 has already shipped."},
                ],
            },
            # 2. Trajectory invoking `think` -> must be dropped
            {
                "system": "# Airline Agent Policy\nCheck rules.",
                "tools": json.dumps(tools),
                "conversations": [
                    {"from": "human", "value": "Cancel flight."},
                    {"from": "function_call", "value": json.dumps({"name": "think", "arguments": {"thought": "check"}})},
                    {"from": "observation", "value": ""},
                    {"from": "gpt", "value": "Cannot cancel."},
                ],
            },
            # 3. Overlaps tau2 excluded turn -> must be dropped
            {
                "system": "# Retail agent policy",
                "tools": json.dumps(tools),
                "conversations": [
                    {"from": "human", "value": "Excluded tau2 task prompt"},
                    {"from": "function_call", "value": json.dumps({"name": "get_order", "arguments": {"order_id": "#202"}})},
                    {"from": "observation", "value": '{"status": "pending"}'},
                    {"from": "gpt", "value": "Order #202 is pending."},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "apigen.json"
            p.write_text(json.dumps(rows), encoding="utf-8")
            recs, stats = load_apigen_mt_5k(
                p, excluded_first_turns={"excluded tau2 task prompt"}, require_final_nl=True
            )
            self.assertEqual(len(recs), 1)
            self.assertEqual(stats.dropped["has_think_call"], 1)
            self.assertEqual(stats.dropped["eval_overlap"], 1)
            rec = recs[0]
            self.assertEqual([t["function"]["name"] for t in rec["tools"]], ["get_order"])
            self.assertEqual(rec["tools"][0]["function"]["parameters"]["type"], "object")
            self.assertEqual(rec["meta"]["domain"], "retail")

            if QWEN35_SNAPSHOT.exists():
                tok = AutoTokenizer.from_pretrained(
                    QWEN35_SNAPSHOT, local_files_only=True, trust_remote_code=False
                )
                tpl = patch_qwen35_chat_template(
                    (QWEN35_SNAPSHOT / "chat_template.jinja").read_text(encoding="utf-8")
                )
                rendered = render_and_mask(tok, rec, tpl, verify_equivalence=True)
                self.assertEqual(len(rendered["assistant_segments"]), 2)
                self.assertIn("<function=get_order>", rendered["assistant_segments"][0])
                self.assertIn("Order #101 has already shipped.", rendered["assistant_segments"][1])


class TopicConflictGuardTests(unittest.TestCase):
    def test_topic_conflict_guard_blocks_same_domain_sensitive_tools(self) -> None:
        records = [
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "atmospheric_conditions_api",
                            "description": "Retrieve live weather forecast and rain humidity for a city.",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "library_catalog_lookup",
                            "description": "Look up library books and ISBN records.",
                            "parameters": {"type": "object", "properties": {"isbn": {"type": "string"}}, "required": ["isbn"]},
                        },
                    },
                ]
            }
        ]
        idx = DistractorToolIndex(records)
        chosen = idx.sample_distractors(
            "What causes rain and humidity during a weather forecast?",
            k=1,
            rng=random.Random(42),
        )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["function"]["name"], "library_catalog_lookup")


if __name__ == "__main__":
    unittest.main()
