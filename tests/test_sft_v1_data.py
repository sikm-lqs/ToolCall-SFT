"""Unit and integration tests for SFT-v1 data cleaning, Qwen3.5 chat template, and loss masking."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from transformers import AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling, pack_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolcall.data.loader import load_tokenized_sft_dataset  # noqa: E402
from toolcall.data.parsers import parse_toolace_bracket_calls  # noqa: E402
from toolcall.data.schema import (  # noqa: E402
    normalize_toolace_parameters,
    normalize_xlam_parameters,
    validate_record,
)
from toolcall.data.template import (  # noqa: E402
    patch_qwen35_chat_template,
    render_and_mask,
)

QWEN35_SNAPSHOT = Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"


class SchemaNormalizationTests(unittest.TestCase):
    def test_xlam_parameter_normalization(self) -> None:
        raw = {
            "query": {"description": "Search query", "type": "str"},
            "limit": {"description": "Max items", "type": "int, optional", "default": 10},
            "ids": {"description": "ID list", "type": "List[int]"},
        }
        norm = normalize_xlam_parameters(raw)
        self.assertEqual(norm["type"], "object")
        self.assertEqual(norm["required"], ["query", "ids"])
        self.assertEqual(norm["properties"]["query"]["type"], "string")
        self.assertEqual(norm["properties"]["limit"]["type"], "integer")
        self.assertEqual(norm["properties"]["limit"]["default"], 10)
        self.assertEqual(
            norm["properties"]["ids"],
            {"type": "array", "items": {"type": "integer"}, "description": "ID list"},
        )

    def test_toolace_parameter_normalization_preserves_property_named_type(self) -> None:
        raw = {
            "type": "dict",
            "properties": {
                "type": {"type": "string", "description": "Filter type"},
                "count": {"type": "int"},
                "score": {"type": "float"},
            },
            "required": ["type"],
        }
        norm = normalize_toolace_parameters(raw)
        self.assertEqual(norm["type"], "object")
        self.assertEqual(norm["properties"]["type"]["type"], "string")
        self.assertEqual(norm["properties"]["count"]["type"], "integer")
        self.assertEqual(norm["properties"]["score"]["type"], "number")

    def test_toolace_bracket_parser_handles_spaces_parens_and_keywords(self) -> None:
        known = [
            "User Feed (Video Posts) V2",
            "Financial Fundamentals API",
            "Market Trends API",
        ]
        text = (
            '[User Feed (Video Posts) V2(username="sunny,days"), '
            'Financial Fundamentals API(from="2025-01-01", to="2025-12-31", limit=5)]'
        )
        calls = parse_toolace_bracket_calls(text, known)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "User Feed (Video Posts) V2")
        self.assertEqual(calls[0]["arguments"], {"username": "sunny,days"})
        self.assertEqual(calls[1]["name"], "Financial Fundamentals API")
        self.assertEqual(
            calls[1]["arguments"],
            {"from": "2025-01-01", "to": "2025-12-31", "limit": 5},
        )


@unittest.skipUnless(QWEN35_SNAPSHOT.exists(), "Local Qwen3.5-2B snapshot required")
class Qwen35TemplateAndMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            QWEN35_SNAPSHOT, local_files_only=True, trust_remote_code=False
        )
        raw_tpl = (QWEN35_SNAPSHOT / "chat_template.jinja").read_text(encoding="utf-8")
        cls.patched_tpl = patch_qwen35_chat_template(raw_tpl)

    def test_multi_turn_parallel_and_refusal_loss_mask(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        multi_turn_rec = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Weather in Beijing and Tokyo?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Beijing"}}},
                        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}},
                    ],
                },
                {"role": "tool", "name": "get_weather", "content": '{"temp": 22}'},
                {"role": "tool", "name": "get_weather", "content": '{"temp": 18}'},
                {"role": "assistant", "content": "Beijing is 22C and Tokyo is 18C."},
                {"role": "user", "content": "Can you book a flight?"},
                {"role": "assistant", "content": "I do not have a flight booking tool."},
            ],
            "tools": tools,
        }
        ok, reason = validate_record(multi_turn_rec)
        self.assertTrue(ok, reason)

        rendered = render_and_mask(
            self.tokenizer, multi_turn_rec, self.patched_tpl, verify_equivalence=True
        )
        segs = rendered["assistant_segments"]
        self.assertEqual(len(segs), 3)
        self.assertIn("<function=get_weather>", segs[0])
        self.assertIn("Beijing is 22C", segs[1])
        self.assertIn("do not have a flight booking tool", segs[2])
        for seg in segs:
            self.assertNotIn("<think>", seg)
            self.assertNotIn("<tool_response>", seg)
            self.assertTrue(seg.endswith("<|im_end|>"))

    def test_loader_and_trl_packing_preserve_distinct_arguments(self) -> None:
        r1 = {
            "messages": [
                {"role": "user", "content": "Check Beijing"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "w", "arguments": {"city": "Beijing"}}}],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "w", "description": "w", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
            "meta": {"source": "xlam-60k"},
        }
        r2 = {
            "messages": [
                {"role": "user", "content": "Add 1+2"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "add", "arguments": {"x": 1, "y": 2}}}],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "add", "description": "a", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}}}],
            "meta": {"source": "toolace"},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sample.jsonl"
            p.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n", encoding="utf-8")
            ds = load_tokenized_sft_dataset(p, self.tokenizer, self.patched_tpl)
            self.assertEqual(set(ds.column_names), {"input_ids", "labels"})
            self.assertEqual(len(ds), 2)
            # Ensure Arrow never injected null parameters from other rows
            decoded_0 = self.tokenizer.decode(ds[0]["input_ids"], skip_special_tokens=False)
            self.assertNotIn("<parameter=x>", decoded_0)
            self.assertNotIn('"x": null', decoded_0)

            packed = pack_dataset(ds, seq_length=1024, strategy="bfd")
            collator = DataCollatorForLanguageModeling(
                pad_token_id=self.tokenizer.pad_token_id, padding_free=True
            )
            batch = collator([packed[0]])
            self.assertIn("cu_seq_lens_q", batch)
            self.assertTrue((batch["labels"] == -100).any())
            self.assertTrue((batch["labels"] != -100).any())


if __name__ == "__main__":
    unittest.main()
