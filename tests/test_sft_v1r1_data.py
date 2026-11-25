"""Unit and integration tests for SFT-v1r1 cleaning, classification, and SmolTalk distractors."""

from __future__ import annotations

from pathlib import Path
import random
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolcall.data.parsers import (  # noqa: E402
    classify_toolace_no_call,
    strip_toolace_policy_prompt,
)
from toolcall.data.smoltalk import (  # noqa: E402
    DistractorToolIndex,
    load_smoltalk_irrelevant_records,
)
from toolcall.data.template import (  # noqa: E402
    patch_qwen35_chat_template,
    render_and_mask,
)

QWEN35_SNAPSHOT = Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"


class ToolaceV1r1CleaningTests(unittest.TestCase):
    def test_strip_toolace_policy_prompt_keeps_date_anchor(self) -> None:
        raw = (
            "You are an expert in composing functions. You are given a question and a set of "
            "possible functions. Based on the question, you will need to make one or more "
            "function/tool calls to achieve the purpose. If none of the function can be used, "
            "point it out. If the given question lacks the parameters required by the function, "
            "also point it out.\nToday is 2024-06-28, Friday.."
        )
        self.assertEqual(strip_toolace_policy_prompt(raw), "Today is 2024-06-28, Friday.")

        pure_policy = (
            "You are an expert in composing functions. You are given a question and a set of "
            "possible functions. Based on the question, you will need to make one or more "
            "function/tool calls to achieve the purpose. If none of the function can be used, "
            "point it out. If the given question lacks the parameters required by the function, "
            "also point it out."
        )
        self.assertEqual(strip_toolace_policy_prompt(pure_policy), "")

    def test_classify_toolace_no_call_separates_subtypes_and_quarantines_leaks(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "SearchJobs",
                    "description": "Search jobs",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
        # 1. Empty tools -> quarantined
        self.assertEqual(
            classify_toolace_no_call([{"role": "assistant", "content": "Empty list."}], []),
            (None, "no_call_empty_tools"),
        )
        # 2. Markdown JSON tool leak -> quarantined
        self.assertEqual(
            classify_toolace_no_call(
                [
                    {
                        "role": "assistant",
                        "content": 'Cannot fetch details, searching instead:\n```\n[{"name": "SearchJobs"}]\n```',
                    }
                ],
                tools,
            ),
            (None, "no_call_suspect_leak"),
        )
        # 3. [No Need to Ask] -> quarantined
        self.assertEqual(
            classify_toolace_no_call(
                [{"role": "assistant", "content": "All args provided.\n\n[No Need to Ask]"}],
                tools,
            ),
            (None, "no_call_suspect_leak"),
        )
        # 4. Inline bracket call -> quarantined
        self.assertEqual(
            classify_toolace_no_call(
                [{"role": "assistant", "content": 'Try calling [SearchJobs(keyword="eng")].'}],
                tools,
            ),
            (None, "no_call_suspect_leak"),
        )
        # 5. Clean irrelevance refusal
        self.assertEqual(
            classify_toolace_no_call(
                [
                    {
                        "role": "assistant",
                        "content": "The provided functions do not include a tool for fetching tweets.",
                    }
                ],
                tools,
            ),
            ("toolace_irrelevant_refusal", None),
        )
        # 6. Clean missing param clarification
        self.assertEqual(
            classify_toolace_no_call(
                [
                    {
                        "role": "assistant",
                        "content": 'The query lacks the required "location" parameter. Please provide it.',
                    }
                ],
                tools,
            ),
            ("toolace_missing_param", None),
        )


class SmoltalkDistractorTests(unittest.TestCase):
    def _sample_tool_records(self) -> list[dict]:
        return [
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculate_triangle_area",
                            "description": "Calculate the area of a triangle.",
                            "parameters": {
                                "type": "object",
                                "properties": {"base": {"type": "number"}, "height": {"type": "number"}},
                                "required": ["base", "height"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "determine_body_mass_index",
                            "description": "Calculate body mass index given weight and height.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "weight": {"type": "number", "description": "Weight in kg"},
                                    "height": {"type": "number", "description": "Height in meters"},
                                },
                                "required": ["weight", "height"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_building_elevation",
                            "description": "Fetch elevation and height of a commercial building.",
                            "parameters": {
                                "type": "object",
                                "properties": {"building_id": {"type": "string"}},
                                "required": ["building_id"],
                            },
                        },
                    },
                ]
            }
        ]

    def test_distractor_index_excludes_math_tools_and_attaches_safe_distractors(self) -> None:
        idx = DistractorToolIndex(self._sample_tool_records())
        names = {t["function"]["name"] for t in idx.tools}
        self.assertNotIn("calculate_triangle_area", names)
        self.assertIn("determine_body_mass_index", names)
        self.assertIn("get_building_elevation", names)

        chosen = idx.sample_distractors(
            "Find the height and weight ratio for this geometry figure.",
            k=2,
            rng=random.Random(42),
        )
        self.assertEqual(len(chosen), 2)

    @unittest.skipUnless(QWEN35_SNAPSHOT.exists(), "Local Qwen3.5-2B snapshot required")
    def test_smoltalk_loader_and_qwen35_masking(self) -> None:
        idx = DistractorToolIndex(self._sample_tool_records())
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN35_SNAPSHOT, local_files_only=True, trust_remote_code=False
        )
        patched_tpl = patch_qwen35_chat_template(
            (QWEN35_SNAPSHOT / "chat_template.jinja").read_text(encoding="utf-8")
        )

        with tempfile.TemporaryDirectory() as td:
            sub_dir = Path(td) / "everyday-conversations"
            sub_dir.mkdir(parents=True)
            rows = [
                {
                    "messages": [
                        {"role": "user", "content": "What is the height of Everest in meters?"},
                        {"role": "assistant", "content": "Mount Everest is 8,848.86 meters tall."},
                        {"role": "user", "content": "Thanks! What about K2?"},
                        {"role": "assistant", "content": "K2 is 8,611 meters tall."},
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": "Custom system"},
                        {"role": "user", "content": "Skip me"},
                        {"role": "assistant", "content": "Skipped"},
                    ]
                },
            ]
            pq.write_table(pa.Table.from_pylist(rows), sub_dir / "train-00000-of-00001.parquet")

            recs, stats = load_smoltalk_irrelevant_records(
                smoltalk_dir=Path(td),
                subset_targets={"everyday-conversations": 5},
                distractor_index=idx,
                seed=7,
            )
            self.assertEqual(len(recs), 1)
            self.assertEqual(stats["everyday-conversations"]["skipped_rows"], 1)
            rec = recs[0]
            self.assertEqual(rec["meta"]["bucket"], "smoltalk_irrelevant_tools")
            self.assertGreaterEqual(len(rec["tools"]), 1)

            rendered = render_and_mask(tokenizer, rec, patched_tpl, verify_equivalence=True)
            self.assertEqual(len(rendered["assistant_segments"]), 2)
            self.assertIn("8,848.86", rendered["assistant_segments"][0])
            self.assertIn("8,611", rendered["assistant_segments"][1])


if __name__ == "__main__":
    unittest.main()
