"""Unit and artifact-contract tests for GRPO Smoke and v1 dataset construction."""

from __future__ import annotations

import json
from pathlib import Path
import random
import unittest

from transformers import AutoTokenizer

from toolcall.data.build_grpo import (
    _build_adversarial_1tool_irrelevance_pool,
    _build_miss_func_recover_pool,
    _clean_messages,
    _render_grpo_prompt,
    _verify_token_equivalence,
)

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = Path("~/.cache/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master").expanduser()


class GrpoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            TOK_DIR, local_files_only=True, trust_remote_code=False
        )

    def test_clean_messages_rejects_forbidden_policy(self) -> None:
        with self.assertRaises(AssertionError):
            _clean_messages(
                [
                    {"role": "system", "content": "If none of the function can be used, refuse."},
                    {"role": "user", "content": "Hi"},
                ]
            )

    def test_prompt_render_matches_official_template_tokens(self) -> None:
        msgs = [{"role": "user", "content": "Check temperature in Paris."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_temp",
                    "description": "Get city temperature",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        prompt, n_tok = _render_grpo_prompt(self.tokenizer, msgs, tools)
        self.assertGreater(n_tok, 20)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertIn("<tools>", prompt)
        _verify_token_equivalence(self.tokenizer, prompt, msgs, tools)

    def test_miss_func_recover_and_adversarial_irrelevance_builders(self) -> None:
        rec = {
            "messages": [
                {"role": "user", "content": "Book a flight from Beijing to Tokyo."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "book_flight",
                                "arguments": {"origin": "Beijing", "destination": "Tokyo"},
                            },
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "book_flight",
                        "description": "Book airline tickets between cities.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "origin": {"type": "string"},
                                "destination": {"type": "string"},
                            },
                            "required": ["origin", "destination"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "compute_bmi",
                        "description": "Compute body mass index from weight and height.",
                        "parameters": {
                            "type": "object",
                            "properties": {"weight": {"type": "number"}, "height": {"type": "number"}},
                            "required": ["weight", "height"],
                        },
                    },
                },
            ],
            "meta": {"source": "xlam-60k", "source_id": "test_1"},
        }
        rng = random.Random(42)
        recover = _build_miss_func_recover_pool([rec], self.tokenizer, 4096, set(), set(), rng, limit=5)
        self.assertEqual(len(recover), 1)
        self.assertEqual(recover[0]["task_type"], "call")
        self.assertEqual(recover[0]["bucket"], "multi_turn_miss_func_recover")
        self.assertEqual(recover[0]["ground_truth"]["tool_calls"][0]["name"], "book_flight")

        adv = _build_adversarial_1tool_irrelevance_pool([rec], self.tokenizer, 4096, set(), set(), rng, limit=5)
        self.assertEqual(len(adv), 1)
        self.assertEqual(adv[0]["task_type"], "non_call")
        self.assertEqual(adv[0]["meta"]["num_candidate_tools"], 1)
        self.assertEqual(adv[0]["tools"][0]["function"]["name"], "compute_bmi")

    def test_frozen_grpo_artifacts_if_present(self) -> None:
        for name, exp_train, exp_val in (("grpo_smoke", 300, 60), ("grpo_v1", 1500, 150)):
            d = ROOT / "data/processed" / name
            if not (d / "manifest.json").exists():
                continue
            manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["outputs"]["train.jsonl"]["count"], exp_train)
            self.assertEqual(manifest["outputs"]["val.jsonl"]["count"], exp_val)
            self.assertEqual(manifest["invariants"]["train_val_group_overlap"], 0)
            self.assertEqual(manifest["invariants"]["excluded_val_overlap"], 0)
            self.assertTrue(manifest["invariants"]["token_equivalence_verified"])


if __name__ == "__main__":
    unittest.main()
