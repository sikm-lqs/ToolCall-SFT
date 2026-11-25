"""Unit tests for SFT training preflight checks, manifest SHA validation, and loss-mask inspection."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from datasets import Dataset
import torch
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling, pack_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolcall.data.loader import load_tokenized_sft_dataset  # noqa: E402
from toolcall.data.template import render_and_mask  # noqa: E402
from toolcall.training.preflight import (  # noqa: E402
    inspect_dataloader_batches,
    verify_data_manifest,
    verify_template_contract,
)
from toolcall.training.sft import run_sft  # noqa: E402

SFT_V1_DIR = ROOT / "data/processed/sft_v1"
QWEN35_SNAPSHOT = Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"


@unittest.skipUnless(
    SFT_V1_DIR.exists() and QWEN35_SNAPSHOT.exists(),
    "Built SFT-v1 dataset and local Qwen3.5-2B snapshot required",
)
class SftTrainingPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            QWEN35_SNAPSHOT, local_files_only=True, trust_remote_code=False
        )
        cls.patched_tpl = (SFT_V1_DIR / "qwen35_chat_template_train.jinja").read_text(
            encoding="utf-8"
        )

    def test_manifest_sha256_and_template_contract_pass(self) -> None:
        verified = verify_data_manifest(
            SFT_V1_DIR,
            ["train.jsonl", "val.jsonl", "smoke_v0.jsonl", "qwen35_chat_template_train.jinja"],
        )
        self.assertEqual(len(verified), 4)
        report = verify_template_contract(
            self.tokenizer, self.patched_tpl, SFT_V1_DIR / "val.jsonl", sample_count=20
        )
        self.assertTrue(report["byte_identical"])
        self.assertNotEqual(report["pad_token_id"], report["eos_token_id"])

    def test_dataloader_batch_inspection_passes_on_packed_val_slice(self) -> None:
        ds = load_tokenized_sft_dataset(
            SFT_V1_DIR / "val.jsonl",
            self.tokenizer,
            patched_template=self.patched_tpl,
            max_records=16,
        )
        packed = pack_dataset(ds, seq_length=4096, strategy="bfd")
        collator = DataCollatorForLanguageModeling(
            pad_token_id=self.tokenizer.pad_token_id, padding_free=True
        )
        batches = [collator([packed[i] for i in range(min(2, len(packed)))])]

        with tempfile.TemporaryDirectory() as td:
            report = inspect_dataloader_batches(
                batches, self.tokenizer, Path(td), num_batches=1
            )
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["batches"][0]["has_cu_seq_lens_q"])
            self.assertTrue((Path(td) / "preflight_loss_mask_batches.txt").is_file())

    def test_assistant_heavy_pack_requires_exact_source_labels(self) -> None:
        rendered = render_and_mask(
            self.tokenizer,
            {"messages": [
                {"role": "user", "content": "Give a long answer."},
                {"role": "assistant", "content": "A long natural answer. " * 60},
            ]},
            self.patched_tpl,
        )
        ids = rendered["input_ids"]
        labels = [
            token if active else -100
            for token, active in zip(ids, rendered["assistant_masks"], strict=True)
        ]
        reference = Dataset.from_list([{"input_ids": ids, "labels": labels}])
        packed = pack_dataset(reference, seq_length=512, strategy="bfd")
        collator = DataCollatorForLanguageModeling(
            pad_token_id=self.tokenizer.pad_token_id, padding_free=True
        )
        batch = collator([packed[0]])
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(AssertionError, "trainable ratio"):
                inspect_dataloader_batches([batch], self.tokenizer, Path(td), num_batches=1)
            report = inspect_dataloader_batches(
                [batch], self.tokenizer, Path(td), num_batches=1, reference_dataset=reference
            )
            checked = report["batches"][0]
            self.assertGreater(checked["active_ratio"], 0.45)
            self.assertEqual(checked["source_verified_sequences"], 1)

            first_active = rendered["assistant_masks"].index(1)
            altered = {**batch, "labels": batch["labels"].clone()}
            altered["labels"][0, first_active] = -100
            with self.assertRaisesRegex(AssertionError, "source labels"):
                inspect_dataloader_batches(
                    [altered], self.tokenizer, Path(td), num_batches=1, reference_dataset=reference
                )

    def test_dataloader_inspection_rejects_leaked_user_tokens(self) -> None:
        ids = self.tokenizer.encode("<|im_start|>user\nhi<|im_end|>\n", add_special_tokens=False)
        bad_batch = [
            {
                "input_ids": torch.tensor([ids]),
                "labels": torch.tensor([ids]),  # Everything unmasked -> must fail!
            }
        ]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AssertionError):
                inspect_dataloader_batches(bad_batch, self.tokenizer, Path(td), num_batches=1)

    def test_packing_without_padding_free_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "bad_cfg.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "experiment: bad-cfg",
                        "model:",
                        "  model_id: Qwen/Qwen3.5-0.8B",
                        "data:",
                        "  dir: data/processed/sft_v1",
                        "  train_file: val.jsonl",
                        "  chat_template_file: qwen35_chat_template_train.jinja",
                        "training:",
                        "  packing: true",
                        "  padding_free: false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                run_sft(cfg_path, preflight_only=True)


if __name__ == "__main__":
    unittest.main()
