"""Regression checks for Qwen3.5 export to the vLLM serving architecture."""

from pathlib import Path
import shutil
import tempfile
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from toolcall.training.merge_lora import merge_checkpoint


BASE_TOKENIZER = Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"


@unittest.skipUnless(BASE_TOKENIZER.is_dir(), "Local Qwen3.5 tokenizer required")
class SftMergeLoraTests(unittest.TestCase):
    def test_merge_exports_original_wrapper_with_trained_text_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, adapter, merged = (Path(tmp) / name for name in ("base", "adapter", "merged"))
            cfg = Qwen3_5Config(
                text_config={
                    "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
                    "num_hidden_layers": 1, "head_dim": 8, "num_attention_heads": 4,
                    "num_key_value_heads": 2, "layer_types": ["full_attention"],
                },
                vision_config={
                    "depth": 1, "hidden_size": 16, "intermediate_size": 24,
                    "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16,
                },
            )
            original = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16)
            original.save_pretrained(base)
            for name in ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
                shutil.copy2(BASE_TOKENIZER / name, base / name)
            (base / "preprocessor_config.json").write_text('{"image_processor_type":"Qwen2VLImageProcessor"}')

            text_model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
            tuned = get_peft_model(
                text_model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"])
            )
            proj = tuned.base_model.model.model.layers[0].self_attn.q_proj
            before = proj.base_layer.weight.detach().clone()
            with torch.no_grad():
                proj.lora_A["default"].weight.fill_(0.125)
                proj.lora_B["default"].weight.fill_(0.125)
            tuned.save_pretrained(adapter)

            info = merge_checkpoint(base, adapter, merged)
            self.assertEqual(info["contract"]["architectures"], ["Qwen3_5ForConditionalGeneration"])
            self.assertEqual(info["contract"]["model_type"], "qwen3_5")
            self.assertEqual(
                (merged / "chat_template.jinja").read_bytes(),
                (base / "chat_template.jinja").read_bytes(),
            )
            self.assertTrue((merged / "preprocessor_config.json").is_file())
            served, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
                merged, output_loading_info=True, dtype=torch.bfloat16
            )
            self.assertFalse(loading["missing_keys"])
            self.assertFalse(loading["unexpected_keys"])
            after = served.model.language_model.layers[0].self_attn.q_proj.weight
            self.assertFalse(torch.equal(before, after))
            torch.testing.assert_close(after, before + 0.03125, rtol=0, atol=0)
            torch.testing.assert_close(
                served.model.visual.patch_embed.proj.weight,
                original.model.visual.patch_embed.proj.weight,
                rtol=0, atol=0,
            )
