"""Tests for GRPO dataset loading, config building, and Arrow-safe reward dispatch."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import yaml

from toolcall.training.grpo_reward import grpo_reward_func

ROOT = Path(__file__).resolve().parents[1]


class GrpoTrainingIntegrationTests(unittest.TestCase):
    def test_configs_parse_cleanly(self) -> None:
        for fname in (
            "grpo_smoke_2b.yaml",
            "grpo_v1_2b.yaml",
            "grpo_v2_smoke_2b_colocate.yaml",
            "grpo_v2_2b_colocate.yaml",
        ):
            cfg = yaml.safe_load((ROOT / "configs/train" / fname).read_text(encoding="utf-8"))
            self.assertEqual(cfg["training"]["loss_type"], "grpo")
            self.assertGreater(cfg["training"]["num_generations"], 0)
            self.assertGreater(cfg["training"]["top_k"], 0)
            self.assertIn(cfg["vllm"]["mode"], ("server", "colocate"))
            self.assertIn("merged-checkpoint-68", cfg["model"]["local_path"])

    def test_smoke_jsonl_roundtrips_into_reward_func(self) -> None:
        smoke_train = ROOT / "data/processed/grpo_smoke/train.jsonl"
        if not smoke_train.exists():
            self.skipTest("grpo_smoke not built locally")
        rows = [json.loads(line) for line in smoke_train.read_text(encoding="utf-8").splitlines()[:8]]
        # Simulate Arrow-loaded columns from grpo.load_grpo_dataset
        gt_jsons = [json.dumps(r["ground_truth"], ensure_ascii=False) for r in rows]
        tools_jsons = [json.dumps(r["tools"], ensure_ascii=False) for r in rows]
        task_types = [r["task_type"] for r in rows]
        buckets = [r["bucket"] for r in rows]
        # Give clean empty non-call text for all rows: non_call should get positive, call should get -2.0
        completions = ["I do not have a suitable tool for this request."] * len(rows)
        rewards = grpo_reward_func(
            completions,
            task_type=task_types,
            bucket=buckets,
            ground_truth_json=gt_jsons,
            tools_json=tools_jsons,
        )
        for r, rew in zip(rows, rewards, strict=True):
            if r["task_type"] == "call":
                self.assertAlmostEqual(rew, -2.0, places=6)
            else:
                self.assertGreaterEqual(rew, 2.5)

    def test_vllm_param_name_prefix_adapter_for_qwen3_5(self) -> None:
        from types import SimpleNamespace
        import tempfile
        from toolcall.training.grpo import install_vllm_param_name_prefix_adapter

        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp)
            (snap / "config.json").write_text(
                json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}),
                encoding="utf-8",
            )
            vllm_gen = SimpleNamespace(
                _fix_param_name_to_vllm=lambda name, extra_prefixes=None: name
            )
            trainer = SimpleNamespace(vllm_generation=vllm_gen)
            installed = install_vllm_param_name_prefix_adapter(
                trainer,
                model_config=SimpleNamespace(model_type="qwen3_5_text"),
                snapshot_dir=snap,
            )
            self.assertTrue(installed)
            self.assertEqual(
                vllm_gen._fix_param_name_to_vllm("model.layers.0.mlp.down_proj.weight"),
                "model.language_model.layers.0.mlp.down_proj.weight",
            )
            self.assertEqual(
                vllm_gen._fix_param_name_to_vllm("model.language_model.embed_tokens.weight"),
                "model.language_model.embed_tokens.weight",
            )


if __name__ == "__main__":
    unittest.main()
