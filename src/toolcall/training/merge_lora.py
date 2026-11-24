"""Merge trained Qwen3.5 LoRA adapters into standalone CausalLM weights for vLLM serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .preflight import resolve_model_snapshot


def _copy_official_inference_tokenizer_files(base_snapshot: Path, out_dir: Path) -> None:
    for fname in (
        "chat_template.jinja",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ):
        src = base_snapshot / fname
        if src.is_file():
            shutil.copy2(src, out_dir / fname)


def verify_merged_inference_contract(merged_dir: Path) -> dict[str, Any]:
    """Verify merged config and unmodified official chat_template.jinja tool-call rendering."""
    cfg = json.loads((merged_dir / "config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(
        merged_dir, local_files_only=True, trust_remote_code=False
    )
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Call lookup_city for Paris."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_city",
                    "description": "Look up a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert "<tools>" in rendered and "lookup_city" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return {
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
        "prompt_tokens_probe": len(tok.encode(rendered, add_special_tokens=False)),
    }


def merge_checkpoint(
    base_snapshot: Path,
    adapter_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Merge a LoRA adapter checkpoint with its base Qwen3.5 CausalLM and save to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)

    _copy_official_inference_tokenizer_files(base_snapshot, output_dir)
    contract = verify_merged_inference_contract(output_dir)

    info = {
        "base_snapshot": str(base_snapshot),
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "contract": contract,
    }
    (output_dir / "merge_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B", help="Base model ID")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Optional explicit path to base ModelScope snapshot",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Single LoRA checkpoint directory (e.g. runs/train/<exp>/checkpoint-xxx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target directory for single merged checkpoint",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory containing `checkpoint-*` folders (with --all-checkpoints)",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Merge every `checkpoint-*` inside --run-dir into `<run-dir>/merged-<ckpt>`",
    )
    args = parser.parse_args()

    base_snapshot = resolve_model_snapshot(args.model_id, args.base_model)

    if args.all_checkpoints:
        if not args.run_dir:
            parser.error("--all-checkpoints requires --run-dir")
        ckpts = sorted(
            (p for p in args.run_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint-* directories found under {args.run_dir}")
        results = []
        for ckpt in ckpts:
            target = args.run_dir / f"merged-{ckpt.name}"
            print(f"Merging {ckpt} -> {target}", flush=True)
            results.append(merge_checkpoint(base_snapshot, ckpt, target))
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not args.adapter_dir:
        parser.error("Provide --adapter-dir or --run-dir --all-checkpoints")
    output_dir = args.output_dir or (args.adapter_dir.parent / f"merged-{args.adapter_dir.name}")
    info = merge_checkpoint(base_snapshot, args.adapter_dir, output_dir)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
