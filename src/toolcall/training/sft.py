"""SFT trainer entrypoint with strict manifest, template, and loss-mask preflight verification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import yaml

from ..data.loader import load_tokenized_sft_dataset
from .merge_lora import merge_checkpoint
from .preflight import (
    inspect_dataloader_batches,
    resolve_model_snapshot,
    verify_data_manifest,
    verify_model_forward_pass,
    verify_template_contract,
)

ROOT = Path(__file__).resolve().parents[3]
SWANLAB_PROJECT = "toolcall-sft"


def _compute_run_metadata(cfg: dict[str, Any], trainer: SFTTrainer) -> dict[str, Any]:
    args = trainer.args
    train_len = len(trainer.train_dataset)
    world_size = max(1, args.world_size)
    eff_bs = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(train_len / eff_bs)
    est_total_steps = math.ceil(steps_per_epoch * args.num_train_epochs)
    return {
        "experiment": cfg["experiment"],
        "model_id": cfg["model"]["model_id"],
        "train_packed_or_raw_rows": train_len,
        "effective_batch_size": eff_bs,
        "estimated_steps_per_epoch": steps_per_epoch,
        "estimated_total_steps": est_total_steps,
    }


def _check_flash_linear_attention() -> bool:
    try:
        import fla  # noqa: F401

        return True
    except ImportError:
        return False


def run_sft(
    config_path: Path,
    *,
    model_path: Path | None = None,
    preflight_only: bool = False,
    skip_sha_check: bool = False,
    swanlab_mode: str | None = None,
    output_dir_override: Path | None = None,
    no_packing: bool = False,
    use_kernels_override: bool | None = None,
    skip_forward_check: bool = False,
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp_name = cfg["experiment"]
    model_id = cfg["model"]["model_id"]
    data_cfg = cfg["data"]
    train_cfg = dict(cfg["training"])
    lora_cfg = cfg.get("lora") or {}

    if no_packing:
        train_cfg["packing"] = False
        train_cfg["padding_free"] = False
        # Without packing, keep per-device batch size at 2 (effective 32 with accum=16)
        # so worst-case long samples (~5k tokens) stay well within 40GB VRAM despite 248k-vocab fp32 logits.
        if int(train_cfg.get("per_device_train_batch_size", 1)) == 1:
            train_cfg["per_device_train_batch_size"] = 2
            train_cfg["per_device_eval_batch_size"] = 2
            train_cfg["gradient_accumulation_steps"] = 16

    packing = bool(train_cfg.get("packing", True))
    padding_free = bool(train_cfg.get("padding_free", True))
    if packing and not padding_free:
        raise ValueError(
            "packing=True requires padding_free=True on Qwen3.5 to prevent cross-sequence attention/conv leakage; "
            "pass --no-packing to run unpacked sequences safely without causal-conv1d."
        )

    data_dir = (ROOT / data_cfg["dir"]).resolve()
    train_file = data_cfg["train_file"]
    val_file = data_cfg.get("val_file")
    tpl_file = data_cfg["chat_template_file"]

    required = [train_file, tpl_file] + ([val_file] if val_file else [])
    if not skip_sha_check:
        verified_sha = verify_data_manifest(data_dir, required)
        print("Manifest SHA-256 verified:", verified_sha, flush=True)

    snapshot = resolve_model_snapshot(model_id, model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    patched_template = (data_dir / tpl_file).read_text(encoding="utf-8")

    # 1. Verify official chat_template vs patched {% generation %} template byte-identity
    tpl_report = verify_template_contract(
        tokenizer, patched_template, data_dir / train_file, sample_count=32
    )
    print("Template contract verified:", tpl_report, flush=True)

    # 2. Load datasets via Arrow-safe loader (prevents nested dict None-key merging)
    max_train_samples = data_cfg.get("max_train_samples")
    train_ds = load_tokenized_sft_dataset(
        data_dir / train_file,
        tokenizer,
        patched_template=patched_template,
        max_records=max_train_samples,
    )
    eval_ds = (
        load_tokenized_sft_dataset(
            data_dir / val_file, tokenizer, patched_template=patched_template
        )
        if val_file
        else None
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        output_dir_override.resolve()
        if output_dir_override
        else (ROOT / "runs/train" / f"{exp_name}-{stamp}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    has_fla = _check_flash_linear_attention()
    print(f"Environment check: flash-linear-attention={has_fla}, cuda={torch.cuda.is_available()}", flush=True)

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(snapshot, **model_kwargs)
    model.config.use_cache = False

    use_kernels = (
        use_kernels_override
        if use_kernels_override is not None
        else bool(train_cfg.get("use_kernels", False))
    )
    if use_kernels and torch.cuda.is_available():
        from kernels import Mode

        # Hub kernels (e.g. kernels-community/mamba-ssm causal_conv1d_fn) only match CUDA
        # and must be kernelized in Mode.TRAINING after moving off CPU.
        model.to("cuda")
        model.train()
        model.set_use_kernels(True, mode=Mode.TRAINING)
        print("Activated HF Hub kernels in Mode.TRAINING on CUDA", flush=True)

    peft_config = None
    if lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 64)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 128)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_cfg.get("target_modules", "all-linear"),
        )

    use_eval = eval_ds is not None and float(train_cfg.get("eval_steps", 0)) > 0
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        run_name=f"{exp_name}-{stamp}",
        seed=int(train_cfg.get("seed", 42)),
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        max_length=int(train_cfg.get("max_length", 8192)),
        packing=packing,
        packing_strategy=str(train_cfg.get("packing_strategy", "bfd")),
        padding_free=padding_free,
        # Pre-tokenized `train_ds`/`eval_ds` already carry exact assistant-only `-100` labels built
        # from `qwen35_chat_template_train.jinja`; TRL preserves `labels` as-is and would reject
        # pre-tokenized non-conversational columns if `assistant_only_loss=True` were passed here.
        assistant_only_loss=False,
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        learning_rate=float(train_cfg.get("learning_rate", 1.5e-4)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_steps=float(train_cfg.get("warmup_steps", 0.05)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 2)),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        eval_strategy="steps" if use_eval else "no",
        eval_steps=float(train_cfg.get("eval_steps", 0.25)) if use_eval else None,
        save_strategy="steps",
        save_steps=float(train_cfg.get("save_steps", 0.25)),
        save_total_limit=int(train_cfg.get("save_total_limit", 4)),
        load_best_model_at_end=bool(use_eval and train_cfg.get("load_best_model_at_end", False)),
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
        report_to="none",
    )

    callbacks: list[Any] = []
    mode = swanlab_mode or cfg.get("swanlab", {}).get("mode", "online")
    if not preflight_only and mode != "disabled":
        from swanlab.integration.transformers import SwanLabCallback

        callbacks.append(
            SwanLabCallback(
                project=cfg.get("swanlab", {}).get("project", SWANLAB_PROJECT),
                experiment_name=f"{exp_name}-{stamp}",
                description=cfg.get("description", exp_name),
                log_dir=str((ROOT / "runs/swanlab").resolve()),
                mode=mode,
                tags=[exp_name, model_id.split("/")[-1], "sft-v1"],
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    run_meta = _compute_run_metadata(cfg, trainer)
    print("Run plan:", json.dumps(run_meta, ensure_ascii=False), flush=True)

    # 3. Runtime DataLoader batch verification BEFORE step 0
    dataloader = trainer.get_train_dataloader()
    mask_report = inspect_dataloader_batches(
        dataloader,
        tokenizer,
        out_dir,
        num_batches=2,
    )
    print("Preflight loss-mask batch check passed:", json.dumps(mask_report, ensure_ascii=False), flush=True)

    forward_report = None
    if not skip_forward_check and torch.cuda.is_available():
        forward_report = verify_model_forward_pass(trainer, dataloader)
        print("Preflight GPU forward pass passed:", json.dumps(forward_report, ensure_ascii=False), flush=True)

    if preflight_only:
        summary = {
            "status": "preflight_passed",
            "output_dir": str(out_dir),
            "template_contract": tpl_report,
            "run_meta": run_meta,
            "loss_mask_check": mask_report,
            "forward_check": forward_report,
        }
        (out_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.monotonic()
    train_result = trainer.train()
    wall_seconds = round(time.monotonic() - t0, 2)

    trainer.save_model(str(out_dir / "final_adapter"))
    eval_metrics = trainer.evaluate() if eval_ds is not None else {}

    peak_vram_gib = (
        round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if torch.cuda.is_available()
        else 0.0
    )

    summary = {
        "status": "completed",
        "output_dir": str(out_dir),
        "wall_seconds": wall_seconds,
        "peak_vram_gib": peak_vram_gib,
        "flash_linear_attention": has_fla,
        "use_kernels": use_kernels,
        "packing": packing,
        "padding_free": padding_free,
        "run_meta": run_meta,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "loss_mask_check": mask_report,
        "forward_check": forward_report,
    }

    if cfg.get("auto_merge_best") and trainer.state.best_model_checkpoint:
        best_ckpt = Path(trainer.state.best_model_checkpoint)
        merged_dir = out_dir / "merged-best"
        print(f"Auto-merging best checkpoint {best_ckpt} -> {merged_dir}", flush=True)
        summary["merged_best"] = merge_checkpoint(snapshot, best_ckpt, merged_dir)

    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Train YAML config")
    parser.add_argument("--model-path", type=Path, default=None, help="Override local model snapshot")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate SHA-256, template, packing, and loss masks without launching training",
    )
    parser.add_argument(
        "--skip-sha-check",
        action="store_true",
        help="Skip manifest.json SHA-256 check (only for custom temporary datasets)",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "offline", "local", "disabled"],
        default=None,
        help="Override SwanLab mode",
    )
    parser.add_argument(
        "--no-packing",
        action="store_true",
        help="Disable sequence packing and padding_free (safe fallback if causal-conv1d packed kernel is unavailable)",
    )
    parser.add_argument(
        "--no-use-kernels",
        action="store_true",
        help="Disable HF Hub kernels (use_kernels=False)",
    )
    parser.add_argument(
        "--skip-forward-check",
        action="store_true",
        help="Skip the 1-batch GPU forward probe during preflight",
    )
    args = parser.parse_args()

    summary = run_sft(
        args.config.resolve(),
        model_path=args.model_path,
        preflight_only=args.preflight_only,
        skip_sha_check=args.skip_sha_check,
        swanlab_mode=args.swanlab_mode,
        output_dir_override=args.output_dir,
        no_packing=args.no_packing,
        use_kernels_override=False if args.no_use_kernels else None,
        skip_forward_check=args.skip_forward_check,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
