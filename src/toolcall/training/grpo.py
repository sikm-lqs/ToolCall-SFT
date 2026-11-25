"""GRPO trainer entrypoint for Qwen3.5 tool calling using ToolRL + 3.4-B symmetric rewards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from datasets import Dataset
from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import yaml

from .grpo_reward import grpo_reward_func
from .preflight import resolve_model_snapshot, verify_data_manifest

ROOT = Path(__file__).resolve().parents[3]
SWANLAB_PROJECT = "toolcall-sft"


def load_grpo_dataset(jsonl_path: Path, max_records: int | None = None) -> Dataset:
    """Load GRPO JSONL using JSON-serialized nested columns to avoid PyArrow schema collisions."""
    prompts: list[str] = []
    task_types: list[str] = []
    buckets: list[str] = []
    gt_jsons: list[str] = []
    tools_jsons: list[str] = []

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if max_records is not None and idx >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row["prompt"])
            if not prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
                raise ValueError(f"Prompt at row {idx} missing Qwen3.5 non-thinking suffix")
            prompts.append(prompt)
            task_types.append(str(row["task_type"]))
            buckets.append(str(row["bucket"]))
            gt_jsons.append(json.dumps(row["ground_truth"], ensure_ascii=False))
            tools_jsons.append(json.dumps(row.get("tools") or [], ensure_ascii=False))

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "task_type": task_types,
            "bucket": buckets,
            "ground_truth_json": gt_jsons,
            "tools_json": tools_jsons,
        }
    )


def _step_or_ratio(val: Any, default: float | int) -> float | int:
    v = float(default if val is None else val)
    return int(v) if v >= 1.0 else v


def build_grpo_config(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    stamp: str,
    use_eval: bool,
    no_vllm: bool = False,
) -> GRPOConfig:
    """Construct TRL GRPOConfig from experiment YAML dictionary."""
    exp_name = cfg["experiment"]
    train_cfg = cfg["training"]
    vllm_cfg = cfg.get("vllm") or {}
    use_vllm = False if no_vllm else bool(vllm_cfg.get("enabled", True))
    vllm_mode = str(vllm_cfg.get("mode", "server"))

    kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "run_name": f"{exp_name}-{stamp}",
        "seed": int(train_cfg.get("seed", 42)),
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 8)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "num_generations": int(train_cfg.get("num_generations", 4)),
        "num_generations_eval": int(train_cfg.get("num_generations_eval", 4)),
        "max_completion_length": int(train_cfg.get("max_completion_length", 512)),
        "temperature": float(train_cfg.get("temperature", 0.7)),
        "top_p": float(train_cfg.get("top_p", 0.95)),
        "top_k": int(train_cfg.get("top_k", 512)),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-5)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "warmup_steps": _step_or_ratio(train_cfg.get("warmup_steps"), 0.05),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1.0)),
        "beta": float(train_cfg.get("beta", 0.02)),
        "loss_type": str(train_cfg.get("loss_type", "grpo")),
        "use_vllm": use_vllm,
        "vllm_mode": vllm_mode,
        "vllm_server_base_url": vllm_cfg.get("server_base_url", "http://127.0.0.1:8000"),
        "vllm_server_timeout": float(vllm_cfg.get("server_timeout", 300.0)),
        "vllm_group_port": int(vllm_cfg.get("group_port", 51216)),
        "vllm_importance_sampling_correction": bool(
            vllm_cfg.get("importance_sampling_correction", True)
        ),
        "vllm_tensor_parallel_size": int(vllm_cfg.get("tensor_parallel_size", 1)),
        "vllm_gpu_memory_utilization": float(vllm_cfg.get("gpu_memory_utilization", 0.35)),
        "vllm_enable_sleep_mode": bool(vllm_cfg.get("enable_sleep_mode", False)),
        "logging_steps": int(train_cfg.get("logging_steps", 5)),
        "eval_strategy": "steps" if use_eval else "no",
        "eval_steps": _step_or_ratio(train_cfg.get("eval_steps"), 0.5) if use_eval else None,
        "eval_on_start": bool(train_cfg.get("eval_on_start", False)) if use_eval else False,
        "save_strategy": "steps",
        "save_steps": _step_or_ratio(train_cfg.get("save_steps"), 0.5),
        "save_total_limit": int(train_cfg.get("save_total_limit", 4)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 2)),
        "ddp_find_unused_parameters": bool(train_cfg.get("ddp_find_unused_parameters", False)),
        "report_to": "none",
    }
    if vllm_cfg.get("max_model_length"):
        kwargs["vllm_max_model_length"] = int(vllm_cfg["max_model_length"])
    return GRPOConfig(**kwargs)


def run_grpo(
    config_path: Path,
    *,
    model_path: Path | None = None,
    preflight_only: bool = False,
    skip_sha_check: bool = False,
    swanlab_mode: str | None = None,
    output_dir_override: Path | None = None,
    no_vllm: bool = False,
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp_name = cfg["experiment"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    lora_cfg = cfg.get("lora") or {}

    data_dir = (ROOT / data_cfg["dir"]).resolve()
    train_file = data_cfg["train_file"]
    val_file = data_cfg.get("val_file")
    required = [train_file] + ([val_file] if val_file else [])
    if not skip_sha_check:
        verified_sha = verify_data_manifest(data_dir, required)
        print("Manifest SHA-256 verified:", verified_sha, flush=True)

    train_ds = load_grpo_dataset(
        data_dir / train_file, max_records=data_cfg.get("max_train_samples")
    )
    eval_ds = load_grpo_dataset(data_dir / val_file) if val_file else None

    explicit_model_path = model_path
    if explicit_model_path is None and model_cfg.get("local_path"):
        cand = Path(model_cfg["local_path"]).expanduser()
        explicit_model_path = cand if cand.is_absolute() else (ROOT / cand).resolve()
    snapshot = resolve_model_snapshot(model_cfg["model_id"], explicit_model_path)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    stamp = os.environ.get("RUN_STAMP") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        output_dir_override.resolve()
        if output_dir_override
        else (ROOT / "runs/train" / f"{exp_name}-{stamp}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (out_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    eff_prompts_per_step = (
        int(train_cfg.get("per_device_train_batch_size", 4))
        * int(train_cfg.get("gradient_accumulation_steps", 8))
        * world_size
    ) // int(train_cfg.get("num_generations", 4))
    est_steps_per_epoch = math.ceil(len(train_ds) / max(1, eff_prompts_per_step))
    run_meta = {
        "experiment": exp_name,
        "base_model_snapshot": str(snapshot),
        "world_size": world_size,
        "vllm_mode": (cfg.get("vllm") or {}).get("mode", "server"),
        "train_rows": len(train_ds),
        "val_rows": len(eval_ds) if eval_ds is not None else 0,
        "num_generations": int(train_cfg.get("num_generations", 4)),
        "effective_prompts_per_step": eff_prompts_per_step,
        "estimated_steps_per_epoch": est_steps_per_epoch,
        "estimated_total_steps": math.ceil(
            est_steps_per_epoch * float(train_cfg.get("num_train_epochs", 1.0))
        ),
    }
    if rank == 0:
        print("GRPO run plan:", json.dumps(run_meta, ensure_ascii=False), flush=True)

    if preflight_only:
        summary = {
            "status": "preflight_passed",
            "output_dir": str(out_dir),
            "run_meta": run_meta,
        }
        if rank == 0:
            (out_dir / "train_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        return summary

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        ),
    )
    model.config.use_cache = False

    if bool(train_cfg.get("use_kernels", True)) and torch.cuda.is_available():
        from kernels import Mode

        model.to(torch.device("cuda", local_rank))
        model.train()
        model.set_use_kernels(True, mode=Mode.TRAINING)
        print(f"[rank {rank}] Activated HF Hub kernels in Mode.TRAINING on cuda:{local_rank}", flush=True)

    peft_config = None
    if lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_cfg.get("target_modules", "all-linear"),
        )

    use_eval = eval_ds is not None and float(train_cfg.get("eval_steps", 0)) > 0
    grpo_args = build_grpo_config(
        cfg, out_dir=out_dir, stamp=stamp, use_eval=use_eval, no_vllm=no_vllm
    )

    callbacks: list[Any] = []
    mode = swanlab_mode or cfg.get("swanlab", {}).get("mode", "online")
    if mode != "disabled" and rank == 0:
        from swanlab.integration.transformers import SwanLabCallback

        callbacks.append(
            SwanLabCallback(
                project=cfg.get("swanlab", {}).get("project", SWANLAB_PROJECT),
                experiment_name=f"{exp_name}-{stamp}",
                description=cfg.get("description", exp_name),
                log_dir=str((ROOT / "runs/swanlab").resolve()),
                mode=mode,
                tags=[exp_name, "grpo", "qwen3.5-2b"],
            )
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_func,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    install_vllm_param_name_prefix_adapter(
        trainer, model_config=model.config, snapshot_dir=snapshot
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.monotonic()
    train_result = trainer.train()
    wall_seconds = round(time.monotonic() - t0, 2)
    trainer.save_model(str(out_dir / "final_adapter"))

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
        "run_meta": run_meta,
        "train_metrics": train_result.metrics,
    }
    if rank == 0:
        (out_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return summary


def install_vllm_param_name_prefix_adapter(
    trainer: Any, *, model_config: Any, snapshot_dir: Path
) -> bool:
    """Prefix text-only Qwen3_5ForCausalLM parameter names when syncing to VL vLLM."""
    vllm_gen = getattr(trainer, "vllm_generation", None)
    if vllm_gen is None or getattr(model_config, "model_type", "") != "qwen3_5_text":
        return False
    cfg_path = snapshot_dir / "config.json"
    if not cfg_path.exists():
        return False
    archs = json.loads(cfg_path.read_text(encoding="utf-8")).get("architectures") or []
    if "Qwen3_5ForConditionalGeneration" not in archs:
        return False

    orig_fix = vllm_gen._fix_param_name_to_vllm

    def _adapted_fix(name: str, extra_prefixes: list[str] | None = None) -> str:
        fixed = orig_fix(name, extra_prefixes=extra_prefixes)
        if fixed.startswith("model.") and not fixed.startswith("model.language_model."):
            return "model.language_model." + fixed.removeprefix("model.")
        return fixed

    vllm_gen._fix_param_name_to_vllm = _adapted_fix
    print(
        "Enabled vLLM weight-name prefix adapter: model.* -> model.language_model.*",
        flush=True,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="GRPO YAML config")
    parser.add_argument("--model-path", type=Path, default=None, help="Override base/merged model path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-sha-check", action="store_true")
    parser.add_argument("--swanlab-mode", choices=["online", "offline", "local", "disabled"], default=None)
    parser.add_argument("--no-vllm", action="store_true", help="Fallback to native HF generate without vLLM")
    args = parser.parse_args()

    summary = run_grpo(
        args.config.resolve(),
        model_path=args.model_path,
        preflight_only=args.preflight_only,
        skip_sha_check=args.skip_sha_check,
        swanlab_mode=args.swanlab_mode,
        output_dir_override=args.output_dir,
        no_vllm=args.no_vllm,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
