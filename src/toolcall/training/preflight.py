"""Pre-flight verification helpers for SFT data integrity, template parity, and loss masking."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import file_digest
import json
from pathlib import Path
import re
from typing import Any

import torch

_FORBIDDEN_IN_LOSS_MARKERS = (
    "<|im_start|>system",
    "<|im_start|>user",
    "<tool_response>",
    "</tool_response>",
    "<think>\n\n</think>",
)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return file_digest(fh, "sha256").hexdigest()


def verify_data_manifest(data_dir: Path, required_files: list[str]) -> dict[str, str]:
    """Verify SHA-256 digests of frozen SFT artifacts against `manifest.json`."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}

    verified: dict[str, str] = {}
    for rel in required_files:
        target = data_dir / rel
        if not target.is_file():
            raise FileNotFoundError(f"Missing required SFT artifact: {target}")
        expected = (outputs.get(rel) or {}).get("sha256")
        if not expected:
            raise ValueError(f"Missing sha256 entry for {rel} in {manifest_path}")
        actual = _sha256_file(target)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {target}: expected {expected}, got {actual}"
            )
        verified[rel] = actual
    return verified


def verify_template_contract(
    tokenizer: Any,
    patched_template: str,
    sample_jsonl: Path,
    sample_count: int = 32,
) -> dict[str, Any]:
    """Verify `{% generation %}` support, distinct pad/eos IDs, and byte-level template equivalence."""
    if "{%- generation %}" not in patched_template and "{% generation %}" not in patched_template:
        raise ValueError("Training chat template is missing `{% generation %}` tags")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError(
            f"pad_token_id ({tokenizer.pad_token_id}) must differ from eos_token_id ({tokenizer.eos_token_id})"
        )

    checked = 0
    with sample_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if checked >= sample_count:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec["messages"]
            tools = rec.get("tools") or None

            orig = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            patched = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                chat_template=patched_template,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            if orig != patched:
                raise AssertionError(
                    f"Template mismatch at sample #{checked} ({rec.get('meta', {}).get('sample_id')})"
                )
            checked += 1

    return {
        "checked_samples": checked,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "byte_identical": True,
    }


def _extract_unmasked_segments(
    tokenizer: Any,
    ids: list[int],
    labels: list[int],
) -> list[str]:
    segments: list[str] = []
    idx = 0
    n = len(ids)
    while idx < n:
        if labels[idx] != -100:
            end = idx
            while end < n and labels[end] != -100:
                if labels[end] != ids[end]:
                    raise AssertionError(
                        f"Label mismatch at pos {end}: label={labels[end]} vs input_id={ids[end]}"
                    )
                end += 1
            segments.append(tokenizer.decode(ids[idx:end], skip_special_tokens=False))
            idx = end
        else:
            idx += 1
    return segments


def _format_batch_mask_view(
    tokenizer: Any,
    ids: list[int],
    labels: list[int],
    max_spans: int = 12,
) -> str:
    lines: list[str] = []
    idx = 0
    n = len(ids)
    spans = 0
    while idx < n and spans < max_spans:
        is_train = labels[idx] != -100
        end = idx
        while end < n and (labels[end] != -100) == is_train:
            end += 1
        tag = "TRAIN" if is_train else "MASK "
        snippet = tokenizer.decode(ids[idx:end], skip_special_tokens=False)
        if len(snippet) > 400 and not is_train:
            snippet = snippet[:180] + "\n... [TRUNCATED MASKED PROMPT] ...\n" + snippet[-180:]
        lines.append(f"--- [{tag}] tokens={end - idx} ---")
        lines.append(snippet)
        idx = end
        spans += 1
    return "\n".join(lines)


def _verify_packed_source_labels(
    batch: dict[str, Any], reference_dataset: Iterable[dict[str, Any]]
) -> int:
    ids = batch["input_ids"].reshape(-1).tolist()
    labels = batch["labels"].reshape(-1).tolist()
    bounds = batch["cu_seq_lens_q"].tolist()
    if (
        len(bounds) < 2
        or bounds[0] != 0
        or bounds[-1] != len(ids)
        or any(a >= b for a, b in zip(bounds, bounds[1:]))
    ):
        raise AssertionError(f"Invalid packed sequence boundaries: {bounds!r}")

    pending: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for start, stop in zip(bounds, bounds[1:]):
        pending.setdefault(tuple(ids[start:stop]), set()).add(tuple(labels[start:stop]))
    lengths = {len(key) for key in pending}
    for row in reference_dataset:
        original = row["input_ids"]
        if len(original) not in lengths:
            continue
        key = tuple(original)
        if key in pending:
            pending[key].discard(tuple(row["labels"]))
            if not pending[key]:
                del pending[key]
            if not pending:
                return len(bounds) - 1
    raise AssertionError(
        "Packed inputs or source labels differ from the original assistant-only tokenization"
    )


def inspect_dataloader_batches(
    dataloader: Any,
    tokenizer: Any,
    output_dir: Path,
    num_batches: int = 2,
    min_active_ratio: float = 0.03,
    max_active_ratio: float = 0.45,
    *,
    reference_dataset: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pull real training batches and assert assistant-only loss masking before training starts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_reports: list[dict[str, Any]] = []
    visual_dumps: list[str] = []

    iterator = iter(dataloader)
    for b_idx in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        input_ids: torch.Tensor = batch["input_ids"]
        labels: torch.Tensor = batch["labels"]
        total_tokens = int(labels.numel())
        active_tokens = int((labels != -100).sum().item())
        ratio = active_tokens / max(total_tokens, 1)

        if active_tokens == 0:
            raise AssertionError(f"Batch {b_idx} has ZERO trainable tokens (all -100)!")
        outlier = not (min_active_ratio <= ratio <= max_active_ratio)
        source_verified_sequences = None
        if reference_dataset is not None and "cu_seq_lens_q" in batch:
            source_verified_sequences = _verify_packed_source_labels(batch, reference_dataset)
        elif outlier:
            raise AssertionError(
                f"Batch {b_idx} trainable ratio {ratio:.4f} outside expected [{min_active_ratio}, {max_active_ratio}]; "
                "suspect broken loss mask!"
            )

        rows = input_ids.shape[0]
        decoded_segments_count = 0
        for r_idx in range(rows):
            ids_list = input_ids[r_idx].tolist()
            lbl_list = labels[r_idx].tolist()
            segments = _extract_unmasked_segments(tokenizer, ids_list, lbl_list)
            decoded_segments_count += len(segments)
            for seg in segments:
                for marker in _FORBIDDEN_IN_LOSS_MARKERS:
                    if marker in seg:
                        raise AssertionError(
                            f"Forbidden non-assistant marker {marker!r} leaked into trainable loss segment: {seg!r}"
                        )
                if not seg.endswith("<|im_end|>"):
                    raise AssertionError(
                        f"Trainable segment does not end with <|im_end|>: {seg[-80:]!r}"
                    )
            if r_idx == 0:
                visual_dumps.append(
                    f"================ BATCH {b_idx} ROW 0 (active={active_tokens}/{total_tokens}, ratio={ratio:.4f}) ================\n"
                    + _format_batch_mask_view(tokenizer, ids_list, lbl_list)
                )

        batch_reports.append(
            {
                "batch_index": b_idx,
                "shape": list(input_ids.shape),
                "has_cu_seq_lens_q": "cu_seq_lens_q" in batch,
                "total_tokens": total_tokens,
                "active_tokens": active_tokens,
                "active_ratio": round(ratio, 4),
                "active_ratio_outlier": outlier,
                "source_verified_sequences": source_verified_sequences,
                "assistant_segments": decoded_segments_count,
            }
        )

    if not batch_reports:
        raise RuntimeError("Train dataloader yielded zero batches during preflight check")

    report = {
        "status": "passed",
        "num_batches_checked": len(batch_reports),
        "batches": batch_reports,
    }
    (output_dir / "preflight_loss_mask_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "preflight_loss_mask_batches.txt").write_text(
        "\n\n".join(visual_dumps) + "\n", encoding="utf-8"
    )
    return report


def verify_model_forward_pass(trainer: Any, dataloader: Any) -> dict[str, Any]:
    """Run one real forward pass before training starts to catch kernel/cu_seqlens issues early."""
    batch = next(iter(dataloader))
    device = trainer.args.device
    gpu_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    was_training = trainer.model.training
    trainer.model.train()
    try:
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=bool(trainer.args.bf16 and device.type == "cuda"),
            ),
        ):
            outputs = trainer.model(**gpu_batch)
    finally:
        if not was_training:
            trainer.model.eval()

    loss_val = float(outputs.loss.detach().cpu())
    input_shape = list(gpu_batch["input_ids"].shape)
    has_cu = "cu_seq_lens_q" in gpu_batch
    del gpu_batch, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not torch.isfinite(torch.tensor(loss_val)):
        raise AssertionError(f"Non-finite loss in preflight forward pass: {loss_val}")
    return {
        "status": "passed",
        "device": str(device),
        "loss": round(loss_val, 4),
        "input_shape": input_shape,
        "packed_cu_seqlens": has_cu,
    }


def resolve_model_snapshot(model_id: str, override: Path | None = None) -> Path:
    """Resolve a local ModelScope Qwen snapshot directory cleanly."""
    if override is not None:
        resolved = override.expanduser().resolve()
        if not (resolved / "config.json").is_file():
            raise FileNotFoundError(f"Missing config.json in {resolved}")
        return resolved

    slug = re.sub(r"^Qwen/", "Qwen--", model_id.strip())
    candidate = Path.home() / ".cache/modelscope/models" / slug / "snapshots/master"
    if not (candidate / "config.json").is_file():
        raise FileNotFoundError(
            f"Missing local ModelScope snapshot for {model_id} at {candidate}. "
            "Run toolcall-download or pass --model-path."
        )
    return candidate
