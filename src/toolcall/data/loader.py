"""Safe JSONL-to-tokenized HuggingFace Dataset loader avoiding PyArrow nested-dict merging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from .template import patch_qwen35_chat_template, render_and_mask


def load_tokenized_sft_dataset(
    jsonl_path: Path,
    tokenizer: Any,
    patched_template: str | None = None,
    max_records: int | None = None,
) -> Dataset:
    """Load OpenAI-style JSONL and pre-compute exact `input_ids` and assistant-only `labels`."""
    if patched_template is None:
        raw_tpl = tokenizer.chat_template
        if not raw_tpl:
            raise ValueError("Tokenizer has no chat_template and none was provided")
        patched_template = patch_qwen35_chat_template(raw_tpl)

    input_ids_col: list[list[int]] = []
    labels_col: list[list[int]] = []

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if max_records is not None and idx >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rendered = render_and_mask(tokenizer, rec, patched_template)
            ids = rendered["input_ids"]
            mask = rendered["assistant_masks"]
            labels = [tok if flag == 1 else -100 for tok, flag in zip(ids, mask, strict=True)]
            input_ids_col.append(ids)
            labels_col.append(labels)

    return Dataset.from_dict({"input_ids": input_ids_col, "labels": labels_col})
