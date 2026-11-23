"""File and metric helpers shared by the benchmark workers."""

import json
import math
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def request_metrics(rows: list[dict], seconds: float, maximum: int) -> dict:
    success = [row for row in rows if "finish_reason" in row]
    failed = [row for row in rows if "error" in row]
    context = [
        row
        for row in failed
        if any(
            term in row["error"].lower()
            for term in ("context length", "max_model_len", "maximum model length")
        )
    ]
    length = [row for row in success if row["finish_reason"] == "length"]
    latencies = sorted(
        row["latency_seconds"] for row in success if "latency_seconds" in row
    )
    output = sum(row.get("output_tokens", 0) for row in success)
    input_tokens = sum(row.get("input_tokens", 0) for row in success)
    metrics = {
        "calls": len(rows),
        "successful_calls": len(success),
        "request_errors": len(failed),
        "context_rejections": len(context),
        "length_stops": len(length),
        "output_cap_stops": sum(row.get("output_tokens") == maximum for row in length),
        "input_tokens": input_tokens,
        "output_tokens": output,
        "output_tokens_per_second": output / seconds,
        "total_tokens_per_second": (output + input_tokens) / seconds,
    }
    if success:
        metrics["length_stop_rate"] = len(length) / len(success)
        metrics["output_cap_stop_rate"] = metrics["output_cap_stops"] / len(success)
    if latencies:
        metrics["latency_p95_seconds"] = latencies[math.ceil(len(latencies) * 0.95) - 1]
    return metrics


def result(metrics: dict, groups: dict, score_key: str, **details) -> dict:
    return {"metrics": metrics, "groups": groups, "score_key": score_key, **details}
