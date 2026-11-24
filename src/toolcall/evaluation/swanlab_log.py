"""Record a benchmark run's numeric metrics in SwanLab."""

import argparse
import json
from pathlib import Path

import swanlab

PROJECT = "toolcall-sft"

PROFILE_NOTES = {
    ("bfcl-v4", "core"): "BFCL V4 core：Non-Live + Live + Multi-turn 共 17 类 4,441 题；未跑 Web Search 与 Memory",
    ("bfcl-v4", "no-web"): "BFCL V4 no-web：19 类 4,906 题；未跑 Web Search 与 Memory",
    ("bfcl-v4", "full"): "BFCL V4 full：22 类全量（含 Web Search 与 Memory）",
    ("bfcl-v4", "smoke"): "BFCL V4 smoke：冒烟子集（3 题）",
    ("bfcl-v4", "custom"): "BFCL V4 自定义子集（--categories / --case-ids-file，类目见 config）",
    ("api-bank", "full"): "API-Bank full：314 段对话（753 工具调用 + 528 回复），L1/L2/L3 调用准确率",
    ("api-bank", "smoke"): "API-Bank smoke：冒烟子集",
    ("tau2", "full"): "τ²-Bench full：Retail 114 + Airline 50 + Telecom 114 共 278 题，pass^1 任务成功率",
    ("tau2", "smoke"): "τ²-Bench smoke：冒烟子集",
}


def describe(settings, status=None, metrics=None):
    """Build the short human-readable note shown under the run name on SwanLab."""
    key = (settings.get("benchmark"), settings.get("profile"))
    lines = [PROFILE_NOTES.get(key, f"{key[0]} ({key[1] or 'custom'})")]
    categories = settings.get("categories")
    if categories and key == ("bfcl-v4", "custom"):
        lines[0] = f"BFCL V4 自定义子集：{', '.join(categories)}"
    serving = ", ".join(
        part
        for part in (
            f"ctx={settings['context_window']}" if settings.get("context_window") else "",
            f"conc={settings['concurrency']}" if settings.get("concurrency") else "",
            f"temp={settings.get('temperature')}",
            f"max_tokens={settings.get('max_tokens')}",
            f"timeout={settings.get('timeout')}s",
            f"checkpoint={settings['checkpoint_revision'][:12]}"
            if settings.get("checkpoint_revision")
            else "",
        )
        if part
    )
    if serving:
        lines.append(f"协议：{serving}")
    if status:
        errors = []
        for name in ("request_errors", "context_rejections", "length_stops"):
            count = (metrics or {}).get(name, (metrics or {}).get(f"agent_{name}", 0))
            if count:
                errors.append(f"{name}={int(count)}")
        lines.append(
            f"状态 {status}"
            + (f"；系统级错误待归因：{', '.join(errors)}" if errors else "；无系统级错误")
        )
    return "\n".join(lines)


def publish(metrics, config, images, *, name, log_dir, group=None, description=None):
    with swanlab.init(
        project=PROJECT,
        name=name,
        config=config,
        description=description,
        log_dir=str(log_dir),
        group=group,
        mode="online",
        on_online_error="raise",
    ) as run:
        swanlab.log(
            {
                **metrics,
                **{
                    f"charts/{key}": swanlab.Image(str(path))
                    for key, path in images.items()
                },
            }
        )
        return getattr(run, "id", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics", type=Path, help="JSON object of metric names and values"
    )
    parser.add_argument("--run-name", required=True, help="descriptive experiment name")
    parser.add_argument("--config", type=Path, help="JSON snapshot of the run settings")
    parser.add_argument(
        "--description", help="short note shown under the run name on SwanLab"
    )
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if (
        not isinstance(metrics, dict)
        or not metrics
        or any(type(value) not in (float, int) for value in metrics.values())
    ):
        parser.error("metrics must be a nonempty JSON object with numeric values")
    if not args.run_name.strip():
        parser.error("--run-name must not be empty")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}

    publish(
        metrics,
        config,
        {},
        name=args.run_name.strip(),
        log_dir=Path("runs/swanlab"),
        description=args.description,
    )
