"""Record a benchmark run's numeric metrics in SwanLab."""

import argparse
import json
from pathlib import Path

import swanlab


def publish(metrics, config, images, *, mode, project, name, log_dir, group=None):
    with swanlab.init(
        project=project,
        name=name,
        config=config,
        log_dir=str(log_dir),
        group=group,
        mode=mode,
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
    parser.add_argument("--mode", choices=("local", "online"), default="local")
    parser.add_argument("--project", default="toolcall-sft")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", type=Path, help="JSON snapshot of the run settings")
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if (
        not isinstance(metrics, dict)
        or not metrics
        or any(type(value) not in (float, int) for value in metrics.values())
    ):
        parser.error("metrics must be a nonempty JSON object with numeric values")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}

    publish(
        metrics,
        config,
        {},
        project=args.project,
        name=args.run_name,
        mode=args.mode,
        log_dir=Path("runs/swanlab"),
    )
