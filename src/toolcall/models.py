"""Fetch official Qwen snapshots into the shared ModelScope cache."""

import argparse
from pathlib import Path

import yaml
from modelscope import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="Model YAML files")
    parser.add_argument("--cache-dir", type=Path, help="Override the ModelScope cache")
    args = parser.parse_args()

    for path in args.configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_id = config["model_id"]
        directory = snapshot_download(
            model_id,
            revision=config["revision"],
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        print(f"{model_id}\t{Path(directory).resolve()}", flush=True)
