"""Launch a benchmark in its isolated environment and report to SwanLab."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import yaml

from .common import read_json, read_jsonl, request_metrics, result, write_json
from .report import scalar_metrics, write_charts
from .telemetry import monitor, peak_metrics

WORKERS = {
    "bfcl-v4": ("bfcl", "bfcl"),
    "api-bank": ("api_bank", "api-bank"),
    "tau2": ("tau2", "tau2"),
}


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def endpoint(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError(
            "use an HTTP(S) endpoint URL without credentials, query or fragment"
        )
    return value.rstrip("/")


def run_slug(value):
    slug = re.sub(r"[^\w.-]+", "-", value.lower()).strip(".-")
    if not slug:
        raise argparse.ArgumentTypeError("use a name containing letters or digits")
    return slug


def model_slug(model):
    return run_slug(model.rstrip("/").rsplit("/", 1)[-1])


def experiment_name(settings, started_at):
    parts = (
        model_slug(settings["model"]),
        settings.get("variant"),
        settings["benchmark"],
        settings.get("profile"),
        started_at.strftime("%Y%m%d-%H%M%SZ"),
    )
    return "-".join(part for part in parts if part)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="benchmark", required=True)
    setup = commands.add_parser(
        "prepare", help="install pinned eval environments and download assets"
    )
    setup.add_argument("suite", choices=[*WORKERS, "all"])
    report = commands.add_parser("report", help="render and log a previously saved run")
    report.add_argument("run", type=Path)
    report.add_argument(
        "--swanlab",
        choices=("online", "off"),
        default="off",
        help="publish a saved run to SwanLab Cloud (default: redraw only)",
    )
    for name in WORKERS:
        command = commands.add_parser(name)
        profiles = (
            ("smoke", "full", "no-web", "core")
            if name == "bfcl-v4"
            else ("smoke", "full")
        )
        command.add_argument("--profile", choices=profiles, default="smoke")
        command.add_argument("--model", default="Qwen/Qwen3.5-2B")
        command.add_argument(
            "--base-url", type=endpoint, default="http://127.0.0.1:18003/v1"
        )
        command.add_argument("--output", type=Path)
        command.add_argument("--worker-python", type=Path)
        command.add_argument("--max-tokens", type=positive, default=1024)
        command.add_argument("--concurrency", type=positive)
        command.add_argument("--temperature", type=float, default=0)
        command.add_argument("--timeout", type=positive, default=180)
        command.add_argument("--thinking", action="store_true")
        command.add_argument(
            "--checkpoint-revision",
            help="checkpoint revision, digest or adapter identifier",
        )
        command.add_argument(
            "--gpu-index",
            type=int,
            help="local NVML device index when running on the serving host",
        )
        command.add_argument(
            "--variant", type=run_slug, help="short experiment label, e.g. sft-lora"
        )
        if name == "bfcl-v4":
            selection = command.add_mutually_exclusive_group()
            selection.add_argument("--categories", nargs="+")
            selection.add_argument(
                "--case-ids-file", type=Path, help="official BFCL category-to-IDs JSON"
            )
        if name == "tau2":
            command.add_argument(
                "--domains", nargs="+", choices=("retail", "airline", "telecom")
            )
            command.add_argument("--task-ids", nargs="+")
            command.add_argument(
                "--limit", type=positive, help="number of tasks per domain"
            )
            command.add_argument("--trials", type=positive)
            command.add_argument("--max-steps", type=positive)
            command.add_argument("--seed", type=int, default=300)
            command.add_argument("--user-model")
            command.add_argument("--user-base-url", type=endpoint)
            command.add_argument("--user-temperature", type=float, default=0)
    return cli


def load_defaults(root, benchmark, profile):
    file = {"bfcl-v4": "bfcl.yaml", "tau2": "tau2.yaml", "api-bank": "api-bank.yaml"}[
        benchmark
    ]
    document = yaml.safe_load((root / "configs/eval" / file).read_text())
    return {
        **{k: v for k, v in document.items() if k not in ("smoke", "full", "no-web")},
        **document.get(profile, {}),
    }


def serving_metadata(base_url, model):
    opener = build_opener(ProxyHandler({}))
    key = os.getenv("TOOLCALL_AGENT_API_KEY", "local")
    try:
        request = Request(
            base_url + "/models", headers={"Authorization": f"Bearer {key}"}
        )
        with opener.open(request, timeout=10) as response:
            models = json.load(response)["data"]
    except (URLError, TimeoutError, KeyError, ValueError) as error:
        raise RuntimeError(f"Cannot read models from {base_url}: {error}") from error
    served = next((row for row in models if row["id"] == model), None)
    if served is None:
        raise ValueError(f"Endpoint did not advertise '{model}'.")
    metadata = {"context_window": served.get("max_model_len")}
    try:
        with opener.open(
            base_url.removesuffix("/v1") + "/version", timeout=2
        ) as response:
            metadata["serving_version"] = json.load(response).get("version")
    except (URLError, TimeoutError, ValueError):
        pass
    return metadata


def launch(args, root):
    module, environment = WORKERS[args.benchmark]
    settings = load_defaults(root, args.benchmark, args.profile)
    settings.update(
        {
            k: v
            for k, v in vars(args).items()
            if v is not None and k not in {"worker_python", "output", "case_ids_file"}
        }
    )
    settings["workspace"] = str(root)
    if getattr(args, "case_ids_file", None):
        settings["fixed_case_ids"] = read_json(args.case_ids_file.resolve())
    if args.benchmark == "bfcl-v4" and (
        settings.get("categories") or settings.get("fixed_case_ids")
    ):
        settings["profile"] = "custom"
        settings.pop("test_category", None)
        settings.pop("case_ids", None)
    if (
        args.benchmark == "tau2"
        and settings.get("task_ids")
        and len(settings["domains"]) != 1
    ):
        raise ValueError("--task-ids requires exactly one domain")
    worker = (
        args.worker_python or root / "runs/tools" / environment / "bin/python"
    ).absolute()
    if not worker.is_file():
        raise FileNotFoundError(
            f"Missing worker {worker}. Run toolcall-eval prepare {args.benchmark}."
        )
    settings.update(serving_metadata(args.base_url, args.model))
    started_at = datetime.now(timezone.utc)
    settings["run_name"] = experiment_name(settings, started_at)
    run_id = "-".join(
        part
        for part in (
            settings["profile"],
            settings.get("variant"),
            started_at.strftime("%Y%m%dT%H%M%S.%fZ"),
        )
        if part
    )
    run = (
        args.output
        or root / "runs/eval" / args.benchmark / model_slug(args.model) / run_id
    ).resolve()
    run.mkdir(parents=True)
    write_json(run / "config.json", settings)
    env = {
        k: v
        for k, v in os.environ.items()
        if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }
    env.update(
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    print(f"Run directory: {run}", flush=True)
    start = time.monotonic()
    with (
        monitor(args.base_url, run / "resources.jsonl", args.gpu_index),
        (run / "benchmark.log").open("w") as log,
    ):
        child = subprocess.Popen(
            [str(worker), "-m", f"toolcall.evaluation.{module}", str(run)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
    elapsed = time.monotonic() - start
    if code:
        requests = read_jsonl(run / "model-requests.jsonl")
        summary = result(
            request_metrics(requests, elapsed, settings["max_tokens"]),
            {},
            "accuracy",
            status="failed",
        )
        summary["metrics"]["worker_exit_code"] = code
    else:
        summary = read_json(run / "summary.json")
        summary["status"] = (
            "completed_with_request_errors"
            if summary["metrics"].get("request_errors")
            else "completed"
        )
    resources = read_jsonl(run / "resources.jsonl")
    summary["metrics"].update(peak_metrics(resources), wall_seconds=elapsed)
    write_json(run / "summary.json", summary)
    report_run(run, root, upload=True)
    if code:
        raise RuntimeError(
            f"{args.benchmark} exited with {code}. See {run / 'benchmark.log'}."
        )


def report_run(run, root, *, upload):
    summary = read_json(run / "summary.json")
    resources = read_jsonl(run / "resources.jsonl")
    charts = write_charts(run, summary, resources)
    metrics = scalar_metrics(summary)
    write_json(run / "metrics.json", metrics)
    print(f"Summary: {run / 'summary.json'}", flush=True)
    if upload:
        from .swanlab_log import PROJECT, publish

        record = run / "swanlab.json"
        previous = read_json(record) if record.is_file() else {}
        if previous.get("id") and previous.get("project") == PROJECT:
            return
        config_path = run / "config.json"
        settings = read_json(config_path)
        settings["run_status"] = summary.get("status", "completed")
        name = settings.get("run_name")
        if not name:
            started_at = datetime.fromtimestamp(
                config_path.stat().st_mtime, timezone.utc
            )
            name = experiment_name(settings, started_at)
        log_dir = root / "runs/swanlab"
        identifier = publish(
            metrics,
            settings,
            charts,
            name=name,
            log_dir=log_dir,
            group=settings["benchmark"],
        )
        write_json(
            record,
            {
                "mode": "online",
                "project": PROJECT,
                "name": name,
                "id": identifier,
                "log_dir": str(log_dir),
            },
        )


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if args.benchmark == "prepare":
            from .assets import prepare

            prepare(Path.cwd(), args.suite)
        elif args.benchmark == "report":
            report_run(args.run.resolve(), Path.cwd(), upload=args.swanlab == "online")
        else:
            launch(args, Path.cwd())
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
