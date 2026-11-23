"""Run the pinned Sierra simulator with OpenAI-compatible agent and user models."""

from collections import Counter
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
from threading import Lock
import time

from .common import read_json, read_jsonl, request_metrics, result, write_json


def run_tau2(run: Path):
    import tau2
    from tau2.data_model.simulation import RunConfig
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.run import run_domain
    from tau2.utils import llm_utils

    config = read_json(run / "config.json")
    checkout = Path(tau2.__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config["git_commit"]:
        raise RuntimeError(
            f"τ² source revision {revision} differs from {config['git_commit']}"
        )
    user_model = config.get("user_model", config["model"])
    user_url = config.get("user_base_url", config["base_url"])
    config.update(user_model=user_model, user_base_url=user_url, native_tool_calls=True)
    write_json(run / "config.json", config)
    completion = llm_utils.completion
    lock = Lock()

    def measured_completion(**kwargs):
        role = kwargs.pop("toolcall_role")
        key = "TOOLCALL_AGENT_API_KEY" if role == "agent" else "TOOLCALL_USER_API_KEY"
        kwargs["api_key"] = os.getenv(key, "local")
        row = {"role": role}
        start = time.monotonic()
        try:
            response = completion(**kwargs)
            usage = response.usage
            row.update(
                finish_reason=response.choices[0].finish_reason,
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
            )
            return response
        except Exception as error:
            row.update(
                error=type(error).__name__ + ": " + str(error),
                status=getattr(error, "status_code", None),
            )
            raise
        finally:
            row["latency_seconds"] = time.monotonic() - start
            with (
                lock,
                (run / "model-requests.jsonl").open("a", encoding="utf-8") as output,
            ):
                output.write(json.dumps(row) + "\n")

    llm_utils.completion = measured_completion

    def llm_arguments(role, url, temperature, thinking):
        arguments = dict(
            api_base=url,
            temperature=temperature,
            max_completion_tokens=config["max_tokens"],
            timeout=config["timeout"],
            num_retries=0,
            toolcall_role=role,
        )
        if role == "agent" or url == config["base_url"]:
            arguments["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking}
            }
        return arguments

    groups = {}
    reasons = Counter()
    start = time.monotonic()
    for domain in config["domains"]:
        destination = run / f"{domain}.json"
        options = RunConfig(
            domain=domain,
            task_ids=config.get("task_ids"),
            num_tasks=config.get("limit"),
            num_trials=config["trials"],
            seed=config["seed"],
            max_steps=config["max_steps"],
            llm_agent="openai/" + config["model"],
            llm_user="openai/" + user_model,
            llm_args_agent=llm_arguments(
                "agent", config["base_url"], config["temperature"], config["thinking"]
            ),
            llm_args_user=llm_arguments(
                "user", user_url, config["user_temperature"], False
            ),
            max_concurrency=config["concurrency"],
            save_to=str(destination.with_suffix("")),
            log_level="ERROR",
        )
        results = run_domain(options)
        expected = len(results.tasks) * config["trials"]
        if not expected or len(results.simulations) != expected:
            raise RuntimeError(
                f"τ² {domain} expected {expected} simulations, got {len(results.simulations)}"
            )
        per_domain_reasons = Counter(
            row.termination_reason.value for row in results.simulations
        )
        reasons.update(per_domain_reasons)
        metrics = compute_metrics(results).as_dict()
        groups[domain] = {
            "task_success_rate": float(metrics["pass_hat_1"]),
            "average_reward": float(metrics["avg_reward"]),
            "tasks": len(results.tasks),
            "simulations": len(results.simulations),
            "mean_duration_seconds": mean(row.duration for row in results.simulations),
            **{
                key: float(value)
                for key, value in metrics.items()
                if key.startswith("pass_hat_")
            },
            **{
                f"termination_{key}": value for key, value in per_domain_reasons.items()
            },
        }
    elapsed = time.monotonic() - start
    requests = read_jsonl(run / "model-requests.jsonl")
    metrics = request_metrics(requests, elapsed, config["max_tokens"])
    agent = request_metrics(
        [row for row in requests if row["role"] == "agent"],
        elapsed,
        config["max_tokens"],
    )
    if not agent["successful_calls"]:
        raise RuntimeError("τ² produced no successful agent requests")
    metrics.update(
        simulation_seconds=elapsed,
        domain_mean_task_success_rate=mean(
            group["task_success_rate"] for group in groups.values()
        ),
        **{f"agent_{key}": value for key, value in agent.items()},
    )
    write_json(
        run / "summary.json",
        result(metrics, groups, "task_success_rate", termination_reasons=dict(reasons)),
    )


if __name__ == "__main__":
    run_tau2(Path(sys.argv[1]).resolve())
