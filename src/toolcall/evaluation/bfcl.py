"""Register Qwen native FC and invoke the official BFCL V4 CLI."""

from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, local
import time

from .common import read_json, read_jsonl, request_metrics, result, write_json


def register(config, run):
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    class LocalFC(OpenAICompletionsHandler):
        request = local()
        lock = Lock()

        def _build_client_kwargs(self):
            kwargs = super()._build_client_kwargs()
            return dict(kwargs, max_retries=0, timeout=config["timeout"])

        def inference(self, test_entry, include_input_log, exclude_state_log):
            self.request.id = test_entry["id"]
            self.request.sequence = 0
            return super().inference(test_entry, include_input_log, exclude_state_log)

        def generate_with_backoff(self, **kwargs):
            kwargs.update(
                max_completion_tokens=config["max_tokens"],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": config["thinking"]}
                },
            )
            self.request.sequence += 1
            row = {"id": self.request.id, "step": self.request.sequence}
            start = time.monotonic()
            try:
                response, latency = super().generate_with_backoff(**kwargs)
                row.update(
                    finish_reason=response.choices[0].finish_reason,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )
                return response, latency
            except Exception as error:
                row.update(error=str(error), status=getattr(error, "status_code", None))
                raise
            finally:
                row["latency_seconds"] = time.monotonic() - start
                with (
                    self.lock,
                    (run / "model-requests.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream,
                ):
                    stream.write(json.dumps(row) + "\n")

    name = config["model"]
    MODEL_CONFIG_MAPPING[name] = ModelConfig(
        model_name=name,
        display_name=name,
        url="https://github.com/QwenLM/Qwen3.5",
        org="Qwen",
        license="Apache 2.0",
        model_handler=LocalFC,
        is_fc_model=True,
        underscore_to_dot=True,
    )


def selection(config):
    from bfcl_eval.constants.category_mapping import (
        ALL_SCORING_CATEGORIES,
        MEMORY_CATEGORY,
        WEB_SEARCH_CATEGORY,
    )
    from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument

    if "fixed_case_ids" in config:
        chosen = config["fixed_case_ids"]
        if not isinstance(chosen, dict) or not chosen:
            raise ValueError(
                "BFCL ID file must map categories to nonempty lists of case IDs"
            )
        for key, ids in chosen.items():
            if (
                key not in ALL_SCORING_CATEGORIES
                or not isinstance(ids, list)
                or not ids
                or not all(isinstance(i, str) for i in ids)
            ):
                raise ValueError(f"Invalid BFCL ID selection for {key}")
            available = {
                row["id"] for row in load_dataset_entry(key, include_prereq=False)
            }
            if not set(ids) <= available or len(ids) != len(set(ids)):
                raise ValueError(f"Unknown or repeated case IDs for {key}")
        return list(chosen), chosen
    if config.get("categories"):
        names = [name for item in config["categories"] for name in item.split(",")]
        categories = parse_test_category_argument(names)
        if not set(categories) <= set(ALL_SCORING_CATEGORIES):
            raise ValueError(
                "Native FC uses BFCL scoring categories. Use all_scoring for the complete suite."
            )
        return categories, None
    if config["profile"] == "smoke":
        return [config["test_category"]], {config["test_category"]: config["case_ids"]}
    excluded = set()
    if config["profile"] != "full":
        excluded |= set(WEB_SEARCH_CATEGORY)
    if config["profile"] == "core":
        excluded |= set(MEMORY_CATEGORY)
    return [key for key in ALL_SCORING_CATEGORIES if key not in excluded], None


def include_dependencies(ids):
    from bfcl_eval.constants.category_mapping import MEMORY_CATEGORY
    from bfcl_eval.utils import load_dataset_entry

    generation_ids = dict(ids)
    for category in set(ids) & set(MEMORY_CATEGORY):
        entries = {row["id"]: row for row in load_dataset_entry(category)}
        prerequisites = dict.fromkeys(
            dependency
            for identifier in ids[category]
            for dependency in entries[identifier].get("depends_on", [])
        )
        generation_ids[category] = [*prerequisites, *ids[category]]
    return generation_ids


def score_summary(run, categories, seconds, score_seconds, maximum, *, partial):
    groups = {}
    for path in (run / "score").rglob("BFCL_v4_*_score.json"):
        category = path.name.removeprefix("BFCL_v4_").removesuffix("_score.json")
        with path.open() as stream:
            groups[category] = json.loads(stream.readline())
    if set(groups) != set(categories):
        raise RuntimeError(
            f"Expected BFCL categories {categories}; received {sorted(groups)}"
        )
    totals = sum(group["total_count"] for group in groups.values())
    correct = sum(group["correct_count"] for group in groups.values())
    requests = read_jsonl(run / "model-requests.jsonl")
    metrics = request_metrics(requests, seconds, maximum)
    if not metrics["successful_calls"]:
        raise RuntimeError("BFCL produced no successful model requests")
    metrics.update(
        cases=totals,
        correct=correct,
        microaccuracy=correct / totals,
        generation_seconds=seconds,
        scoring_seconds=score_seconds,
        cases_per_second=totals / seconds,
    )
    import csv
    from bfcl_eval.constants.category_mapping import (
        ALL_SCORING_CATEGORIES,
        MEMORY_CATEGORY,
        WEB_SEARCH_CATEGORY,
    )

    with (run / "score/data_overall.csv").open() as stream:
        official_row = next(csv.DictReader(stream))
    if not partial and set(categories) == set(ALL_SCORING_CATEGORIES):
        metrics["official_overall_accuracy"] = (
            float(official_row["Overall Acc"].rstrip("%")) / 100
        )
    if not partial and set(categories) == set(ALL_SCORING_CATEGORIES) - set(
        WEB_SEARCH_CATEGORY
    ):
        metrics["no_web_weighted_accuracy"] = (
            float(official_row["Overall Acc"].rstrip("%")) / 80
        )
    if not partial and set(categories) == set(ALL_SCORING_CATEGORIES) - set(
        WEB_SEARCH_CATEGORY
    ) - set(MEMORY_CATEGORY):
        metrics["core_weighted_accuracy"] = (
            float(official_row["Overall Acc"].rstrip("%")) / 60
        )
    return result(
        metrics,
        {key: groups[key] for key in categories},
        "accuracy",
        official_table=official_row,
    )


def run_bfcl(run):
    from bfcl_eval.constants.category_mapping import WEB_SEARCH_CATEGORY

    config = read_json(run / "config.json")
    installed = version("bfcl-eval")
    if installed != config["version"]:
        raise RuntimeError(
            f"BFCL installed={installed} but config expects {config['version']}"
        )
    categories, ids = selection(config)
    if set(categories) & set(WEB_SEARCH_CATEGORY) and not os.getenv("SERPAPI_API_KEY"):
        raise RuntimeError(
            "Web Search requires SERPAPI_API_KEY. Use --profile no-web for the other V4 categories."
        )
    config.update(
        resolved_categories=categories,
        fixed_case_ids=ids,
        parser="qwen3_coder",
        native_tool_calls=True,
        underscore_to_dot=True,
    )
    write_json(run / "config.json", config)
    common = ["--model", config["model"]]
    category_arg = ["--test-category", ",".join(categories)]
    generate_scope = category_arg
    evaluate_scope = category_arg
    if ids:
        write_json(run / "test_case_ids_to_generate.json", include_dependencies(ids))
        generate_scope = ["--run-ids"]
        evaluate_scope = [*category_arg, "--partial-eval"]
    commands = (
        [
            "generate",
            *common,
            *generate_scope,
            "--num-threads",
            str(config["concurrency"]),
            "--temperature",
            str(config["temperature"]),
            "--exclude-state-log",
        ],
        ["evaluate", *common, *evaluate_scope],
    )
    env = dict(
        os.environ,
        BFCL_PROJECT_ROOT=str(run),
        OPENAI_BASE_URL=config["base_url"],
        OPENAI_API_KEY=os.getenv("TOOLCALL_AGENT_API_KEY", "local"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    durations = []
    for command in commands:
        start = time.monotonic()
        subprocess.run(
            [sys.executable, "-m", __spec__.name, str(run), *command],
            env=env,
            check=True,
        )
        durations.append(time.monotonic() - start)
    summary = score_summary(
        run, categories, *durations, config["max_tokens"], partial=bool(ids)
    )
    write_json(run / "summary.json", summary)


def main():
    run = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        from bfcl_eval.__main__ import cli

        register(read_json(run / "config.json"), run)
        cli(args=sys.argv[2:], prog_name="bfcl")
    else:
        run_bfcl(run)


if __name__ == "__main__":
    main()
