"""Run the API-Bank release with Qwen native tool calls and upstream API checks."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import ModuleType

import httpx

from .common import read_json, read_jsonl, write_json


OFFLINE_APIS = frozenset({"Translate", "Dictionary"})
CALL_PROMPT = (
    "Based on the conversation and available API descriptions, generate exactly the next "
    "API request using a tool call. Use only the available tools. Call ToolSearcher to "
    "find a tool if retrieval is needed. This year is 2023."
)
RESPONSE_PROMPT = (
    "Based on the conversation and API results, generate the next natural language "
    "assistant response. This year is 2023."
)


def init_backend(run, upstream):
    # Import the untouched upstream APIs, with databases isolated from the checkout.
    import shutil

    sandbox = run / "api-workdir"
    sandbox.mkdir(exist_ok=True)
    for name in ("apis", "lv3_apis"):
        link = sandbox / name
        if not link.exists():
            link.symlink_to(upstream / name, target_is_directory=True)
    if not (sandbox / "init_database").exists():
        shutil.copytree(upstream / "init_database", sandbox / "init_database")
    os.chdir(sandbox)
    sys.path.insert(0, str(upstream))

    # Translate is replayed with the published fixture and never initializes a web client.
    googletrans = ModuleType("googletrans")

    class OfflineTranslator:
        def __init__(self):
            raise RuntimeError(
                "Live Translate is disabled; use the recorded evaluation fixture"
            )

    googletrans.Translator = OfflineTranslator
    sys.modules["googletrans"] = googletrans
    import nltk

    nltk_root = Path(
        os.environ.get(
            "APIBANK_NLTK_DATA", Path.home() / ".cache/toolcall-api-bank-nltk"
        )
    )
    nltk.data.path.insert(0, str(nltk_root))
    from tool_manager import ToolManager

    return ToolManager


def tool_catalog(manager):
    return {
        name: json.loads(manager.get_api_description(name))
        for name in manager.list_all_apis()
    }


def schema(info):
    kind = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "list(str)": "array",
    }
    fields = {}
    required = []
    for name, prop in info["input_parameters"].items():
        details = {"description": prop["description"], "type": kind[prop["type"]]}
        if prop["type"] == "list(str)":
            details["items"] = {"type": "string"}
        elif prop["type"] == "list":
            details["items"] = {}
        fields[name] = details
        if "optional" not in str(prop.get("description", "")).lower() and not prop.get(
            "optional"
        ):
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": info["name"],
            "description": info["description"],
            "parameters": {
                "type": "object",
                "properties": fields,
                "required": required,
            },
        },
    }


def encode_history(history):
    messages = []
    discovered = set()
    for event in history:
        if event["role"] in ("User", "AI"):
            messages.append(
                {
                    "role": "user" if event["role"] == "User" else "assistant",
                    "content": event["text"],
                }
            )
            continue
        name, arguments, result = (
            event["api_name"],
            event["param_dict"],
            event["result"],
        )
        call_id = f"call_apibank_{len(messages)}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
        if name == "ToolSearcher":
            output = result["output"]
            for tool in output if isinstance(output, list) else [output]:
                if isinstance(tool, dict) and tool.get("name"):
                    discovered.add(tool["name"])
    return messages, discovered


def make_case(level, track, dialogue, step, history, catalog, fixed_tools, gold):
    messages, discovered = encode_history(history)
    chosen = sorted((fixed_tools | discovered) & catalog.keys())
    return {
        "id": f"{level}:{dialogue}:{track}:{step}",
        "level": level,
        "track": track,
        "dialogue": dialogue,
        "step": step,
        "messages": [
            {
                "role": "system",
                "content": CALL_PROMPT if track == "call" else RESPONSE_PROMPT,
            },
            *messages,
        ],
        "tools": [schema(catalog[name]) for name in chosen],
        "gold": gold,
        "gold_in_tools": gold["api_name"] in chosen if track == "call" else None,
    }


def make_manifest(tool_manager, upstream, data):
    l12 = tool_catalog(tool_manager())
    l3 = tool_catalog(tool_manager("./lv3_apis"))
    all_cases = []
    dirs = (("L1", "level-1-given-desc"), ("L2", "level-2-toolsearcher"))
    for level, directory in dirs:
        for file in sorted((upstream / "lv1-lv2-samples" / directory).glob("*.jsonl")):
            rows = read_jsonl(file)
            offered = (
                {event["api_name"] for event in rows if event["role"] == "API"}
                if level == "L1"
                else {"ToolSearcher"}
            )
            for index, event in enumerate(rows):
                if event["role"] != "API":
                    continue
                gold = {
                    "api_name": event["api_name"],
                    "input": event["param_dict"],
                    "result": event["result"],
                }
                all_cases.append(
                    make_case(
                        level,
                        "call",
                        file.stem,
                        index,
                        rows[:index],
                        l12,
                        offered,
                        gold,
                    )
                )
                if index + 1 < len(rows) and rows[index + 1]["role"] == "AI":
                    all_cases.append(
                        make_case(
                            level,
                            "response",
                            file.stem,
                            index,
                            rows[: index + 1],
                            l12,
                            offered,
                            rows[index + 1]["text"],
                        )
                    )
    histories = read_json(data / "level-3.json")
    history_by_id = {}
    for i, row in enumerate(histories):
        history_by_id[i] = [{"role": "User", "text": row["requirement"].strip()}] + [
            {
                "role": "API",
                "api_name": call["api_name"],
                "param_dict": call["input"],
                "result": call["output"],
            }
            for call in row["apis"]
        ]
    for row in read_json(data / "level-3-batch-inf.json"):
        identifier, index = row["sample_id"], row["api_id"]
        history = history_by_id[identifier]
        event = history[index + 1]
        gold = {
            "api_name": event["api_name"],
            "input": event["param_dict"],
            "result": event["result"],
        }
        all_cases.append(
            make_case(
                "L3",
                "call",
                str(identifier),
                index,
                history[: index + 1],
                l3,
                {"ToolSearcher"},
                gold,
            )
        )
    for row in read_json(data / "level-3-batch-inf-response.json"):
        identifier = row["sample_id"]
        all_cases.append(
            make_case(
                "L3",
                "response",
                str(identifier),
                row["api_id"],
                history_by_id[identifier],
                l3,
                {"ToolSearcher"},
                row["output"],
            )
        )
    counts = Counter((c["level"], c["track"]) for c in all_cases)
    assert counts == {
        ("L1", "call"): 389,
        ("L2", "call"): 119,
        ("L3", "call"): 245,
        ("L1", "response"): 375,
        ("L2", "response"): 103,
        ("L3", "response"): 50,
    }, counts
    assert len({c["id"] for c in all_cases}) == len(all_cases)
    return all_cases


def select_cases(cases, scope, seed):
    if scope == "full":
        return cases
    groups = defaultdict(list)
    for case in cases:
        groups[(case["level"], case["track"])].append(case)
    targets = {"L1": (20, 14), "L2": (20, 13), "L3": (20, 13)}
    rng = random.Random(seed)
    selected = set()
    for level, (calls, responses) in targets.items():
        for track, count in (("call", calls), ("response", responses)):
            selected.update(
                case["id"] for case in rng.sample(groups[level, track], count)
            )
    result = [case for case in cases if case["id"] in selected]
    assert len(result) == 100
    return result


def inference(case, client, config):
    payload = {
        "model": config["model"],
        "messages": case["messages"],
        "temperature": config["temperature"],
        "max_completion_tokens": config["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": config["thinking"]},
    }
    if case["tools"]:
        payload.update(
            tools=case["tools"],
            tool_choice="auto" if case["track"] == "call" else "none",
        )
    started = time.perf_counter()
    record = {"id": case["id"]}
    try:
        response = client.post("chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        record.update(
            message=choice["message"],
            finish_reason=choice["finish_reason"],
            usage=body["usage"],
        )
    except httpx.HTTPStatusError as error:
        record.update(
            error=error.response.text[:850], status=error.response.status_code
        )
    except httpx.RequestError as error:
        record.update(error=f"{type(error).__name__}: {error}")
    record["latency_seconds"] = time.perf_counter() - started
    return record


def run_api_bank(run):
    from .api_bank_score import score

    config = read_json(run / "config.json")
    assets = Path(config["workspace"]) / "runs/upstream"
    upstream = assets / "api-bank/api-bank"
    data = assets / "api-bank-data"
    if not upstream.is_dir() or not data.is_dir():
        raise FileNotFoundError(
            "API-Bank assets missing. Run toolcall-eval prepare api-bank."
        )
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config["official_sources"]["repository"]:
        raise RuntimeError(
            "API-Bank checkout differs from the pinned revision. Run toolcall-eval prepare api-bank."
        )
    config.update(
        mode="native_fc_gold_history",
        native_tool_calls=True,
        parser="qwen3_coder",
        offline_replay_apis=sorted(OFFLINE_APIS),
    )
    cases = select_cases(
        make_manifest(init_backend(run, upstream), upstream, data),
        config["profile"],
        config.get("seed", 42),
    )
    write_json(
        run / "cases.json",
        [
            {
                key: case[key]
                for key in ("id", "level", "track", "dialogue", "step", "gold_in_tools")
            }
            for case in cases
        ],
    )
    start = time.monotonic()
    concurrency = config["concurrency"]
    api_key = os.getenv("TOOLCALL_AGENT_API_KEY", "local")
    with httpx.Client(
        base_url=config["base_url"] + "/",
        timeout=config["timeout"],
        limits=httpx.Limits(
            max_connections=concurrency, max_keepalive_connections=concurrency
        ),
        headers={"Authorization": f"Bearer {api_key}"},
        trust_env=False,
    ) as client:
        models = client.get("models").raise_for_status().json()["data"]
        served = next((row for row in models if row["id"] == config["model"]), None)
        if served is None:
            raise ValueError(f"Endpoint did not advertise {config['model']}")
        if served.get("max_model_len"):
            config["context_window"] = served["max_model_len"]
        write_json(run / "config.json", config)
        with (
            (run / "predictions.jsonl").open("w", encoding="utf-8") as output,
            ThreadPoolExecutor(max_workers=concurrency) as pool,
        ):
            for index, row in enumerate(
                pool.map(
                    functools.partial(inference, client=client, config=config), cases
                ),
                1,
            ):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                if index % 25 == 0 or index == len(cases):
                    print(
                        f"API-Bank: {index}/{len(cases)} in {time.monotonic() - start:.1f}s",
                        flush=True,
                    )
    score(
        run,
        config,
        cases,
        time.monotonic() - start,
        init_backend(run, upstream),
        assets,
    )


if __name__ == "__main__":
    run_api_bank(Path(sys.argv[1]).resolve())
