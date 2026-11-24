"""Export offline model-boundary requests, labels and saved outputs for pinned benchmarks."""

from collections import Counter
from copy import deepcopy
from hashlib import file_digest
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/eval-inspection"
RUNS = ROOT / "runs"
MODEL = "Qwen/Qwen3.5-2B"
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def rel(path):
    return str(path.relative_to(ROOT))


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def export(suite, subset, case, request, label, prediction=None, **context):
    path = OUT / suite / f"{subset}.json"
    write(
        path,
        dict(
            suite=suite,
            subset=subset,
            request=request,
            label=label,
            prediction=prediction,
            **context,
            case=case,
        ),
    )
    return rel(path)


def bfcl():
    from importlib.metadata import version
    from bfcl_eval.constants.category_mapping import (
        ALL_CATEGORIES,
        MEMORY_CATEGORY,
        MULTI_TURN_CATEGORY,
        NON_LIVE_CATEGORY,
        LIVE_CATEGORY,
        WEB_SEARCH_CATEGORY,
    )
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
    from bfcl_eval import utils

    assert version("bfcl-eval") == "2026.3.23"

    class Captured(Exception):
        pass

    class OfflineHandler(OpenAICompletionsHandler):
        def generate_with_backoff(self, **kwargs):
            raise Captured(
                {
                    **kwargs,
                    "max_completion_tokens": 1024,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )

    from unittest.mock import patch

    with patch("bfcl_eval.model_handler.api_inference.openai_completion.OpenAI"):
        handler = OfflineHandler(MODEL, 0, MODEL, True, underscore_to_dot=True)
        format_handler = OfflineHandler(MODEL, 0, MODEL, False, underscore_to_dot=True)

    def capture(entry, messages=None):
        if entry["id"].startswith("format_sensitivity"):
            handler_ = format_handler
            data = handler_._pre_query_processing_prompting(entry)
            data = handler_.add_first_turn_message_prompting(data, entry["question"][0])
            query = handler_._query_prompting
        else:
            handler_ = handler
            data = handler_._pre_query_processing_FC({}, entry)
            data = handler_._compile_tools(data, entry)
            data = handler_.add_first_turn_message_FC(
                data, messages if messages is not None else entry["question"][0]
            )
            query = handler_._query_FC
        try:
            query(data)
        except Captured as found:
            return found.args[0]
        raise AssertionError("No request captured")

    def history(category, ids):
        candidates = sorted(
            (RUNS / "eval/bfcl-v4").rglob(f"BFCL_v4_{category}_result.json"),
            reverse=True,
        )
        for folder in (
            "bfcl-v4-qwen35-2b-no-web-1024",
            "bfcl-v4-qwen35-2b-five-categories-ctx10240",
        ):
            candidates += sorted(
                (RUNS / "archive/bfcl-v4" / folder / "result").rglob(
                    f"BFCL_v4_{category}_result.json"
                )
            )
        found = {}
        for path in candidates:
            for line in rows(path):
                if line["id"] not in ids:
                    continue
                if line["id"] not in found or (
                    not found[line["id"]][0].get("inference_log")
                    and line.get("inference_log")
                ):
                    found[line["id"]] = (line, path)
        return found

    groups = {
        **dict.fromkeys(NON_LIVE_CATEGORY, "Non-Live"),
        **dict.fromkeys(LIVE_CATEGORY, "Live"),
        **dict.fromkeys(MULTI_TURN_CATEGORY, "Multi-turn"),
        **dict.fromkeys(MEMORY_CATEGORY, "Memory"),
        **dict.fromkeys(WEB_SEARCH_CATEGORY, "Web Search"),
        "format_sensitivity": "非计分诊断",
    }
    relevance = {
        "irrelevance": False,
        "live_irrelevance": False,
        "live_relevance": True,
    }
    index = []
    for category in ALL_CATEGORIES:
        entries = load_dataset_entry(category, include_prereq=False)
        predictions = history(category, {r["id"] for r in entries})

        def has_turn_log(identifier):
            return any(
                isinstance(i, dict) and i.get("begin_of_turn_query")
                for i in predictions.get(identifier, ({}, None))[0].get(
                    "inference_log", []
                )
            )

        entry = (
            next((r for r in entries if has_turn_log(r["id"])), entries[0])
            if category in MEMORY_CATEGORY + MULTI_TURN_CATEGORY
            else entries[0]
        )
        identifier = entry["id"]
        prediction, source = predictions.get(identifier, (None, None))
        ground = None
        if category in relevance:
            ground = {
                "expected_tool_call_present": relevance[category],
                "source": "derived from upstream relevance_file_runner, not a reference reply",
            }
        else:
            canonical = identifier
            if category in MEMORY_CATEGORY:
                canonical = canonical.replace(category, "memory", 1)
            if category in WEB_SEARCH_CATEGORY:
                canonical = canonical.replace(category, "web_search", 1)
            all_labels = load_ground_truth_entry(category)
            if category == "format_sensitivity":
                ground = all_labels[entries.index(entry)]
            else:
                ground = next((g for g in all_labels if g["id"] == canonical), None)
                assert ground is not None, (identifier, canonical)
        log = (prediction or {}).get("inference_log") or []
        turn_log = [
            value
            for value in log
            if isinstance(value, dict) and value.get("begin_of_turn_query")
        ]
        first_messages = (
            turn_log[0]["begin_of_turn_query"]
            if category in MEMORY_CATEGORY and turn_log
            else None
        )
        assert category not in MEMORY_CATEGORY or first_messages, (
            f"Missing archived Memory context: {identifier}"
        )
        request = capture(deepcopy(entry), first_messages)
        path = export(
            "bfcl-v4",
            category,
            entry,
            request,
            ground,
            prediction,
            id=identifier,
            prediction_source=rel(source) if source else None,
            request_status="上游 prompting 格式诊断示例，非本项目 Native FC"
            if category == "format_sensitivity"
            else "完整首请求，离线重建",
            scenario_notes="后续由历史输出和环境状态决定；预测中保留已有 inference_log。",
        )
        index.append(
            dict(
                category=category,
                group=groups[category],
                rows=len(entries),
                id=identifier,
                path=path,
                request_messages=len(request["messages"]),
                tools=len(request.get("tools", [])),
                prediction=bool(prediction),
                core=groups[category] in ("Non-Live", "Live", "Multi-turn"),
                fields=sorted(entry),
                label_fields=list(ground),
            )
        )
        print("bfcl-v4", category, len(entries), identifier, flush=True)
    assert sum(r["rows"] for r in index if r["core"]) == 4441
    paths = {
        rel(p): sha(p)
        for p in utils.PROMPT_PATH.rglob("*")
        if p.is_file() and "unused_datasets" not in p.parts
    }
    write(
        OUT / "bfcl-v4/index.json",
        dict(version=version("bfcl-eval"), subsets=index, source_files=paths),
    )


def api_bank():
    from toolcall.evaluation.api_bank import init_backend, make_manifest, inference

    upstream = RUNS / "upstream/api-bank/api-bank"
    raw = RUNS / "upstream/api-bank-data"
    pin = "f30ccf22b4e2617fab32958d4c03f5c1f2e7dfcf"
    assert (
        subprocess.check_output(
            ["git", "-C", upstream, "rev-parse", "HEAD"], text=True
        ).strip()
        == pin
    )
    run = RUNS / "eval/api-bank/qwen3.5-2b/20261123-104525-867552-api-bank-full"
    config = read(run / "config.json")
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        cases = make_manifest(init_backend(Path(temporary), upstream), upstream, raw)
    actual = {row["id"]: row for row in rows(run / "predictions.jsonl")}
    scores = {row["id"]: row for row in rows(run / "scores.jsonl")}
    counts = Counter((row["level"], row["track"]) for row in cases)
    index = []

    class Capture(Exception):
        pass

    class OfflineClient:
        def post(self, endpoint, *, json):
            assert endpoint == "chat/completions"
            raise Capture(json)

    for (level, track), count in sorted(counts.items()):
        candidates = [c for c in cases if c["level"] == level and c["track"] == track]
        case = next(
            (
                c
                for c in candidates
                if c["tools"]
                and scores.get(c["id"], {}).get("correct")
                and c["gold_in_tools"] is not False
            ),
            candidates[0],
        )
        if track == "call" and level in {"L2", "L3"}:
            case = next(
                (
                    c
                    for c in candidates
                    if c["gold_in_tools"]
                    and c["gold"]["api_name"] != "ToolSearcher"
                    and c["id"] in actual
                ),
                case,
            )
        try:
            inference(case, OfflineClient(), config)
        except Capture as captured:
            request = captured.args[0]
        else:
            raise AssertionError("No captured API-Bank payload")
        assert "gold" not in request
        subset = f"{level.lower()}-{track}"
        path = export(
            "api-bank",
            subset,
            case,
            request,
            case["gold"],
            actual.get(case["id"]),
            id=case["id"],
            request_status="从原生 FC 适配器和参考历史离线重建",
            prediction_source=rel(run / "predictions.jsonl"),
            score=scores.get(case["id"]),
        )
        index.append(
            dict(
                category=subset,
                group=f"{level}",
                rows=count,
                id=case["id"],
                path=path,
                fields=sorted(case),
                tools=len(request.get("tools", [])),
            )
        )
    assert len(cases) == 1281
    write(
        OUT / "api-bank/index.json",
        dict(
            version=pin,
            level3=config["official_sources"]["level3_dataset"],
            source_files={rel(p): sha(p) for p in raw.glob("*.json")},
            subsets=index,
        ),
    )


def tau2():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator import UserSimulator
    from tau2.utils.llm_utils import to_litellm_messages
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

    upstream = RUNS / "upstream/tau2-sparse"
    pin = "f8de30c298689cbe0117d76a378e7315a17e5bd8"
    assert (
        subprocess.check_output(
            ["git", "-C", upstream, "rev-parse", "HEAD"], text=True
        ).strip()
        == pin
    )
    smoke = RUNS / "eval/tau2/qwen3.5-2b/20261123-104431-366707-tau2-smoke/retail.json"
    observed = read(smoke)["simulations"][0]
    greeting = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
    index = []
    demonstrations = {
        "airline": "Hi, I need help with my flight reservation.",
        "telecom": "Hi, my mobile data is not working. Can you help?",
    }

    def payload(messages, tools, model):
        return dict(
            model=model,
            messages=to_litellm_messages(messages),
            tools=[t.openai_schema for t in tools] if tools else None,
            tool_choice="auto" if tools else None,
        )

    for domain in ("retail", "airline", "telecom"):
        module = importlib.import_module(f"tau2.domains.{domain}.environment")
        tasks = module.get_tasks()
        task = tasks[0]
        env = module.get_environment()
        tools = env.get_tools()
        try:
            user_tools = env.get_user_tools()
        except ValueError:
            user_tools = None
        simulator = UserSimulator(
            tools=user_tools, instructions=str(task.user_scenario)
        )
        simulator_state = simulator.get_init_state()
        simulator_state.messages.append(greeting)
        user_request = payload(
            simulator_state.system_messages + simulator_state.flip_roles(),
            user_tools,
            MODEL if domain == "retail" else "<正式评测待配置的独立模拟器>",
        )
        agent = LLMAgent(tools=tools, domain_policy=env.get_policy())
        agent_state = agent.get_init_state(message_history=[greeting])
        if domain == "retail":
            assert str(task.id) == str(observed["task_id"])
            first_user = UserMessage.model_validate(observed["messages"][1])
            prediction = observed
        else:
            first_user = UserMessage(role="user", content=demonstrations[domain])
            prediction = None
        request = payload(
            agent_state.system_messages + agent_state.messages + [first_user],
            tools,
            MODEL,
        )
        request.update(
            temperature=0,
            max_completion_tokens=1024,
            chat_template_kwargs={"enable_thinking": False},
        )
        record = task.model_dump(mode="json")
        path = export(
            "tau2-bench",
            domain,
            record,
            request,
            record["evaluation_criteria"],
            prediction,
            id=task.id,
            simulator_first_request=user_request,
            request_status="重建自 2B Retail smoke 的开场和用户实录"
            if prediction
            else "完整首请求演示；用户开场为人工示例，不是模拟器/模型实录",
            prediction_source=rel(smoke) if prediction else None,
        )
        index.append(
            dict(
                category=domain,
                group="域",
                rows=len(tasks),
                id=task.id,
                path=path,
                fields=sorted(record),
                tools=len(tools),
            )
        )
    assert [r["rows"] for r in index] == [114, 50, 114]
    files = upstream / "data/tau2/domains"
    write(
        OUT / "tau2-bench/index.json",
        dict(
            version=f"v0.2.0 / {pin}",
            subsets=index,
            source_files={
                rel(p): sha(p) for p in sorted(files.rglob("*")) if p.is_file()
            },
        ),
    )


if __name__ == "__main__":
    {"bfcl": bfcl, "api-bank": api_bank, "tau2": tau2}[sys.argv[1]]()
