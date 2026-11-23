"""Execute native calls against API-Bank's published backends and per-API checkers."""

from collections import Counter, defaultdict
import functools
import json
import random
import statistics
import time

from .api_bank import OFFLINE_APIS
from .common import read_jsonl, request_metrics, result, write_json


def bind_retriever(assets):
    from sentence_transformers import SentenceTransformer
    from apis import tool_search as low
    from lv3_apis import tool_search as high

    model_dir = assets / "api-bank-encoder"
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(
            "API-Bank retriever weights missing. Run toolcall-eval prepare api-bank."
        )
    encoder = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True)
    encoder.encode = functools.lru_cache(maxsize=4096)(encoder.encode)

    def cached_encoder(_model_name):
        return encoder

    low.SentenceTransformer = cached_encoder
    high.SentenceTransformer = cached_encoder


def normalize_offline(name, params):
    if name == "Dictionary":
        return {
            k: (
                " ".join(v.replace("_", " ").casefold().split())
                if k == "keyword" and isinstance(v, str)
                else v
            )
            for k, v in params.items()
        }
    locale = {"spanish": "es", "french": "fr", "english": "en"}
    out = dict(params)
    for k in ("src_lang", "tgt_lang"):
        if k in out and isinstance(out[k], str):
            val = out[k].strip().casefold()
            out[k] = locale.get(val, val)
    if out.get("src_lang") in ("auto", None):
        out.pop("src_lang", None)
    return out


def call_score(case, name, params, manager):
    gold = case["gold"]
    expected = gold["api_name"]
    if name != expected:
        return False, "wrong_tool"
    if name in OFFLINE_APIS:
        # The dataset records I/O for these live services. Replaying the published
        # input avoids provider drift and keeps model-generated text on this host.
        is_match = normalize_offline(name, params) == normalize_offline(
            name, gold["input"]
        )
        return is_match, "fixture_replay" if is_match else "fixture_argument_mismatch"
    if (
        name == "ToolSearcher"
        and case["level"] == "L3"
        and isinstance(params.get("keywords"), str)
    ):
        from lv3_evaluator import split_by_uppercase

        params = {**params, "keywords": split_by_uppercase(params["keywords"])}
    # ToolManager's original parser expects the literal string 'True' or 'False'.
    params = {
        key: str(value) if isinstance(value, bool) else value
        for key, value in params.items()
    }
    try:
        result = manager.api_call(name, **params)
        tool = manager.init_tool(name)
        correct = bool(tool.check_api_call_correctness(result, gold["result"]))
        if correct:
            return True, "execution_match"
        if result.get("exception"):
            if "punkt_tab" in str(result["exception"]):
                return False, "backend_resource_missing"
            if name == "ToolSearcher" and result.get("output") is None:
                return False, "retriever_exception"
            return False, "tool_exception"
        return False, "execution_mismatch"
    except Exception as error:
        return False, f"execution_error:{type(error).__name__}"


def model_call(row, permitted):
    calls = row["message"].get("tool_calls") or []
    if not calls:
        return None, None, "missing_tool_call"
    if len(calls) != 1:
        return None, None, "unexpected_multiple_calls"
    try:
        call = calls[0]["function"]
        name = call["name"]
        if name not in permitted:
            return None, None, "tool_not_offered"
        params = json.loads(call["arguments"])
        if not isinstance(params, dict):
            return None, None, "arguments_not_object"
        return name, params, None
    except (ValueError, TypeError, KeyError):
        return None, None, "invalid_arguments"


def score(run, config, cases, elapsed, tool_manager, assets):
    started = time.monotonic()
    prediction = {row["id"]: row for row in read_jsonl(run / "predictions.jsonl")}
    if set(prediction) != {case["id"] for case in cases} or len(prediction) != len(
        cases
    ):
        raise RuntimeError("API-Bank prediction IDs differ from the selected cases")
    bind_retriever(assets)
    from rouge import Rouge

    rouge = Rouge()
    # Preserve the reference evaluator's shared Level-3 state within each track.
    model_l3 = tool_manager("./lv3_apis")
    oracle_l3 = tool_manager("./lv3_apis")
    results = []
    oracle_random = random.Random(42).getstate()
    model_random = random.Random(42).getstate()
    for case in cases:
        row = prediction[case["id"]]
        record = {
            key: case[key]
            for key in ("id", "level", "track", "dialogue", "step", "gold_in_tools")
        }
        if case["track"] == "response":
            reply = (row.get("message", {}).get("content") or "").strip()
            if case["level"] == "L3":
                reply = reply.replace("User:", "").replace("AI:", "").strip()
            try:
                value = (
                    rouge.get_scores(reply, case["gold"])[0]["rouge-l"]["f"]
                    if reply
                    else 0
                )
            except ValueError:
                value = 0
            record.update(
                rouge_l_f1=round(value, 4) if case["level"] != "L3" else value
            )
        else:
            gold = case["gold"]
            random.setstate(oracle_random)
            oracle, oracle_reason = call_score(
                case,
                gold["api_name"],
                gold["input"],
                oracle_l3 if case["level"] == "L3" else tool_manager(),
            )
            oracle_random = random.getstate()
            if "error" in row:
                correct, reason = False, "inference_error"
            else:
                offered = {item["function"]["name"] for item in case["tools"]}
                name, params, reason = model_call(row, offered)
                if reason:
                    correct = False
                else:
                    random.setstate(model_random)
                    correct, reason = call_score(
                        case,
                        name,
                        params,
                        model_l3 if case["level"] == "L3" else tool_manager(),
                    )
                    model_random = random.getstate()
            record.update(
                correct=correct,
                reason=reason,
                oracle_correct=oracle,
                oracle_reason=oracle_reason,
                gold_api=gold["api_name"],
            )
        results.append(record)
    with (run / "scores.jsonl").open("w") as out:
        for record in results:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    call_rows = [row for row in results if row["track"] == "call"]
    response_rows = [row for row in results if row["track"] == "response"]
    requests = [
        (
            {"error": row["error"]}
            if "error" in row
            else {
                "finish_reason": row["finish_reason"],
                "latency_seconds": row["latency_seconds"],
                "input_tokens": row["usage"]["prompt_tokens"],
                "output_tokens": row["usage"]["completion_tokens"],
            }
        )
        for row in prediction.values()
    ]
    metrics = request_metrics(requests, elapsed, config["max_tokens"])
    if not metrics["successful_calls"]:
        raise RuntimeError("API-Bank produced no successful model requests")
    metrics.update(
        samples=len(results),
        call_samples=len(call_rows),
        response_samples=len(response_rows),
        correct_calls=sum(row["correct"] for row in call_rows),
        call_microaccuracy=statistics.mean(row["correct"] for row in call_rows),
        mean_rouge_l_f1=statistics.mean(row["rouge_l_f1"] for row in response_rows),
        generation_seconds=elapsed,
        samples_per_second=len(results) / elapsed,
    )
    groups = {}
    for level in ("L1", "L2", "L3"):
        calls = [row for row in call_rows if row["level"] == level]
        replies = [row for row in response_rows if row["level"] == level]
        available = [row for row in calls if row["gold_in_tools"]]
        groups[level] = {
            "calls": len(calls),
            "correct": sum(row["correct"] for row in calls),
            "accuracy": statistics.mean(row["correct"] for row in calls),
            "reference_tool_coverage": len(available) / len(calls),
            "calls_with_reference_tool_offered": len(available),
            "accuracy_when_reference_tool_offered": statistics.mean(
                row["correct"] for row in available
            )
            if available
            else None,
            "oracle_correct": sum(row["oracle_correct"] for row in calls),
            "oracle_accuracy": statistics.mean(row["oracle_correct"] for row in calls),
            "gold_tool_not_offered": sum(not row["gold_in_tools"] for row in calls),
            "offline_fixture_cases": sum(
                row["gold_api"] in OFFLINE_APIS for row in calls
            ),
            "call_result_reasons": dict(Counter(row["reason"] for row in calls)),
            "oracle_result_reasons": dict(
                Counter(row["oracle_reason"] for row in calls)
            ),
            "response_samples": len(replies),
            "rouge_l_f1": statistics.mean(row["rouge_l_f1"] for row in replies),
        }
    if config["profile"] == "full":
        dialogues = defaultdict(list)
        for row in call_rows:
            if row["level"] == "L3":
                dialogues[row["dialogue"]].append(row["correct"])
        assert len(dialogues) == 50
        metrics["L3_dialogues_all_steps_correct"] = sum(
            all(rows) for rows in dialogues.values()
        )
    metrics["scoring_seconds"] = time.monotonic() - started
    write_json(run / "summary.json", result(metrics, groups, "accuracy"))
