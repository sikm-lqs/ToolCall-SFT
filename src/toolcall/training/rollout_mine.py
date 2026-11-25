"""Dual-GPU DP=2 async vLLM rollout & ToolRL+3.4-B reward difficulty mining."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

from .grpo_reward import score_completion
from .preflight import verify_data_manifest


def is_rollout_pass(
    scored: dict[str, float | bool], *, task_type: str
) -> bool:
    """Strict correctness criterion aligned with ToolRL + 3.4-B symmetric reward."""
    if float(scored["r_format"]) < 1.0 or bool(scored["has_leak"]):
        return False
    if task_type == "call":
        return float(scored["r_correct"]) >= 2.999
    return not bool(scored["has_call_intent"]) and float(scored["r_correct"]) >= 2.5


def score_prompt_rollouts(
    row: dict[str, Any], completions: list[str], completion_lens: list[int] | None = None
) -> dict[str, Any]:
    """Score N rollouts for one prompt and compute difficulty / variance statistics."""
    task_type = str(row["task_type"])
    bucket = str(row["bucket"])
    gt = row["ground_truth"]
    tools = row.get("tools")

    breakdowns = [
        score_completion(
            text, task_type=task_type, bucket=bucket, ground_truth=gt, tools=tools
        )
        for text in completions
    ]
    r_finals = [round(float(b["r_final"]), 6) for b in breakdowns]
    r_formats = [float(b["r_format"]) for b in breakdowns]
    r_corrects = [round(float(b["r_correct"]), 6) for b in breakdowns]
    pass_flags = [is_rollout_pass(b, task_type=task_type) for b in breakdowns]
    pass_count = sum(1 for flag in pass_flags if flag)
    n = max(1, len(completions))
    reward_mean = round(sum(r_finals) / n, 6)
    reward_std = round(statistics.stdev(r_finals), 6) if len(r_finals) >= 2 else 0.0
    lens = completion_lens if completion_lens is not None else [len(c) for c in completions]

    return {
        "sample_id": row["meta"]["sample_id"],
        "task_type": task_type,
        "bucket": bucket,
        "group_key": row["meta"]["group_key"],
        "source": row["meta"]["source"],
        "num_candidate_tools": row["meta"]["num_candidate_tools"],
        "num_prompt_tokens": row["meta"]["num_prompt_tokens"],
        "n_rollouts": len(completions),
        "pass_count": pass_count,
        "pass_rate": round(pass_count / n, 4),
        "pass_flags": pass_flags,
        "r_finals": r_finals,
        "r_formats": r_formats,
        "r_corrects": r_corrects,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "format_pass_count": sum(1 for f in r_formats if f >= 1.0),
        "leak_count": sum(1 for b in breakdowns if bool(b["has_leak"])),
        "completion_tokens": lens,
        "completions": completions,
    }


def summarize_scored_rollouts(
    records: list[dict[str, Any]], *, wall_seconds: float, n_rollouts: int
) -> dict[str, Any]:
    """Aggregate pass_count histograms and reward_std rates overall and per bucket."""
    total_prompts = len(records)
    total_completions = sum(r["n_rollouts"] for r in records)
    total_gen_tokens = sum(sum(r["completion_tokens"]) for r in records)

    def _agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        cnt = len(subset)
        hist = Counter(r["pass_count"] for r in subset)
        mid_cnt = sum(hist[k] for k in range(1, n_rollouts))
        std_gt0 = sum(1 for r in subset if float(r["reward_std"]) > 0.0)
        tot_comp = max(1, sum(r["n_rollouts"] for r in subset))
        fmt_ok = sum(r["format_pass_count"] for r in subset)
        leaks = sum(r["leak_count"] for r in subset)
        mean_rew = sum(r["reward_mean"] for r in subset) / max(1, cnt)
        mean_tok = sum(sum(r["completion_tokens"]) for r in subset) / tot_comp
        return {
            "count": cnt,
            "pass_hist": {str(k): hist.get(k, 0) for k in range(n_rollouts + 1)},
            "mid_25_75_count": mid_cnt,
            "zero_pass_count": hist.get(0, 0),
            "full_pass_count": hist.get(n_rollouts, 0),
            "reward_std_gt0_count": std_gt0,
            "reward_std_gt0_rate": round(std_gt0 / max(1, cnt), 4),
            "zero_std_rate": round(1.0 - (std_gt0 / max(1, cnt)), 4),
            "format_pass_rate": round(fmt_ok / tot_comp, 4),
            "leak_rate": round(leaks / tot_comp, 4),
            "mean_reward": round(mean_rew, 4),
            "mean_completion_tokens": round(mean_tok, 2),
        }

    by_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_b[r["bucket"]].append(r)

    return {
        "wall_seconds": round(wall_seconds, 2),
        "n_rollouts": n_rollouts,
        "total_prompts": total_prompts,
        "total_completions": total_completions,
        "total_completion_tokens": total_gen_tokens,
        "prompts_per_sec": round(total_prompts / max(0.01, wall_seconds), 2),
        "completions_per_sec": round(total_completions / max(0.01, wall_seconds), 2),
        "gen_tokens_per_sec": round(total_gen_tokens / max(0.01, wall_seconds), 2),
        "overall": _agg(records),
        "by_bucket": {b: _agg(by_b[b]) for b in sorted(by_b)},
    }


async def _query_one_prompt(
    client: httpx.AsyncClient,
    endpoint: str,
    model_name: str,
    prompt_text: str,
    *,
    n: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    max_retries: int = 3,
) -> tuple[list[str], list[int]]:
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stop": ["<|im_end|>"],
    }
    url = f"{endpoint.rstrip('/')}/v1/completions"
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = sorted(data["choices"], key=lambda c: int(c.get("index", 0)))
            texts = [str(c.get("text") or "") for c in choices]
            usage = data.get("usage") or {}
            comp_toks_total = int(usage.get("completion_tokens", 0))
            per_lens = [max(1, comp_toks_total // max(1, len(texts)))] * len(texts)
            for idx, c in enumerate(choices):
                lp = c.get("logprobs")
                if isinstance(lp, dict) and isinstance(lp.get("tokens"), list):
                    per_lens[idx] = len(lp["tokens"])
            return texts, per_lens
        except Exception as exc:
            last_err = exc
            if attempt + 1 < max_retries:
                await asyncio.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"vLLM request failed at {url}: {last_err}") from last_err


async def run_dp2_rollouts(
    rows: list[dict[str, Any]],
    *,
    endpoints: list[str],
    model_name: str,
    n: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    concurrency_per_server: int,
) -> list[dict[str, Any]]:
    """Run work-stealing async rollouts across multiple vLLM endpoints and score them."""
    import httpx

    queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
    for idx, row in enumerate(rows):
        queue.put_nowait((idx, row))

    results: list[dict[str, Any] | None] = [None] * len(rows)
    completed = 0
    total = len(rows)
    t0 = time.monotonic()

    limits = httpx.Limits(
        max_connections=concurrency_per_server * len(endpoints) + 16,
        max_keepalive_connections=concurrency_per_server * len(endpoints) + 16,
    )
    timeout = httpx.Timeout(300.0, connect=30.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:

        async def _worker(endpoint: str) -> None:
            nonlocal completed
            while True:
                try:
                    idx, row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    texts, lens = await _query_one_prompt(
                        client,
                        endpoint,
                        model_name,
                        row["prompt"],
                        n=n,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_tokens=max_tokens,
                    )
                    results[idx] = score_prompt_rollouts(row, texts, lens)
                    completed += 1
                    if completed % 250 == 0 or completed == total:
                        dt = time.monotonic() - t0
                        rate = completed / max(0.01, dt)
                        eta = (total - completed) / max(0.01, rate)
                        print(
                            f"[rollout] {completed}/{total} prompts ({rate:.2f} prompts/s, elapsed={dt:.1f}s, eta={eta:.1f}s)",
                            flush=True,
                        )
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(_worker(ep))
            for ep in endpoints
            for _ in range(concurrency_per_server)
        ]
        await asyncio.gather(*workers)

    assert all(r is not None for r in results)
    return [r for r in results if r is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Candidate JSONL path")
    parser.add_argument("--output", type=Path, required=True, help="Scored rollouts JSONL path")
    parser.add_argument("--summary", type=Path, required=True, help="Rollout summary JSON path")
    parser.add_argument(
        "--endpoints",
        type=str,
        default="http://127.0.0.1:8000,http://127.0.0.1:8001",
        help="Comma-separated vLLM server base URLs",
    )
    parser.add_argument("--model", type=str, default="merged-checkpoint-68")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency-per-server", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--skip-sha-check", action="store_true")
    args = parser.parse_args()

    inp = args.input.resolve()
    if not args.skip_sha_check and (inp.parent / "manifest.json").exists():
        verified = verify_data_manifest(inp.parent, [inp.name])
        print("Manifest SHA-256 verified:", verified, flush=True)

    rows = [json.loads(line) for line in inp.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_records is not None:
        rows = rows[: args.max_records]

    endpoints = [ep.strip() for ep in args.endpoints.split(",") if ep.strip()]
    print(
        f"Starting DP={len(endpoints)} rollout on {len(rows)} prompts (n={args.n}, T={args.temperature}, concurrency={args.concurrency_per_server}/GPU)",
        flush=True,
    )
    t0 = time.monotonic()
    scored = asyncio.run(
        run_dp2_rollouts(
            rows,
            endpoints=endpoints,
            model_name=args.model,
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            concurrency_per_server=args.concurrency_per_server,
        )
    )
    wall_seconds = time.monotonic() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for rec in scored:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize_scored_rollouts(scored, wall_seconds=wall_seconds, n_rollouts=args.n)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
