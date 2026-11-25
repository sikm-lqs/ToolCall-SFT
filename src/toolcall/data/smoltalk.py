"""SmolTalk general-QA loader and high-distraction irrelevant-tool attacher (SFT-v1r1)."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
from pathlib import Path
import random
import re
from typing import Any

import pyarrow.parquet as pq

from .schema import validate_record

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
        "does", "for", "from", "get", "give", "given", "has", "have", "how",
        "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on",
        "or", "please", "provide", "return", "set", "should", "show", "some",
        "tell", "that", "the", "their", "them", "there", "these", "this", "to",
        "use", "using", "value", "values", "what", "when", "where", "which",
        "who", "will", "with", "would", "you", "your", "api", "function",
        "parameter", "string", "integer", "number", "boolean", "object",
        "array", "default", "specified", "based", "data", "info", "information",
        "list", "details", "fetch", "retrieve", "check", "find",
    }
)

# Exclude generic math/text/code/search helper tools so attaching them to
# general QA, math, or writing prompts cannot create false-negative labels.
_EXCLUDED_TOOL_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "ai", "algebra", "answer", "area", "arithmetic", "ask", "assistant",
        "auto", "average", "bot", "calc", "calculate", "calculation",
        "calculator", "captcha", "chat", "chatbot", "code", "combination",
        "compiler", "compute", "convert", "converter", "currency", "date",
        "define", "definition", "derivative", "dialogue", "dictionary", "diff",
        "differential", "divide", "equation", "eval", "evaluate", "exchange",
        "exec", "execute", "explain", "extract", "factor", "factorial",
        "fibonacci", "forecast", "formula", "geometry", "gpt", "grammar",
        "helper", "integral", "interpreter", "llm", "math", "mathematics",
        "matrix", "multiply", "news", "openai", "paraphrase", "prime",
        "problem", "proofread", "python", "quadratic", "quiz", "rewrite",
        "root", "script", "search", "simplify", "solve", "solver", "spell",
        "sql", "sqrt", "statistics", "stock", "subtract", "suggest", "sum",
        "summarize", "summarizer", "summary", "temperature", "text", "time",
        "timezone", "tokenize", "translate", "translator", "translation",
        "triangle", "trivia", "tutor", "unit", "volume", "weather", "web",
        "write", "paraphraser",
    }
)

# High-risk real-time / functional topics: never attach tools whose doc_tokens
# touch the same topic when the query itself also mentions it.
_SENSITIVE_TOPIC_TOKENS: frozenset[str] = frozenset(
    {
        "weather", "forecast", "temperature", "rain", "humidity", "climate",
        "currency", "exchange", "usd", "eur", "stock", "price", "crypto",
        "bitcoin", "flight", "hotel", "restaurant", "recipe", "calorie",
        "movie", "song", "music", "lyrics", "email", "sms", "tweet", "twitter",
        "instagram", "youtube", "tiktok", "spotify", "database", "sql",
        "translate", "translation", "dictionary", "synonym", "antonym",
        "route", "directions", "distance", "map", "address", "postcode", "zip",
        "timezone", "calendar", "schedule", "reminder", "alarm",
    }
)

_FORBIDDEN_CONTENT_MARKERS: tuple[str, ...] = (
    "<tool_call>",
    "<tool_response>",
    "<function=",
    "<|im_start|>",
    "<|im_end|>",
    "<functioncall>",
)


def _split_identifier_tokens(text: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return _TOKEN_RE.findall(spaced.lower())


def _tokenize_content(text: str) -> set[str]:
    return {
        tok
        for tok in _split_identifier_tokens(text)
        if len(tok) >= 3 and tok not in _STOPWORDS
    }


def _tool_signature(tool: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    fn = tool["function"]
    props = sorted((fn.get("parameters") or {}).get("properties", {}).keys())
    return fn["name"].lower(), tuple(props)


def _is_safe_distractor_tool(tool: dict[str, Any]) -> bool:
    fn = tool["function"]
    name_tokens = set(_split_identifier_tokens(fn["name"]))
    if name_tokens & _EXCLUDED_TOOL_NAME_TOKENS:
        return False
    desc_head = " ".join((fn.get("description") or "").lower().split()[:12])
    if any(
        phrase in desc_head
        for phrase in (
            "solve math",
            "calculate the area",
            "translate text",
            "summarize text",
            "execute python",
            "web search",
        )
    ):
        return False
    return True


class DistractorToolIndex:
    """Inverted-index retriever for attaching 1-4 high-distraction irrelevant tools."""

    def __init__(self, candidate_records: list[dict[str, Any]]) -> None:
        seen_sigs: set[tuple[str, tuple[str, ...]]] = set()
        self.tools: list[dict[str, Any]] = []
        self.name_tokens: list[set[str]] = []
        self.doc_tokens: list[set[str]] = []

        for rec in candidate_records:
            for tool in rec.get("tools") or []:
                sig = _tool_signature(tool)
                if sig in seen_sigs or not _is_safe_distractor_tool(tool):
                    continue
                seen_sigs.add(sig)
                fn = tool["function"]
                props = (fn.get("parameters") or {}).get("properties") or {}
                text_parts = [fn["name"].replace("_", " ").replace(".", " "), fn.get("description") or ""]
                for pname, pinfo in props.items():
                    text_parts.append(pname.replace("_", " "))
                    if isinstance(pinfo, dict) and pinfo.get("description"):
                        text_parts.append(str(pinfo["description"]))
                toks = _tokenize_content(" ".join(text_parts))
                if not toks:
                    continue
                self.tools.append(deepcopy(tool))
                self.name_tokens.append(_tokenize_content(fn["name"].replace("_", " ").replace(".", " ")))
                self.doc_tokens.append(toks)

        self.inv: dict[str, list[int]] = defaultdict(list)
        for idx, toks in enumerate(self.doc_tokens):
            for tok in toks:
                self.inv[tok].append(idx)

        n_docs = max(1, len(self.tools))
        self.idf: dict[str, float] = {
            tok: math.log((n_docs + 1.0) / (len(postings) + 1.0)) + 1.0
            for tok, postings in self.inv.items()
            if len(postings) <= max(50, int(n_docs * 0.15))
        }

    def sample_distractors(
        self,
        query_text: str,
        k: int,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        """Retrieve `k` unique-named distractor tools with lexical overlap but no direct match."""
        q_toks = _tokenize_content(query_text)
        scores: Counter[int] = Counter()
        for tok in q_toks:
            weight = self.idf.get(tok)
            if weight is None:
                continue
            for doc_idx in self.inv[tok]:
                scores[doc_idx] += weight

        ranked = [doc_idx for doc_idx, _ in scores.most_common(60)]
        # Skip the very highest-overlap items if their name overlaps strongly with the query.
        filtered: list[int] = []
        for doc_idx in ranked:
            ntoks = self.name_tokens[doc_idx]
            dtoks = self.doc_tokens[doc_idx]
            if ntoks and len(ntoks & q_toks) >= 2 and (len(ntoks & q_toks) / len(ntoks)) > 0.5:
                continue
            # Topic-conflict guard: never attach a tool touching a real-time/API topic
            # if the user query also mentions that same sensitive topic.
            if (q_toks & _SENSITIVE_TOPIC_TOKENS) & dtoks:
                continue
            filtered.append(doc_idx)

        rng.shuffle(filtered)
        chosen: list[dict[str, Any]] = []
        used_names: set[str] = set()

        for doc_idx in filtered:
            tool = self.tools[doc_idx]
            tname = tool["function"]["name"]
            if tname in used_names:
                continue
            used_names.add(tname)
            chosen.append(deepcopy(tool))
            if len(chosen) == k:
                return chosen

        # Top up with random safe tools if lexical pool was smaller than k
        if len(chosen) < k and self.tools:
            for _ in range(k * 12):
                doc_idx = rng.randrange(len(self.tools))
                if (q_toks & _SENSITIVE_TOPIC_TOKENS) & self.doc_tokens[doc_idx]:
                    continue
                tool = self.tools[doc_idx]
                tname = tool["function"]["name"]
                if tname not in used_names:
                    used_names.add(tname)
                    chosen.append(deepcopy(tool))
                    if len(chosen) == k:
                        break

        return chosen


def load_smoltalk_irrelevant_records(
    smoltalk_dir: Path,
    subset_targets: dict[str, int],
    distractor_index: DistractorToolIndex,
    seed: int,
    max_total_chars: int = 3500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample clean SmolTalk QA/chat rows and attach 1-4 high-distraction irrelevant tools."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    tool_counts_choices = (1, 1, 2, 3, 4)

    for subset_name, target_count in sorted(subset_targets.items()):
        subset_dir = smoltalk_dir / subset_name
        parquet_files = sorted(subset_dir.glob("train-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No train parquet files under {subset_dir}")

        candidates: list[tuple[int, list[dict[str, Any]]]] = []
        skipped = 0
        raw_total = 0

        for pf_path in parquet_files:
            rows = pq.ParquetFile(pf_path).read().to_pylist()
            for row_idx, row in enumerate(rows):
                raw_total += 1
                msgs = row.get("messages") or []
                if not msgs or msgs[0].get("role") == "system":
                    skipped += 1
                    continue
                if msgs[-1].get("role") != "assistant":
                    skipped += 1
                    continue
                # Enforce alternating user/assistant turns with clean non-empty text
                valid_turns = True
                total_chars = 0
                clean_msgs: list[dict[str, str]] = []
                for turn_idx, msg in enumerate(msgs):
                    expected_role = "user" if turn_idx % 2 == 0 else "assistant"
                    content = str(msg.get("content") or "").strip()
                    if msg.get("role") != expected_role or not content:
                        valid_turns = False
                        break
                    if any(marker in content for marker in _FORBIDDEN_CONTENT_MARKERS):
                        valid_turns = False
                        break
                    total_chars += len(content)
                    clean_msgs.append({"role": expected_role, "content": content})

                if not valid_turns or total_chars > max_total_chars:
                    skipped += 1
                    continue
                candidates.append((row_idx, clean_msgs))

        rng.shuffle(candidates)
        chosen_subset = candidates[: int(target_count)]

        built_subset = 0
        for row_idx, clean_msgs in chosen_subset:
            k_tools = rng.choice(tool_counts_choices)
            query_text = " ".join(m["content"] for m in clean_msgs if m["role"] == "user")
            tools = distractor_index.sample_distractors(query_text, k=k_tools, rng=rng)
            if not tools:
                continue

            num_user_turns = sum(1 for m in clean_msgs if m["role"] == "user")
            record = {
                "messages": clean_msgs,
                "tools": tools,
                "meta": {
                    "source": f"smoltalk:{subset_name}",
                    "source_id": f"{subset_name}:{row_idx}",
                    "bucket": "smoltalk_irrelevant_tools",
                    "num_turns": num_user_turns,
                    "num_user_turns": num_user_turns,
                    "num_tool_messages": 0,
                    "num_tool_calls": 0,
                    "num_candidate_tools": len(tools),
                    "has_tool_response": False,
                    "is_parallel": False,
                    "is_refusal": True,
                },
            }
            ok, _ = validate_record(record)
            if ok:
                records.append(record)
                built_subset += 1

        stats[subset_name] = {
            "raw_rows": raw_total,
            "eligible_rows": len(candidates),
            "skipped_rows": skipped,
            "sampled_rows": built_subset,
        }

    return records, stats
