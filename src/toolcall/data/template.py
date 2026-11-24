"""Qwen3.5 chat template patching (`{% generation %}`) and loss-mask verification."""

from __future__ import annotations

from typing import Any

_ORIGINAL_ASSISTANT_BLOCK = """    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                    {%- else %}
                        {{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\\n' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '\\n</parameter>\\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}"""

_PATCHED_ASSISTANT_BLOCK = """    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' }}
        {%- endif %}
        {%- generation %}
        {{- content }}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                    {%- else %}
                        {{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\\n' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '\\n</parameter>\\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>' }}
        {%- endgeneration %}
        {{- '\\n' }}"""


def patch_qwen35_chat_template(raw_template: str) -> str:
    """Wrap Qwen3.5 assistant outputs + `<|im_end|>` in `{% generation %}` without altering rendered text."""
    if "{%- generation %}" in raw_template or "{% generation %}" in raw_template:
        return raw_template
    if _ORIGINAL_ASSISTANT_BLOCK not in raw_template:
        raise ValueError("Could not locate expected Qwen3.5 assistant block in chat_template.jinja")
    return raw_template.replace(_ORIGINAL_ASSISTANT_BLOCK, _PATCHED_ASSISTANT_BLOCK, 1)


def render_and_mask(
    tokenizer: Any,
    record: dict[str, Any],
    patched_template: str,
    verify_equivalence: bool = False,
) -> dict[str, Any]:
    """Tokenize a normalized record and return `input_ids`, `assistant_masks`, and decoded segments."""
    messages = record["messages"]
    tools = record.get("tools") or None

    if verify_equivalence:
        orig_text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        patched_text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=patched_template,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if orig_text != patched_text:
            raise AssertionError("Patched template rendered text diverges from official template")

    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=patched_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=False,
    )
    input_ids: list[int] = encoded["input_ids"]
    mask: list[int] = encoded["assistant_masks"]
    if len(input_ids) != len(mask):
        raise AssertionError(f"Length mismatch: ids={len(input_ids)} vs mask={len(mask)}")

    expected_turns = sum(1 for m in messages if m["role"] == "assistant")
    segments: list[str] = []
    idx = 0
    while idx < len(mask):
        if mask[idx] == 1:
            end = idx
            while end < len(mask) and mask[end] == 1:
                end += 1
            segments.append(tokenizer.decode(input_ids[idx:end], skip_special_tokens=False))
            idx = end
        else:
            idx += 1

    if len(segments) != expected_turns:
        raise AssertionError(
            f"Assistant mask segment count {len(segments)} != expected turns {expected_turns}"
        )
    for seg in segments:
        if not seg.endswith("<|im_end|>"):
            raise AssertionError(f"Assistant mask segment missing <|im_end|>: {seg!r}")
        if "<|im_start|>user" in seg or "<|im_start|>system" in seg or "<tool_response>" in seg:
            raise AssertionError(f"Non-assistant marker leaked into loss segment: {seg!r}")

    return {
        "input_ids": input_ids,
        "assistant_masks": mask,
        "num_tokens": len(input_ids),
        "num_assistant_tokens": sum(mask),
        "assistant_segments": segments,
    }


def format_masked_inspection(
    tokenizer: Any,
    input_ids: list[int],
    assistant_masks: list[int],
) -> str:
    """Format a human-readable `[MASK]` vs `[TRAIN]` inspection view of tokenized chat spans."""
    lines: list[str] = []
    idx = 0
    while idx < len(assistant_masks):
        flag = assistant_masks[idx]
        end = idx
        while end < len(assistant_masks) and assistant_masks[end] == flag:
            end += 1
        tag = "TRAIN (labels != -100)" if flag == 1 else "MASK  (labels == -100)"
        text = tokenizer.decode(input_ids[idx:end], skip_special_tokens=False)
        lines.append(f"=== [{tag}] tokens={end - idx} ===")
        lines.append(text)
        idx = end
    return "\n".join(lines) + "\n"
