"""Render all captured first requests locally, without model weights or inference."""

from copy import deepcopy
from hashlib import file_digest
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

from evaluation import OUT, read, write


def main(snapshot):
    model = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    for index in sorted(OUT.glob("*/index.json")):
        summary = read(index)
        for row in summary["subsets"]:
            path = index.parent / f"{row['category']}.json"
            record = read(path)
            request = record["request"]
            messages = deepcopy(request["messages"])
            for message in messages:
                for call in message.get("tool_calls") or []:
                    fn = call["function"]
                    if isinstance(fn["arguments"], str):
                        fn["arguments"] = json.loads(fn["arguments"])
            prompt = model.apply_chat_template(
                messages,
                tools=request.get("tools"),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            target = path.with_suffix(".prompt.txt")
            target.write_text(prompt)
            row["prompt_tokens"] = len(model.encode(prompt, add_special_tokens=False))
            row["prompt_file"] = str(Path(row["path"]).with_suffix(".prompt.txt"))
        write(index, summary)
    files = {}
    for name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        with (snapshot / name).open("rb") as stream:
            files[name] = file_digest(stream, "sha256").hexdigest()
    write(
        OUT / "qwen3.5-2b-tokenizer.json",
        dict(files=files, enable_thinking=False, local_only=True),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
