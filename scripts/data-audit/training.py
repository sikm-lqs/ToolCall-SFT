"""Inventory local ModelScope snapshots; never execute downloaded dataset code."""

from collections import Counter, defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import urllib.parse
import urllib.request

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SOURCES = {
    "xlam-60k": (
        "Salesforce/xlam-function-calling-60k",
        "Salesforce/xlam-function-calling-60k",
        133242,
        "CC BY 4.0",
    ),
    "apigen-mt-5k": (
        "AI-ModelScope/APIGen-MT-5k",
        "Salesforce/APIGen-MT-5k",
        103067,
        "CC BY-NC 4.0",
    ),
    "toolace": (
        "mlabonne/ToolACE",
        "Team-ACE/ToolACE",
        84236,
        "镜像卡未声明；以 Team-ACE 原始许可为准",
    ),
    "glaive-v2": (
        "AI-ModelScope/glaive-function-calling-v2",
        "glaiveai/glaive-function-calling-v2",
        19092,
        "镜像数据卡：Apache 2.0",
    ),
    "magpie-pro-300k": (
        "Magpie-Align/Magpie-Pro-300K-Filtered",
        "Magpie-Align/Magpie-Pro-300K-Filtered",
        65035,
        "镜像仓库元数据为 Apache 2.0，随附数据卡为 llama3；按未解决的冲突处理",
    ),
    "smoltalk": (
        "HuggingFaceTB/smoltalk",
        "HuggingFaceTB/smoltalk",
        142109,
        "仓库标注 Apache 2.0；混合数据需逐来源核对",
    ),
}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def tree(hub_id):
    rows = {}
    for page in range(1, 50):
        params = urllib.parse.urlencode(
            dict(
                Revision="master",
                Root="/",
                Recursive="True",
                PageNumber=page,
                PageSize=100,
            )
        )
        url = f"https://modelscope.cn/api/v1/datasets/{hub_id}/repo/tree?{params}"
        with urllib.request.urlopen(url, timeout=20) as response:
            payload = json.load(response)
        assert payload["Code"] == 200, payload
        batch = payload["Data"]["Files"]
        for row in batch:
            if row["Type"] == "blob":
                rows[row["Path"]] = row
        if len(batch) < 100:
            return rows
    raise RuntimeError("Unexpected ModelScope tree pagination")


def inspect(name, versions):
    repo, author, hub_id, license_note = versions
    root = DATA / "raw" / name
    files = tree(hub_id)
    groups = defaultdict(lambda: {"rows": 0, "files": [], "fields": {}})
    manifest = {}
    samples = {}
    stats = Counter()
    for relative, info in sorted(files.items()):
        path = root / relative
        assert path.is_file(), path
        size = path.stat().st_size
        assert size == info["Size"], (relative, size, info["Size"])
        with path.open("rb") as f:
            digest = file_digest(f, "sha256").hexdigest()
        expected = info.get("Sha256")
        if expected:
            assert digest == expected, path
        manifest[relative] = dict(
            bytes=size,
            sha256=digest,
            upstream_sha256=expected or None,
            file_commit=info.get("CommitId"),
        )
        if path.suffix == ".parquet":
            pf = pq.ParquetFile(path)
            split = path.name.partition("-")[0]
            subset = path.parent.name if name == "smoltalk" else "default"
            key = f"{subset}/{split}"
            group = groups[key]
            group["rows"] += pf.metadata.num_rows
            fields = {f.name: str(f.type) for f in pf.schema_arrow}
            if group["fields"]:
                assert group["fields"] == fields, relative
            group["fields"] = fields
            if key not in samples:
                first = next(pf.iter_batches(batch_size=1)).to_pylist()[0]
                samples[key] = {"file": relative, "row_index": 0, "record": first}
        elif relative in {
            "xlam_function_calling_60k.json",
            "apigen-mt_5k.json",
            "glaive-function-calling-v2.json",
        }:
            with path.open() as f:
                records = json.load(f)
            key = "default/train"
            group = groups[key]
            group["rows"] += len(records)
            types = defaultdict(set)
            for row in records:
                for field, value in row.items():
                    types[field].add(type(value).__name__)
                if name == "xlam-60k":
                    try:
                        calls = json.loads(row["answers"])
                        json.loads(row["tools"])
                        stats[
                            "answers_empty"
                            if not calls
                            else "answers_multiple"
                            if len(calls) > 1
                            else "answers_single"
                        ] += 1
                    except (ValueError, TypeError, KeyError):
                        stats["json_parse_failed"] += 1
                elif name == "apigen-mt-5k":
                    for message in row["conversations"]:
                        stats[f"role:{message['from']}"] += 1
                else:
                    stats[
                        "chat_has_functioncall_marker"
                        if "<functioncall>" in row["chat"]
                        else "chat_no_functioncall_marker"
                    ] += 1
            group["fields"] = {
                field: " | ".join(sorted(values)) for field, values in types.items()
            }
            index = 0
            samples[key] = {
                "file": relative,
                "row_index": index,
                "record": records[index],
            }
        else:
            continue
        group["files"].append(relative)
    assert groups, name
    log = DATA / "manifests/download-logs" / f"{name}.exit"
    assert log.read_text().strip() == "0", name
    dump(DATA / "training-inspection" / name / "examples.json", samples)
    return dict(
        name=name,
        modelscope=repo,
        author_source=author,
        revision="master",
        license=license_note,
        files=manifest,
        bytes=sum(x["bytes"] for x in manifest.values()),
        splits=groups,
        stats=stats,
    )


if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or list(SOURCES)
    for name in names:
        result = inspect(name, SOURCES[name])
        dump(DATA / "manifests" / "training" / f"{name}.json", result)
        print(
            name,
            result["bytes"],
            {k: v["rows"] for k, v in result["splits"].items()},
            flush=True,
        )
