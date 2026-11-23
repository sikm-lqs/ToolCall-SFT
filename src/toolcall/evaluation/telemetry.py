"""Sample vLLM queue/cache metrics and optional local GPU statistics."""

from contextlib import contextmanager
import json
from pathlib import Path
import re
from threading import Event, Thread
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


@contextmanager
def monitor(base_url: str, path: Path, gpu_index: int | None):
    opener = build_opener(ProxyHandler({}))
    stop = Event()
    names = (
        "num_requests_running",
        "num_requests_waiting",
        "kv_cache_usage_perc",
        "num_preemptions_total",
    )
    patterns = {
        name: re.compile(rf"(?m)^vllm:{name}(?:\{{[^\n]*\}})?\s+([0-9.eE+\-]+)$")
        for name in names
    }
    if gpu_index is not None:
        import pynvml

        pynvml.nvmlInit()
        try:
            gpu = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            pynvml.nvmlShutdown()
            raise

    def sample():
        started = time.monotonic()
        with path.open("w", encoding="utf-8") as stream:
            while not stop.is_set():
                row = {"elapsed_seconds": time.monotonic() - started}
                try:
                    with opener.open(
                        base_url.removesuffix("/v1") + "/metrics", timeout=1
                    ) as response:
                        body = response.read().decode()
                    for name, pattern in patterns.items():
                        values = [float(value) for value in pattern.findall(body)]
                        if values:
                            row[name] = (
                                max(values)
                                if name == "kv_cache_usage_perc"
                                else sum(values)
                            )
                except (URLError, TimeoutError, OSError):
                    pass
                if gpu_index is not None:
                    row["gpu_used_mib"] = (
                        pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 2**20
                    )
                    row["gpu_utilization_pct"] = pynvml.nvmlDeviceGetUtilizationRates(
                        gpu
                    ).gpu
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                stop.wait(0.5)

    thread = Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        if gpu_index is not None:
            pynvml.nvmlShutdown()


def peak_metrics(rows: list[dict]) -> dict:
    metrics = {}
    for key in (
        "gpu_used_mib",
        "gpu_utilization_pct",
        "num_requests_running",
        "num_requests_waiting",
        "kv_cache_usage_perc",
    ):
        values = [row[key] for row in rows if key in row]
        if values:
            metrics["peak_" + key] = max(values)
    preemptions = [
        row["num_preemptions_total"] for row in rows if "num_preemptions_total" in row
    ]
    if preemptions:
        metrics["preemptions"] = preemptions[-1] - preemptions[0]
    return metrics
