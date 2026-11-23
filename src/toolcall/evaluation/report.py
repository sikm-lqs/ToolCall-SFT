"""Build dashboard-ready per-run charts and SwanLab metrics."""

from pathlib import Path


def write_charts(
    run_dir: Path, summary: dict, resources: list[dict]
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plots = {}
    score_key = summary["score_key"]
    groups = summary["groups"]

    def group_label(label, row, metric):
        name = label.replace("_", " ")
        count = (
            row.get("response_samples")
            if metric == "rouge_l_f1"
            else row.get("total_count", row.get("calls", row.get("tasks")))
        )
        return f"{name} (n={count})" if count is not None else name

    for chart, metric, title in (
        ("scores", score_key, score_key.replace("_", " ").capitalize()),
        ("response-quality", "rouge_l_f1", "Response quality · ROUGE-L F1"),
        ("tool-coverage", "reference_tool_coverage", "Reference tool coverage"),
    ):
        scores = [
            (group_label(label, row, metric), row[metric])
            for label, row in groups.items()
            if metric in row
        ]
        if not scores:
            continue
        labels, values = zip(*scores)
        fig, ax = plt.subplots(
            figsize=(9, max(3, len(labels) * 0.38 + 1.4)), layout="constrained"
        )
        bars = ax.barh(labels, values, color="#247d78")
        if metric == "rouge_l_f1":
            ax.bar_label(bars, [f"{value:.3f}" for value in values], padding=5)
        else:
            ax.bar_label(bars, [f"{value:.1%}" for value in values], padding=5)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set(xlim=(0, 1.14), title=title)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
        plots[chart] = run_dir / f"{chart}.png"
        fig.savefig(plots[chart], dpi=150)
        plt.close(fig)

    runtime_keys = [
        ("num_requests_running", "Running requests"),
        ("num_requests_waiting", "Waiting requests"),
        ("kv_cache_usage_perc", "KV cache (%)"),
        ("gpu_used_mib", "GPU memory (GiB)"),
    ]
    tracks = [
        (key, label)
        for key, label in runtime_keys
        if any(key in row for row in resources)
    ]
    if tracks:
        fig, axes = plt.subplots(
            len(tracks),
            1,
            figsize=(9, 1.6 * len(tracks) + 0.5),
            sharex=True,
            squeeze=False,
            layout="constrained",
        )
        for ax, (key, label) in zip(axes[:, 0], tracks):
            points = [
                (row["elapsed_seconds"], row[key]) for row in resources if key in row
            ]
            factor = (
                100
                if key == "kv_cache_usage_perc"
                else 1 / 1024
                if key == "gpu_used_mib"
                else 1
            )
            ax.plot(
                [x for x, _ in points],
                [value * factor for _, value in points],
                color="#285ca4",
                marker=".",
            )
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.2)
        axes[-1, 0].set_xlabel("Elapsed seconds")
        plots["resources"] = run_dir / "resources.png"
        fig.savefig(plots["resources"], dpi=150)
        plt.close(fig)
    return plots


def scalar_metrics(summary: dict) -> dict:
    sections = [
        ("", summary["metrics"]),
        *[(f"groups/{name}/", row) for name, row in summary["groups"].items()],
    ]
    return {
        prefix + key: value
        for prefix, section in sections
        for key, value in section.items()
        if type(value) in (float, int)
    }
