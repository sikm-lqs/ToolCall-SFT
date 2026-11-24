"""Generate a Markdown report and offline HTML index from the audited datasets."""

from html import escape
from pathlib import Path
from string import Template

from evaluation import OUT, ROOT, read

REPORTS = ROOT / "reports"
TRAIN = {
    "xlam-60k": (
        "工具选择、参数填充、并行",
        "id + query + tools + answers；后两列是 JSON 字符串。首条对同一工具分别请求 beta / game；没有 tool observation。",
    ),
    "apigen-mt-5k": (
        "多轮、策略、工具反馈",
        "system + tools（JSON 字符串）+ conversations[{from,value}]；human / gpt / function_call / observation。首条为航空售后任务。",
    ),
    "toolace": (
        "复杂参数、并行、追问或不调工具",
        "所下载的镜像只有 conversations[{from,value}]；system 是列表首项，不是独立列。工具定义在 system 文本内；assistant 有 [函数名(参数=值)] 形式，函数名可能含空格。",
    ),
    "glaive-v2": (
        "单轮及多轮混合、无调用边界",
        "system + chat，都是字符串；用 USER / ASSISTANT / <functioncall> / FUNCTION RESPONSE 之类的标记拼接。首条用户想订机票，工具只能查汇率。",
    ),
    "magpie-pro-300k": (
        "通用对话与指令保持",
        "uuid + conversations[{from,value}]；以 human/gpt 文本作监督，没有工具列。首条为自然语言概念解释。",
    ),
    "smoltalk": (
        "通用约束、改写、摘要等混合能力",
        "按配置与 split 读取 Parquet；含 messages[{role,content}]，确切字段见清单。all 是聚合视图，不与单独子集叠加。",
    ),
}
SUITES = {
    "bfcl-v4": ("BFCL V4", "工具选择、结构化参数、并行、拒调、多轮状态和检索"),
    "api-bank": ("API-Bank", "固定参考历史下的逐步检索、调用与自然语言回复"),
    "tau2-bench": ("τ²-Bench", "模型与用户模拟器交互，按任务验收业务状态与结果"),
}
PURPOSE = {
    "simple_python": "一个候选、一次 Python 调用；AST 参数允许值",
    "simple_java": "一个候选、一次 Java 调用；部分参数按 Java 字符串约定表示",
    "simple_javascript": "一个候选、一次 JavaScript 调用；部分参数按 JS 字符串约定表示",
    "multiple": "多个候选中选择一个；AST 函数与参数",
    "parallel": "一个候选发起多个调用；AST 匹配调用集合",
    "parallel_multiple": "多个候选发起多个调用；AST 匹配调用集合",
    "irrelevance": "无合适工具；不得输出可解码的非空调用",
    "live_simple": "实际使用分布下一个候选、一次调用；AST",
    "live_multiple": "实际使用分布下多选一；AST",
    "live_parallel": "实际使用分布下重复调用同一个工具；AST",
    "live_parallel_multiple": "实际使用分布下多个工具、多次调用；AST",
    "live_irrelevance": "实际使用分布下拒调；检查是否输出非空调用",
    "live_relevance": "应该调用；此子类只检查是否有可解码的非空调用",
    "multi_turn_base": "多轮完成正常任务；参考为每轮动作序列，评估执行与状态",
    "multi_turn_miss_func": "先隐藏必要工具、后补发 schema；评估执行与状态",
    "multi_turn_miss_param": "开始信息不足、后续补参；评估执行与状态",
    "multi_turn_long_context": "大环境或长历史下推进任务；评估执行与状态",
    "memory_kv": "KV 记忆后端；有写入前置步骤，标签是事实答案",
    "memory_vector": "向量检索记忆后端；标签是事实答案",
    "memory_rec_sum": "递归摘要记忆后端；标签是事实答案",
    "web_search_base": "搜索工具提供摘要；评分查最终参考事实",
    "web_search_no_snippet": "相同题源不提供摘要；运行时工具回包不同",
    "format_sensitivity": "组合提示风格和调用表达；不属于 Native FC 计分",
    "l1-call": "工具描述直接可见；工具名、参数和参考执行结果",
    "l1-response": "参考调用已执行；生成下一段自然语言回复",
    "l2-call": "从 ToolSearcher 开始；按历史检索结果增补 tools",
    "l2-response": "使用参考检索/调用历史生成自然语言",
    "l3-call": "复杂多 API 链路；用参考历史预测下一次调用",
    "l3-response": "根据链式调用的参考历史生成回复",
    "retail": "零售订单、退换货与身份核验；任务约束和结果",
    "airline": "航班预订、变更或取消；任务约束和结果",
    "telecom": "通信故障诊断与恢复；用户侧与 Agent 侧工具、状态验收",
}
NOTES = (
    (
        "不是每道题都有参考回复",
        "BFCL 单轮大多标函数及可接受参数；相关性只判断有无调用；多轮是执行轨迹/状态；搜索和记忆是答案事实。API-Bank 调用与回复使用不同标签；τ² 用 evaluation_criteria。",
    ),
    (
        "完整首请求 ≠ 整段运行实录",
        "JSON 分开保存 case、request、label、prediction。原生 FC 的完整输入是 messages 加全部 tools；随附 .prompt.txt 为本地 Qwen3.5-2B 离线渲染，不宣称与历史服务逐 token 一致。历史 tool arguments 是字符串，离线渲染前转对象。",
    ),
    (
        "多轮后续由环境和模型决定",
        "已有日志直接保留；其余缺失不伪造。BFCL 的 Web Search 没有本地输出；两个 Web Search 变体的区别体现在工具回包。Memory 使用归档里的首轮记忆提示。",
    ),
    (
        "τ² 区分 Agent 和用户模拟器",
        "任务说明只给用户模拟器，另存 simulator_first_request。Retail 首轮基于既有 smoke；Airline、Telecom 的 Agent 开场为人工示例，没有模拟输出。",
    ),
    (
        "没有调用不总是拒调",
        "没有提供工具、提供的工具不适用、信息不足需要追问，应该分别建桶。Glaive 没有调用标记的记录不是天然的拒调标签。",
    ),
    (
        "仅归档，不等于可直接训练",
        "SmolTalk 的 all 与独立配置不能重复拼接；通用保持不把 apigen-80k 当普通对话。APIGen-MT 与 τ² Retail/Airline 为同域，训练前仍须做评测重合检查。",
    ),
    (
        "镜像一致性与许可",
        "ToolACE 为 mlabonne 镜像，未与作者源逐条比对；Magpie 的 ModelScope 元数据与随附数据卡许可冲突，建议暂不纳入计划发布的训练配方。",
    ),
)
SOURCES = (
    (
        "BFCL V4 官方口径",
        "https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html",
    ),
    (
        "API-Bank 上游",
        "https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank",
    ),
    ("τ²-Bench v0.2.0", "https://github.com/sierra-research/tau2-bench/tree/v0.2.0"),
    ("目录与复现", "../docs/4-数据归档与样本图鉴.md"),
)


def link(label, url):
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'


def markdown_link(label, url):
    return f"[{label}]({url})"


def main():
    md = [
        "# ToolCall-SFT 数据归档与样本图鉴",
        "",
        "2026-11-24（Asia/Shanghai） · 复用项目固定版本；不启动模型，不混入训练。",
        "",
        "## 训练数据",
        "",
        "原始快照在 data/raw/；以下行数由实际文件读取，存档大小用十进制 MB。",
        "",
        "| 数据源 | 行数 | MB | 学什么 | 原始形状和样例 |",
        "|---|---:|---:|---|---|",
    ]
    training, bars, cases = [], [], []
    stats = {}
    total = 0
    for slug, (skill, shape) in TRAIN.items():
        manifest = f"data/manifests/training/{slug}.json"
        item = read(ROOT / manifest)
        stats[slug] = item["stats"]
        size = item["bytes"]
        total += size
        n = sum(
            v["rows"]
            for k, v in item["splits"].items()
            if slug != "smoltalk" or k.startswith("all/")
        )
        md.append(f"| {slug} | {n:,} | {size / 1e6:,.1f} | {skill} | {shape} |")
        example = f"../data/training-inspection/{slug}/examples.json"
        canon = f"https://huggingface.co/datasets/{item['author_source']}"
        training.append(
            f"<details><summary><code>{slug}</code><span>{escape(skill)}</span><b>{n:,} 行</b></summary>"
            f'<div class="body"><p>{escape(shape)}</p><p class="muted">{escape(item["license"])}</p>'
            f"<p>{link('完整原始样例', example)} · {link('字段与校验清单', '../' + manifest)} · "
            f"{link('ModelScope', 'https://modelscope.cn/datasets/' + item['modelscope'])} · {link('作者源', canon)}</p></div></details>"
        )
        bars.append((slug, size))
    md += ["", "### 样例、来源与许可", ""]
    for slug in TRAIN:
        mf = f"../data/manifests/training/{slug}.json"
        ex = f"../data/training-inspection/{slug}/examples.json"
        data = read(ROOT / mf[3:])
        md.append(
            f"- **{slug}**：{data['license']}。{markdown_link('样例', ex)} · {markdown_link('归档校验', mf)}。"
        )
    measured = (
        f"xLAM：{stats['xlam-60k']['answers_single']:,} 单调用、{stats['xlam-60k']['answers_multiple']:,} 多调用，无空调用。"
        f" APIGen-MT：function_call 与 observation 各 {stats['apigen-mt-5k']['role:function_call']:,} 条消息。"
        f" Glaive：{stats['glaive-v2']['chat_has_functioncall_marker']:,} 行带调用标记、"
        f"{stats['glaive-v2']['chat_no_functioncall_marker']:,} 行不带。"
    )
    md += [
        "",
        measured,
        "",
        "### SmolTalk 配置与 split",
        "",
        "| 配置 | Train | Test |",
        "|---|---:|---:|",
    ]
    splits = item["splits"]
    smol_rows = []
    for subset in sorted({k.split("/")[0] for k in splits}):
        train, test = (
            splits.get(f"{subset}/{split}", {}).get("rows", 0)
            for split in ("train", "test")
        )
        md.append(f"| {subset} | {train:,} | {test:,} |")
        smol_rows.append(
            f'<tr><th scope="row">{escape(subset)}</th><td>{train:,}</td><td>{test:,}</td></tr>'
        )
    md += [
        "",
        "## 评测样本",
        "",
        "例如 BFCL simple_python_0 问底为 10、高为 5 的三角形面积。标签并非文本 25，而是 calculate_triangle_area 的可接受参数；2B 历史 smoke 的调用结果为 []。",
        "",
    ]
    for suite, (title, goal) in SUITES.items():
        index = read(OUT / suite / "index.json")
        md += [
            f"### {title}",
            "",
            f"{goal}。版本：{index['version']}。",
            "",
            "| 子类 | 总数 | 测什么 / Label | 样例 | 工具数 / Prompt tokens |",
            "|---|---:|---|---|---:|",
        ]
        for row in index["subsets"]:
            name = row["category"]
            case = read(ROOT / row["path"])
            payload, prompt = ("../" + row[k] for k in ("path", "prompt_file"))
            md.append(
                f"| {name} | {row['rows']:,} | {PURPOSE[name]} | {markdown_link('JSON', payload)} · {markdown_link('Prompt', prompt)} | {row['tools']} / {row['prompt_tokens']:,} |"
            )
            tag = "core" if row.get("core") else row["group"]
            observed = "有历史输出" if case["prediction"] is not None else "无历史输出"
            sample = [
                m.get("content")
                for m in case["request"]["messages"]
                if m["role"] == "user"
            ][-1]
            sample = str(sample).replace("\n", " ")
            sample = sample[:144] + ("…" if len(sample) > 144 else "")
            label_shape = (
                ", ".join(case["label"])
                if isinstance(case["label"], dict)
                else "自然语言字符串"
            )
            cases.append(
                f'<article data-suite="{suite}" data-core="{str(row.get("core", False)).lower()}">'
                f'<details><summary><span class="tag">{escape(title)} · {escape(tag)}</span>'
                f'<code>{escape(name)}</code><span class="muted">{row["rows"]:,} 条</span></summary><div class="body">'
                f"<b>{escape(PURPOSE[name])}</b><p>{escape(sample)}</p>"
                f'<p class="muted"><code>{escape(str(row["id"]))}</code> · {row["tools"]} 工具 · {row["prompt_tokens"]:,} token<br>'
                f"{escape(case['request_status'])} · {observed}<br>标签：{escape(label_shape)}</p>"
                f"<p>{link('完整首请求 Prompt', prompt)} · {link('请求／标签／输出 JSON', payload)}</p></div></details></article>"
            )
        md += [""]
    md += [
        "## 阅读与使用边界",
        "",
        *(f"- **{t}**：{b}" for t, b in NOTES),
        "",
        "## 源码与复现",
        "",
        *(f"- {markdown_link(*s)}" for s in SOURCES),
        "",
    ]
    (REPORTS / "sft-data-analysis.md").write_text("\n".join(md))
    max_size = max(size for _, size in bars)
    bars_html = "".join(
        f'<div class="bar"><code>{name}</code><meter min="0" max="{max_size}" value="{size}" '
        f'aria-label="{name} 下载体积"></meter><span>{size / 1e6:,.1f} MB</span></div>'
        for name, size in bars
    )
    page = Template(
        Path(__file__).with_name("report-template.html").read_text()
    ).substitute(
        total_gb=f"{total / 1e9:.2f}",
        count=len(cases),
        bars=bars_html,
        training="".join(training),
        smol_rows="".join(smol_rows),
        measured=escape(measured),
        cases="".join(cases),
        notes="".join(
            f"<article><h3>{escape(t)}</h3><p>{escape(b)}</p></article>"
            for t, b in NOTES
        ),
        sources=" · ".join(link(*s) for s in SOURCES),
    )
    (REPORTS / "sft-data-analysis.html").write_text(page)


if __name__ == "__main__":
    main()
