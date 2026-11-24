# ToolCall-SFT 数据归档与样本图鉴

2026-11-24（Asia/Shanghai） · 复用项目固定版本；不启动模型，不混入训练。

## 训练数据

原始快照在 data/raw/；以下行数由实际文件读取，存档大小用十进制 MB。

| 数据源 | 行数 | MB | 学什么 | 原始形状和样例 |
|---|---:|---:|---|---|
| xlam-60k | 60,000 | 97.7 | 工具选择、参数填充、并行 | id + query + tools + answers；后两列是 JSON 字符串。首条对同一工具分别请求 beta / game；没有 tool observation。 |
| apigen-mt-5k | 5,000 | 127.8 | 多轮、策略、工具反馈 | system + tools（JSON 字符串）+ conversations[{from,value}]；human / gpt / function_call / observation。首条为航空售后任务。 |
| toolace | 11,300 | 10.6 | 复杂参数、并行、追问或不调工具 | 所下载的镜像只有 conversations[{from,value}]；system 是列表首项，不是独立列。工具定义在 system 文本内；assistant 有 [函数名(参数=值)] 形式，函数名可能含空格。 |
| glaive-v2 | 112,960 | 271.2 | 单轮及多轮混合、无调用边界 | system + chat，都是字符串；用 USER / ASSISTANT / <functioncall> / FUNCTION RESPONSE 之类的标记拼接。首条用户想订机票，工具只能查汇率。 |
| magpie-pro-300k | 300,000 | 563.2 | 通用对话与指令保持 | uuid + conversations[{from,value}]；以 human/gpt 文本作监督，没有工具列。首条为自然语言概念解释。 |
| smoltalk | 1,098,865 | 4,152.7 | 通用约束、改写、摘要等混合能力 | 按配置与 split 读取 Parquet；含 messages[{role,content}]，确切字段见清单。all 是聚合视图，不与单独子集叠加。 |

### 样例、来源与许可

- **xlam-60k**：CC BY 4.0。[样例](../../data/training-inspection/xlam-60k/examples.json) · [归档校验](../../data/manifests/training/xlam-60k.json)。
- **apigen-mt-5k**：CC BY-NC 4.0。[样例](../../data/training-inspection/apigen-mt-5k/examples.json) · [归档校验](../../data/manifests/training/apigen-mt-5k.json)。
- **toolace**：镜像卡未声明；以 Team-ACE 原始许可为准。[样例](../../data/training-inspection/toolace/examples.json) · [归档校验](../../data/manifests/training/toolace.json)。
- **glaive-v2**：镜像数据卡：Apache 2.0。[样例](../../data/training-inspection/glaive-v2/examples.json) · [归档校验](../../data/manifests/training/glaive-v2.json)。
- **magpie-pro-300k**：镜像仓库元数据为 Apache 2.0，随附数据卡为 llama3；按未解决的冲突处理。[样例](../../data/training-inspection/magpie-pro-300k/examples.json) · [归档校验](../../data/manifests/training/magpie-pro-300k.json)。
- **smoltalk**：仓库标注 Apache 2.0；混合数据需逐来源核对。[样例](../../data/training-inspection/smoltalk/examples.json) · [归档校验](../../data/manifests/training/smoltalk.json)。

xLAM：28,461 单调用、31,539 多调用，无空调用。 APIGen-MT：function_call 与 observation 各 21,955 条消息。 Glaive：63,218 行带调用标记、49,742 行不带。

### SmolTalk 配置与 split

| 配置 | Train | Test |
|---|---:|---:|
| all | 1,043,917 | 54,948 |
| apigen-80k | 83,144 | 4,377 |
| everyday-conversations | 2,260 | 119 |
| explore-instruct-rewriting | 30,400 | 1,600 |
| longalign | 3,547 | 187 |
| metamathqa-50k | 47,500 | 2,500 |
| numina-cot-100k | 106,147 | 5,587 |
| openhermes-100k | 95,000 | 5,000 |
| self-oss-instruct | 48,127 | 2,534 |
| smol-constraints | 34,424 | 1,812 |
| smol-magpie-ultra | 409,537 | 21,555 |
| smol-rewrite | 53,342 | 2,808 |
| smol-summarize | 96,356 | 5,072 |
| systemchats-30k | 34,133 | 1,797 |

## 评测样本

例如 BFCL simple_python_0 问底为 10、高为 5 的三角形面积。标签并非文本 25，而是 calculate_triangle_area 的可接受参数；2B 历史 smoke 的调用结果为 []。

### BFCL V4

工具选择、结构化参数、并行、拒调、多轮状态和检索。版本：2026.3.23。

| 子类 | 总数 | 测什么 / Label | 样例 | 工具数 / Prompt tokens |
|---|---:|---|---|---:|
| simple_python | 400 | 一个候选、一次 Python 调用；AST 参数允许值 | [JSON](../../data/eval-inspection/bfcl-v4/simple_python.json) · [Prompt](../../data/eval-inspection/bfcl-v4/simple_python.prompt.txt) | 1 / 371 |
| simple_java | 100 | 一个候选、一次 Java 调用；部分参数按 Java 字符串约定表示 | [JSON](../../data/eval-inspection/bfcl-v4/simple_java.json) · [Prompt](../../data/eval-inspection/bfcl-v4/simple_java.prompt.txt) | 1 / 407 |
| simple_javascript | 50 | 一个候选、一次 JavaScript 调用；部分参数按 JS 字符串约定表示 | [JSON](../../data/eval-inspection/bfcl-v4/simple_javascript.json) · [Prompt](../../data/eval-inspection/bfcl-v4/simple_javascript.prompt.txt) | 1 / 387 |
| multiple | 200 | 多个候选中选择一个；AST 函数与参数 | [JSON](../../data/eval-inspection/bfcl-v4/multiple.json) · [Prompt](../../data/eval-inspection/bfcl-v4/multiple.prompt.txt) | 2 / 709 |
| parallel | 200 | 一个候选发起多个调用；AST 匹配调用集合 | [JSON](../../data/eval-inspection/bfcl-v4/parallel.json) · [Prompt](../../data/eval-inspection/bfcl-v4/parallel.prompt.txt) | 1 / 363 |
| parallel_multiple | 200 | 多个候选发起多个调用；AST 匹配调用集合 | [JSON](../../data/eval-inspection/bfcl-v4/parallel_multiple.json) · [Prompt](../../data/eval-inspection/bfcl-v4/parallel_multiple.prompt.txt) | 2 / 494 |
| irrelevance | 240 | 无合适工具；不得输出可解码的非空调用 | [JSON](../../data/eval-inspection/bfcl-v4/irrelevance.json) · [Prompt](../../data/eval-inspection/bfcl-v4/irrelevance.prompt.txt) | 1 / 368 |
| live_simple | 258 | 实际使用分布下一个候选、一次调用；AST | [JSON](../../data/eval-inspection/bfcl-v4/live_simple.json) · [Prompt](../../data/eval-inspection/bfcl-v4/live_simple.prompt.txt) | 1 / 372 |
| live_multiple | 1,053 | 实际使用分布下多选一；AST | [JSON](../../data/eval-inspection/bfcl-v4/live_multiple.json) · [Prompt](../../data/eval-inspection/bfcl-v4/live_multiple.prompt.txt) | 2 / 827 |
| live_parallel | 16 | 实际使用分布下重复调用同一个工具；AST | [JSON](../../data/eval-inspection/bfcl-v4/live_parallel.json) · [Prompt](../../data/eval-inspection/bfcl-v4/live_parallel.prompt.txt) | 1 / 395 |
| live_parallel_multiple | 24 | 实际使用分布下多个工具、多次调用；AST | [JSON](../../data/eval-inspection/bfcl-v4/live_parallel_multiple.json) · [Prompt](../../data/eval-inspection/bfcl-v4/live_parallel_multiple.prompt.txt) | 2 / 820 |
| live_irrelevance | 884 | 实际使用分布下拒调；检查是否输出非空调用 | [JSON](../../data/eval-inspection/bfcl-v4/live_irrelevance.json) · [Prompt](../../data/eval-inspection/bfcl-v4/live_irrelevance.prompt.txt) | 1 / 986 |
| live_relevance | 16 | 应该调用；此子类只检查是否有可解码的非空调用 | [JSON](../../data/eval-inspection/bfcl-v4/live_relevance.json) · [Prompt](../../data/eval-inspection/bfcl-v4/live_relevance.prompt.txt) | 3 / 1,033 |
| multi_turn_base | 200 | 多轮完成正常任务；参考为每轮动作序列，评估执行与状态 | [JSON](../../data/eval-inspection/bfcl-v4/multi_turn_base.json) · [Prompt](../../data/eval-inspection/bfcl-v4/multi_turn_base.prompt.txt) | 32 / 6,665 |
| multi_turn_miss_func | 200 | 先隐藏必要工具、后补发 schema；评估执行与状态 | [JSON](../../data/eval-inspection/bfcl-v4/multi_turn_miss_func.json) · [Prompt](../../data/eval-inspection/bfcl-v4/multi_turn_miss_func.prompt.txt) | 31 / 6,495 |
| multi_turn_miss_param | 200 | 开始信息不足、后续补参；评估执行与状态 | [JSON](../../data/eval-inspection/bfcl-v4/multi_turn_miss_param.json) · [Prompt](../../data/eval-inspection/bfcl-v4/multi_turn_miss_param.prompt.txt) | 32 / 6,665 |
| multi_turn_long_context | 200 | 大环境或长历史下推进任务；评估执行与状态 | [JSON](../../data/eval-inspection/bfcl-v4/multi_turn_long_context.json) · [Prompt](../../data/eval-inspection/bfcl-v4/multi_turn_long_context.prompt.txt) | 32 / 6,665 |
| memory_kv | 155 | KV 记忆后端；有写入前置步骤，标签是事实答案 | [JSON](../../data/eval-inspection/bfcl-v4/memory_kv.json) · [Prompt](../../data/eval-inspection/bfcl-v4/memory_kv.prompt.txt) | 15 / 3,038 |
| memory_vector | 155 | 向量检索记忆后端；标签是事实答案 | [JSON](../../data/eval-inspection/bfcl-v4/memory_vector.json) · [Prompt](../../data/eval-inspection/bfcl-v4/memory_vector.prompt.txt) | 12 / 2,576 |
| memory_rec_sum | 155 | 递归摘要记忆后端；标签是事实答案 | [JSON](../../data/eval-inspection/bfcl-v4/memory_rec_sum.json) · [Prompt](../../data/eval-inspection/bfcl-v4/memory_rec_sum.prompt.txt) | 5 / 1,164 |
| web_search_base | 100 | 搜索工具提供摘要；评分查最终参考事实 | [JSON](../../data/eval-inspection/bfcl-v4/web_search_base.json) · [Prompt](../../data/eval-inspection/bfcl-v4/web_search_base.prompt.txt) | 2 / 1,092 |
| web_search_no_snippet | 100 | 相同题源不提供摘要；运行时工具回包不同 | [JSON](../../data/eval-inspection/bfcl-v4/web_search_no_snippet.json) · [Prompt](../../data/eval-inspection/bfcl-v4/web_search_no_snippet.prompt.txt) | 2 / 1,092 |
| format_sensitivity | 5,200 | 组合提示风格和调用表达；不属于 Native FC 计分 | [JSON](../../data/eval-inspection/bfcl-v4/format_sensitivity.json) · [Prompt](../../data/eval-inspection/bfcl-v4/format_sensitivity.prompt.txt) | 0 / 342 |

### API-Bank

固定参考历史下的逐步检索、调用与自然语言回复。版本：f30ccf22b4e2617fab32958d4c03f5c1f2e7dfcf。

| 子类 | 总数 | 测什么 / Label | 样例 | 工具数 / Prompt tokens |
|---|---:|---|---|---:|
| l1-call | 389 | 工具描述直接可见；工具名、参数和参考执行结果 | [JSON](../../data/eval-inspection/api-bank/l1-call.json) · [Prompt](../../data/eval-inspection/api-bank/l1-call.prompt.txt) | 3 / 752 |
| l1-response | 375 | 参考调用已执行；生成下一段自然语言回复 | [JSON](../../data/eval-inspection/api-bank/l1-response.json) · [Prompt](../../data/eval-inspection/api-bank/l1-response.prompt.txt) | 3 / 841 |
| l2-call | 119 | 从 ToolSearcher 开始；按历史检索结果增补 tools | [JSON](../../data/eval-inspection/api-bank/l2-call.json) · [Prompt](../../data/eval-inspection/api-bank/l2-call.prompt.txt) | 3 / 1,024 |
| l2-response | 103 | 使用参考检索/调用历史生成自然语言 | [JSON](../../data/eval-inspection/api-bank/l2-response.json) · [Prompt](../../data/eval-inspection/api-bank/l2-response.prompt.txt) | 3 / 930 |
| l3-call | 245 | 复杂多 API 链路；用参考历史预测下一次调用 | [JSON](../../data/eval-inspection/api-bank/l3-call.json) · [Prompt](../../data/eval-inspection/api-bank/l3-call.prompt.txt) | 2 / 567 |
| l3-response | 50 | 根据链式调用的参考历史生成回复 | [JSON](../../data/eval-inspection/api-bank/l3-response.json) · [Prompt](../../data/eval-inspection/api-bank/l3-response.prompt.txt) | 3 / 1,535 |

### τ²-Bench

模型与用户模拟器交互，按任务验收业务状态与结果。版本：v0.2.0 / f8de30c298689cbe0117d76a378e7315a17e5bd8。

| 子类 | 总数 | 测什么 / Label | 样例 | 工具数 / Prompt tokens |
|---|---:|---|---|---:|
| retail | 114 | 零售订单、退换货与身份核验；任务约束和结果 | [JSON](../../data/eval-inspection/tau2-bench/retail.json) · [Prompt](../../data/eval-inspection/tau2-bench/retail.prompt.txt) | 15 / 5,046 |
| airline | 50 | 航班预订、变更或取消；任务约束和结果 | [JSON](../../data/eval-inspection/tau2-bench/airline.json) · [Prompt](../../data/eval-inspection/tau2-bench/airline.prompt.txt) | 14 / 5,063 |
| telecom | 114 | 通信故障诊断与恢复；用户侧与 Agent 侧工具、状态验收 | [JSON](../../data/eval-inspection/tau2-bench/telecom.json) · [Prompt](../../data/eval-inspection/tau2-bench/telecom.prompt.txt) | 13 / 7,532 |

## 阅读与使用边界

- **不是每道题都有参考回复**：BFCL 单轮大多标函数及可接受参数；相关性只判断有无调用；多轮是执行轨迹/状态；搜索和记忆是答案事实。API-Bank 调用与回复使用不同标签；τ² 用 evaluation_criteria。
- **完整首请求 ≠ 整段运行实录**：JSON 分开保存 case、request、label、prediction。原生 FC 的完整输入是 messages 加全部 tools；随附 .prompt.txt 为本地 Qwen3.5-2B 离线渲染，不宣称与历史服务逐 token 一致。历史 tool arguments 是字符串，离线渲染前转对象。
- **多轮后续由环境和模型决定**：已有日志直接保留；其余缺失不伪造。BFCL 的 Web Search 没有本地输出；两个 Web Search 变体的区别体现在工具回包。Memory 使用归档里的首轮记忆提示。
- **τ² 区分 Agent 和用户模拟器**：任务说明只给用户模拟器，另存 simulator_first_request。Retail 首轮基于既有 smoke；Airline、Telecom 的 Agent 开场为人工示例，没有模拟输出。
- **没有调用不总是拒调**：没有提供工具、提供的工具不适用、信息不足需要追问，应该分别建桶。Glaive 没有调用标记的记录不是天然的拒调标签。
- **仅归档，不等于可直接训练**：SmolTalk 的 all 与独立配置不能重复拼接；通用保持不把 apigen-80k 当普通对话。APIGen-MT 与 τ² Retail/Airline 为同域，训练前仍须做评测重合检查。
- **镜像一致性与许可**：ToolACE 为 mlabonne 镜像，未与作者源逐条比对；Magpie 的 ModelScope 元数据与随附数据卡许可冲突，建议暂不纳入计划发布的训练配方。

## 源码与复现

- [BFCL V4 官方口径](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)
- [API-Bank 上游](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
- [τ²-Bench v0.2.0](https://github.com/sierra-research/tau2-bench/tree/v0.2.0)
- [目录与复现](../../docs/04-数据归档与样本图鉴.md)
