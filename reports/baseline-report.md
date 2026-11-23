# ToolCall-SFT 基线评测报告（当前本地已测汇总）

> 生成时间：2026-11-23 12:14 UTC · 纳入运行数：4

## 1. 评测范围与主分总览

| 模型 | 评测集 | 范围 (Profile) | 状态 | 指标口径 | 正确/总数 | 主分 | 上下文 / 上限 / 并发 | Run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-2B` | `api-bank` | `full` | `completed` | Call Accuracy | 499 / 753 | **66.27%** | 32768 / 1024 / 96 | `20261123-104525-867552-api-bank-full` |
| `Qwen/Qwen3.5-2B` | `bfcl-v4` | `historical` | `historical` | No-Web Weighted (Hist.) | 2480 / 4906 | **25.58%** | N/A / N/A / N/A | `history-bfcl-v4-qwen35-2b-no-web-1024` |
| `Qwen/Qwen3.5-2B` | `bfcl-v4` | `smoke` | `completed` | Microaccuracy | 1 / 3 | **33.33%** | 32768 / 1024 / 96 | `20261123-104422-628311-bfcl-v4-smoke` |
| `Qwen/Qwen3.5-2B` | `tau2` | `smoke` | `completed` | Domain Mean Pass^1 | 0 / 2 | **0.00%** | 32768 / 1024 / 1 | `20261123-104431-366707-tau2-smoke` |

## 2. 时间、吞吐与延迟指标

| 模型 | 评测集 | Wall (s) | 生成/模拟 (s) | 评分 (s) | 输入 Tokens | 输出 Tokens | 输出 tok/s | 总 tok/s | P95 延迟 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-2B` | `api-bank` (`full`) | 116.0 | 98.8 | 13.3 | 1,053,527 | 67,590 | 684.1 | 11,346.8 | 16.89 |
| `Qwen/Qwen3.5-2B` | `bfcl-v4` (`historical`) | 683.4 | 673.9 | N/A | 5,988,540 | 552,682 | 820.2 | 9,707.2 | 19.66 |
| `Qwen/Qwen3.5-2B` | `bfcl-v4` (`smoke`) | 7.2 | 4.2 | 3.0 | 1,074 | 339 | 81.2 | 338.3 | 1.61 |
| `Qwen/Qwen3.5-2B` | `tau2` (`smoke`) | 52.7 | 50.4 | N/A | 125,904 | 2,813 | 55.8 | 2,554.7 | 4.56 |

## 3. 显卡与 vLLM 服务压力

| 模型 | 评测集 | 初始显存 (MiB) | 峰值显存 (MiB) | GPU 利用率峰值 | KV Cache 峰值 | 运行/排队峰值 | 抢占次数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-2B` | `api-bank` (`full`) | 15,374 | 15,471 | 100% | 57.86% | 96 / 55 | 0 |
| `Qwen/Qwen3.5-2B` | `bfcl-v4` (`historical`) | 14,866 | 15,201 | 98.9% | 74.52% | 96 / 42 | N/A |
| `Qwen/Qwen3.5-2B` | `bfcl-v4` (`smoke`) | 15,341 | 15,383 | 100% | 0.87% | 2 / 0 | 0 |
| `Qwen/Qwen3.5-2B` | `tau2` (`smoke`) | 15,325 | 15,570 | 100% | 2.51% | 1 / 0 | 0 |

## 4. 分项得分明细

### Qwen/Qwen3.5-2B · `api-bank` (`full` · `20261123-104525-867552-api-bank-full`)

| 子类 / 域 | 正确 / 总数 | 准确率 | 补充指标 |
| --- | --- | --- | --- |
| `L1` | 308 / 389 | 79.18% | reference_tool_coverage=100.00%, oracle_accuracy=86.89%, rouge_l_f1=0.3845 |
| `L2` | 57 / 119 | 47.90% | reference_tool_coverage=89.08%, oracle_accuracy=65.55%, rouge_l_f1=0.2948 |
| `L3` | 134 / 245 | 54.69% | reference_tool_coverage=100.00%, oracle_accuracy=97.96%, rouge_l_f1=0.3759 |

### Qwen/Qwen3.5-2B · `bfcl-v4` (`historical` · `history-bfcl-v4-qwen35-2b-no-web-1024`)

| 子类 / 域 | 正确 / 总数 | 准确率 | 补充指标 |
| --- | --- | --- | --- |
| `memory_kv` | 0 / 155 | 0.00% | — |
| `memory_rec_sum` | 0 / 155 | 0.00% | — |
| `memory_vector` | 0 / 155 | 0.00% | — |
| `live_irrelevance` | 609 / 884 | 68.89% | — |
| `live_multiple` | 760 / 1053 | 72.17% | — |
| `live_parallel_multiple` | 11 / 24 | 45.83% | — |
| `live_parallel` | 10 / 16 | 62.50% | — |
| `live_relevance` | 14 / 16 | 87.50% | — |
| `live_simple` | 169 / 258 | 65.50% | — |
| `multi_turn_base` | 0 / 200 | 0.00% | — |
| `multi_turn_long_context` | 0 / 200 | 0.00% | — |
| `multi_turn_miss_func` | 0 / 200 | 0.00% | — |
| `multi_turn_miss_param` | 0 / 200 | 0.00% | — |
| `irrelevance` | 189 / 240 | 78.75% | — |
| `multiple` | 164 / 200 | 82.00% | — |
| `parallel_multiple` | 141 / 200 | 70.50% | — |
| `parallel` | 102 / 200 | 51.00% | — |
| `simple_java` | 15 / 100 | 15.00% | — |
| `simple_javascript` | 15 / 50 | 30.00% | — |
| `simple_python` | 281 / 400 | 70.25% | — |

### Qwen/Qwen3.5-2B · `bfcl-v4` (`smoke` · `20261123-104422-628311-bfcl-v4-smoke`)

| 子类 / 域 | 正确 / 总数 | 准确率 | 补充指标 |
| --- | --- | --- | --- |
| `simple_python` | 1 / 3 | 33.33% | — |

### Qwen/Qwen3.5-2B · `tau2` (`smoke` · `20261123-104431-366707-tau2-smoke`)

| 子类 / 域 | 正确 / 总数 | 准确率 | 补充指标 |
| --- | --- | --- | --- |
| `retail` | 0 / 2 | 0.00% | average_reward=0.0000 |

## 5. 错误归因与典型错题（系统错误 vs 真实 LLM 错误）

### Qwen/Qwen3.5-2B · `api-bank` (`full`)

- **系统 / 基础设施错误**：`length_stops`: 1, `execution_error:KeyError`: 22, `tool_exception`: 7
- **真实 LLM 能力错误**：`execution_mismatch`: 113, `response_leaked_tool_call`: 106, `wrong_tool`: 47, `missing_tool_call`: 28, `tool_exception`: 20, `execution_error:KeyError`: 10, `fixture_argument_mismatch`: 4, `tool_not_offered`: 2

| 归因分类 | Case ID | 子类 | 错误原因 | 模型输出摘要 | 期望 / 参考 |
| --- | --- | --- | --- | --- | --- |
| LLM 能力 | `L1:AddAgenda-AddAlarm-GetUserToken-level-2-1:response:6` | `L1:response` | `response_leaked_tool_call (tool_choice=none)` | `<tool_call> <function=AddAgenda> <parameter=token> o8i7u6y5t4r3e2w1q0 </parameter> <parameter=content> Meetin…` | `Natural language response without <tool_call>` |
| LLM 能力 | `L1:AddAgenda-AddAlarm-GetUserToken-level-2-1:call:8` | `L1:call` | `execution_mismatch` | `[{"id":"chatcmpl-tool-94c4ff286d3727f7","type":"function","function":{"name":"AddAgenda","arguments":"{\"toke…` | `api=AddAgenda` |
| LLM 能力 | `L1:AddAgenda-AddMeeting-GetUserToken-level-2-2:call:2` | `L1:call` | `tool_exception` | `[{"id":"chatcmpl-tool-bf4185877674b578","type":"function","function":{"name":"GetUserToken","arguments":"{\"u…` | `api=GetUserToken` |
| LLM 能力 | `L1:AddAgenda-AddMeeting-GetUserToken-level-2-3:call:3` | `L1:call` | `wrong_tool` | `[{"id":"chatcmpl-tool-97b59ce9dec34093","type":"function","function":{"name":"AddMeeting","arguments":"{\"tok…` | `api=GetUserToken` |
| 系统/服务 | `L1:AddAgenda-AddMeeting-GetUserToken-level-2-3:call:4` | `L1:call` | `execution_error:KeyError (oracle_correct=False)` | `[{"id":"chatcmpl-tool-a82e6b97712727e6","type":"function","function":{"name":"AddMeeting","arguments":"{\"tok…` | `oracle_reason=execution_mismatch` |
| LLM 能力 | `L1:ModifyPassword-RegisterUser-GetUserToken-level-2-1:call:2` | `L1:call` | `missing_tool_call` | `I will need your username and password to get your token first.` | `api=GetUserToken` |

### Qwen/Qwen3.5-2B · `bfcl-v4` (`historical`)

- **系统 / 基础设施错误**：`request_errors`: 1010, `context_rejections`: 905, `length_stops`: 53
- **真实 LLM 能力错误**：`category_incorrect:live_multiple`: 293, `category_incorrect:live_irrelevance`: 275, `category_incorrect:multi_turn_base`: 200, `category_incorrect:multi_turn_long_context`: 200, `category_incorrect:multi_turn_miss_func`: 200, `category_incorrect:multi_turn_miss_param`: 200, `category_incorrect:memory_kv`: 155, `category_incorrect:memory_rec_sum`: 155

### Qwen/Qwen3.5-2B · `bfcl-v4` (`smoke`)

- **系统 / 基础设施错误**：无
- **真实 LLM 能力错误**：`simple_function_checker:wrong_count`: 2

| 归因分类 | Case ID | 子类 | 错误原因 | 模型输出摘要 | 期望 / 参考 |
| --- | --- | --- | --- | --- | --- |
| LLM 能力 | `simple_python_0` | `simple_python` | `simple_function_checker:wrong_count` | `(empty tool_calls)` | `[{"calculate_triangle_area":{"base":[10],"height":[5],"unit":["units",""]}}]` |

### Qwen/Qwen3.5-2B · `tau2` (`smoke`)

- **系统 / 基础设施错误**：无
- **真实 LLM 能力错误**：`retail:max_steps`: 2

| 归因分类 | Case ID | 子类 | 错误原因 | 模型输出摘要 | 期望 / 参考 |
| --- | --- | --- | --- | --- | --- |
| LLM 能力 | `retail:task_0:trial_0` | `retail` | `termination=max_steps, reward=0.00` | `I understand you want to proceed with an exchange, but I need to clarify that exchanges can only be done for …` | `{"note":"Simulation terminated prematurely. Termination reason: TerminationReason.MAX_STE…` |

