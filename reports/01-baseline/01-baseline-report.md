# ToolCall-SFT 基线评测报告

> 生成时间：2026-11-24 07:37 UTC · 纳入运行数：4

## 1. 评测范围与主分总览

| 模型 | 评测集 | 范围 (Profile) | 状态 | 指标口径 | 正确/总数 | 主分 | 上下文 / 上限 / 并发 | Run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-2B` | `bfcl-v4` | `core` | `completed_with_request_errors` | Core Weighted (No-Agentic) | 2632 / 4441 | **43.48%** | 65536 / 1024 / 128 | `core-baseline-20261123T144655.947343Z` |
| `Qwen/Qwen3.5-2B` | `api-bank` | `full` | `completed` | Call Accuracy | 501 / 753 | **66.53%** | 65536 / 1024 / 96 | `full-baseline-20261123T150622.824657Z` |
| `Qwen/Qwen3.5-0.8B` | `bfcl-v4` | `core` | `completed_with_request_errors` | Core Weighted (No-Agentic) | 2321 / 4441 | **34.30%** | 65536 / 1024 / 128 | `core-baseline-20261123T150821.031121Z` |
| `Qwen/Qwen3.5-0.8B` | `api-bank` | `full` | `completed` | Call Accuracy | 431 / 753 | **57.24%** | 65536 / 1024 / 96 | `full-baseline-20261123T152129.962853Z` |

## 2. 时间、吞吐与延迟指标

| 模型 | 评测集 | Wall (s) | 生成/模拟 (s) | 评分 (s) | 输入 Tokens | 输出 Tokens | 输出 tok/s | 总 tok/s | P95 延迟 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-2B` | `bfcl-v4` (`core`) | 1,115.3 | 1,110.7 | 4.4 | 98,526,952 | 1,310,854 | 1,180.2 | 89,886.9 | 29.17 |
| `Qwen/Qwen3.5-2B` | `api-bank` (`full`) | 50.8 | 35.5 | 12.1 | 1,053,527 | 67,759 | 1,910.7 | 31,617.8 | 5.89 |
| `Qwen/Qwen3.5-0.8B` | `bfcl-v4` (`core`) | 771.9 | 767.6 | 4.1 | 109,269,606 | 1,356,739 | 1,767.5 | 144,116.2 | 15.32 |
| `Qwen/Qwen3.5-0.8B` | `api-bank` (`full`) | 42.8 | 26.8 | 12.7 | 1,053,527 | 73,066 | 2,730.4 | 42,099.6 | 4.54 |

## 3. 显卡与 vLLM 服务压力

| 模型 | 评测集 | 初始显存 (MiB) | 峰值显存 (MiB) | GPU 利用率峰值 | KV Cache 峰值 | 运行/排队峰值 | 抢占次数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-2B` | `bfcl-v4` (`core`) | 34,723 | 38,355 | 100% | 78.23% | 128 / 43 | 0 |
| `Qwen/Qwen3.5-2B` | `api-bank` (`full`) | 38,355 | 38,355 | 100% | 11.78% | 96 / 18 | 0 |
| `Qwen/Qwen3.5-0.8B` | `bfcl-v4` (`core`) | 35,263 | 38,387 | 100% | 67.05% | 128 / 41 | 0 |
| `Qwen/Qwen3.5-0.8B` | `api-bank` (`full`) | 38,387 | 38,387 | 99% | 10.09% | 95 / 14 | 0 |

## 4. 分项得分明细

### Qwen/Qwen3.5-2B · `bfcl-v4` (`core` · `core-baseline-20261123T144655.947343Z`)

| 子类 / 域 | 正确 / 总数 | 准确率 | 补充指标 |
| --- | --- | --- | --- |
| `simple_python` | 282 / 400 | 70.50% | — |
| `simple_java` | 14 / 100 | 14.00% | — |
| `simple_javascript` | 14 / 50 | 28.00% | — |
| `multiple` | 165 / 200 | 82.50% | — |
| `parallel` | 101 / 200 | 50.50% | — |
| `parallel_multiple` | 142 / 200 | 71.00% | — |
| `irrelevance` | 189 / 240 | 78.75% | — |
| `live_simple` | 170 / 258 | 65.89% | — |
| `live_multiple` | 763 / 1053 | 72.46% | — |
| `live_parallel` | 10 / 16 | 62.50% | — |
| `live_parallel_multiple` | 11 / 24 | 45.83% | — |
| `live_irrelevance` | 607 / 884 | 68.67% | — |
| `live_relevance` | 14 / 16 | 87.50% | — |
| `multi_turn_base` | 59 / 200 | 29.50% | — |
| `multi_turn_miss_func` | 5 / 200 | 2.50% | — |
| `multi_turn_miss_param` | 30 / 200 | 15.00% | — |
| `multi_turn_long_context` | 56 / 200 | 28.00% | — |

### Qwen/Qwen3.5-2B · `api-bank` (`full` · `full-baseline-20261123T150622.824657Z`)

| 子类 / 域 | 正确 / 总数 | 准确率 | 补充指标 |
| --- | --- | --- | --- |
| `L1` | 308 / 389 | 79.18% | reference_tool_coverage=100.00%, oracle_accuracy=86.89%, rouge_l_f1=0.3850 |
| `L2` | 57 / 119 | 47.90% | reference_tool_coverage=89.08%, oracle_accuracy=65.55%, rouge_l_f1=0.2924 |
| `L3` | 136 / 245 | 55.51% | reference_tool_coverage=100.00%, oracle_accuracy=97.96%, rouge_l_f1=0.3753 |

### Qwen/Qwen3.5-0.8B · `bfcl-v4` (`core` · `core-baseline-20261123T150821.031121Z`)

| 子类 / 域 | 正确 / 总数 | 准确率 | 补充指标 |
| --- | --- | --- | --- |
| `simple_python` | 270 / 400 | 67.50% | — |
| `simple_java` | 29 / 100 | 29.00% | — |
| `simple_javascript` | 11 / 50 | 22.00% | — |
| `multiple` | 127 / 200 | 63.50% | — |
| `parallel` | 63 / 200 | 31.50% | — |
| `parallel_multiple` | 105 / 200 | 52.50% | — |
| `irrelevance` | 196 / 240 | 81.67% | — |
| `live_simple` | 137 / 258 | 53.10% | — |
| `live_multiple` | 608 / 1053 | 57.74% | — |
| `live_parallel` | 10 / 16 | 62.50% | — |
| `live_parallel_multiple` | 10 / 24 | 41.67% | — |
| `live_irrelevance` | 684 / 884 | 77.38% | — |
| `live_relevance` | 10 / 16 | 62.50% | — |
| `multi_turn_base` | 21 / 200 | 10.50% | — |
| `multi_turn_miss_func` | 4 / 200 | 2.00% | — |
| `multi_turn_miss_param` | 16 / 200 | 8.00% | — |
| `multi_turn_long_context` | 20 / 200 | 10.00% | — |

### Qwen/Qwen3.5-0.8B · `api-bank` (`full` · `full-baseline-20261123T152129.962853Z`)

| 子类 / 域 | 正确 / 总数 | 准确率 | 补充指标 |
| --- | --- | --- | --- |
| `L1` | 262 / 389 | 67.35% | reference_tool_coverage=100.00%, oracle_accuracy=86.89%, rouge_l_f1=0.3759 |
| `L2` | 52 / 119 | 43.70% | reference_tool_coverage=89.08%, oracle_accuracy=65.55%, rouge_l_f1=0.3391 |
| `L3` | 117 / 245 | 47.76% | reference_tool_coverage=100.00%, oracle_accuracy=97.96%, rouge_l_f1=0.3051 |

## 5. 错误归因与典型错题（系统错误 vs 真实 LLM 错误）

### Qwen/Qwen3.5-2B · `bfcl-v4` (`core`)

- **系统 / 基础设施错误**：`request_errors`: 15, `context_rejections`: 12, `length_stops`: 85, `multi_turn:inference_error`: 15
- **真实 LLM 能力错误**：`irrelevance_error:decoder_success`: 328, `multi_turn:instance_state_mismatch`: 283, `value_error:string`: 213, `simple_function_checker:wrong_count`: 190, `parallel_function_checker_no_order:wrong_count`: 145, `multi_turn:force_terminated`: 133, `multi_turn:empty_turn_model_response`: 127, `multi_turn:execution_response_mismatch`: 92

| 归因分类 | Case ID | 子类 | 错误原因 | 模型输出摘要 | 期望 / 参考 |
| --- | --- | --- | --- | --- | --- |
| 系统/服务 | `irrelevance_24` | `request` | `length_stop` | `output_tokens=1024 (hit cap)` | `finish_reason in {tool_calls, stop}` |
| 系统/服务 | `irrelevance_43` | `request` | `length_stop` | `output_tokens=1024 (hit cap)` | `finish_reason in {tool_calls, stop}` |
| 系统/服务 | `irrelevance_72` | `request` | `length_stop` | `output_tokens=1024 (hit cap)` | `finish_reason in {tool_calls, stop}` |
| LLM 能力 | `live_irrelevance_0-0-0` | `live_irrelevance` | `irrelevance_error:decoder_success` | `[{"requests_get":"{\"url\": \"https://date.nager.at/Api\", \"params\": {\"latitude\": 37.4224764, \"longitude…` | `` |
| LLM 能力 | `live_multiple_0-0-0` | `live_multiple` | `value_error:dict_key` | `[{"ChaDri_change_drink":"{\"drink_id\": \"latte\", \"new_preferences\": {\"size\": \"large\", \"milk_type\": …` | `[{"ChaDri.change_drink":{"drink_id":["latte"],"new_preferences":[{"size":["large"],"milk_…` |
| LLM 能力 | `live_multiple_4-2-1` | `live_multiple` | `value_error:string` | `[{"uber_ride":"{\"loc\": \"123 Hanoi Street, Hanoi\", \"type\": \"plus\", \"time\": 10}"}]` | `[{"uber.ride":{"loc":["123 Hanoi Street"],"type":["plus"],"time":[10]}}]` |
| LLM 能力 | `live_multiple_9-4-1` | `live_multiple` | `multiple_function_checker:wrong_count` | `(empty tool_calls)` | `[{"HNA_WQA.search":{"keyword":["Imjin war"],"result_format":["","text"],"language":["","E…` |
| LLM 能力 | `multi_turn_base_0` | `multi_turn_base` | `multi_turn:instance_state_mismatch` | `Model instance for GorillaFileSystem does not match the state with ground truth instance. \| steps=[[[{"cd":"{…` | `[["cd(folder='document')","mkdir(dir_name='temp')","mv(source='final_report.pdf', destina…` |
| LLM 能力 | `multi_turn_base_1` | `multi_turn_base` | `multi_turn:execution_response_mismatch` | `Model response execution results so far does not contain all the ground truth response execution results for …` | `[["ls(a=True)"],["cd(folder='workspace')","mv(source='log.txt',destination='archive')"],[…` |
| LLM 能力 | `multi_turn_base_2` | `multi_turn_base` | `multi_turn:force_terminated` | `['Model was force-terminated during inference phase. The length of the model result turns (3) does not match …` | `[["cd(folder='documents')","touch(file_name='TeamNotes.txt')"],["echo(content='Collaborat…` |
| LLM 能力 | `multi_turn_base_39` | `multi_turn_base` | `multi_turn:empty_turn_model_response` | `Model response list is empty for turn 2 \| steps=[[[{"mkdir":"{\"dir_name\": \"WebDevProjects\"}"}],[]],[[{"cd…` | `[["mkdir(dir_name='WebDevProjects')"],["cd(folder='WebDevProjects')","touch(file_name='st…` |

### Qwen/Qwen3.5-2B · `api-bank` (`full`)

- **系统 / 基础设施错误**：`length_stops`: 1, `execution_error:KeyError`: 22, `tool_exception`: 7
- **真实 LLM 能力错误**：`execution_mismatch`: 109, `response_leaked_tool_call`: 106, `wrong_tool`: 49, `missing_tool_call`: 28, `tool_exception`: 20, `execution_error:KeyError`: 10, `fixture_argument_mismatch`: 4, `tool_not_offered`: 2

| 归因分类 | Case ID | 子类 | 错误原因 | 模型输出摘要 | 期望 / 参考 |
| --- | --- | --- | --- | --- | --- |
| LLM 能力 | `L1:AddAgenda-AddAlarm-GetUserToken-level-2-1:response:6` | `L1:response` | `response_leaked_tool_call (tool_choice=none)` | `<tool_call> <function=AddAgenda> <parameter=token> o8i7u6y5t4r3e2w1q0 </parameter> <parameter=content> Meetin…` | `Natural language response without <tool_call>` |
| LLM 能力 | `L1:AddAgenda-AddAlarm-GetUserToken-level-2-1:call:8` | `L1:call` | `execution_mismatch` | `[{"id":"chatcmpl-tool-90789a64cd63679c","type":"function","function":{"name":"AddAgenda","arguments":"{\"toke…` | `api=AddAgenda` |
| LLM 能力 | `L1:AddAgenda-AddMeeting-GetUserToken-level-2-2:call:2` | `L1:call` | `tool_exception` | `[{"id":"chatcmpl-tool-b088423b69e99a47","type":"function","function":{"name":"GetUserToken","arguments":"{\"u…` | `api=GetUserToken` |
| LLM 能力 | `L1:AddAgenda-AddMeeting-GetUserToken-level-2-3:call:3` | `L1:call` | `wrong_tool` | `[{"id":"chatcmpl-tool-a74cfb9494ea4a4e","type":"function","function":{"name":"AddMeeting","arguments":"{\"tok…` | `api=GetUserToken` |
| 系统/服务 | `L1:AddAgenda-AddMeeting-GetUserToken-level-2-3:call:4` | `L1:call` | `execution_error:KeyError (oracle_correct=False)` | `[{"id":"chatcmpl-tool-a4b14f2156d33a3f","type":"function","function":{"name":"AddMeeting","arguments":"{\"tok…` | `oracle_reason=execution_mismatch` |
| LLM 能力 | `L1:ModifyPassword-RegisterUser-GetUserToken-level-2-1:call:2` | `L1:call` | `missing_tool_call` | `I will need your username and password to get your token first.` | `api=GetUserToken` |

### Qwen/Qwen3.5-0.8B · `bfcl-v4` (`core`)

- **系统 / 基础设施错误**：`request_errors`: 26, `context_rejections`: 25, `length_stops`: 123, `multi_turn:inference_error`: 26
- **真实 LLM 能力错误**：`multi_turn:force_terminated`: 266, `multi_turn:instance_state_mismatch`: 256, `value_error:string`: 250, `irrelevance_error:decoder_success`: 244, `simple_function_checker:wrong_count`: 213, `parallel_function_checker_no_order:wrong_count`: 203, `multiple_function_checker:wrong_count`: 166, `multi_turn:execution_response_mismatch`: 101

| 归因分类 | Case ID | 子类 | 错误原因 | 模型输出摘要 | 期望 / 参考 |
| --- | --- | --- | --- | --- | --- |
| 系统/服务 | `irrelevance_18` | `request` | `length_stop` | `output_tokens=1024 (hit cap)` | `finish_reason in {tool_calls, stop}` |
| 系统/服务 | `irrelevance_72` | `request` | `length_stop` | `output_tokens=1024 (hit cap)` | `finish_reason in {tool_calls, stop}` |
| 系统/服务 | `irrelevance_77` | `request` | `length_stop` | `output_tokens=1024 (hit cap)` | `finish_reason in {tool_calls, stop}` |
| LLM 能力 | `live_irrelevance_15-2-3` | `live_irrelevance` | `irrelevance_error:decoder_success` | `[{"get_current_weather":"{\"location\": \"145.34.45.56, USA\", \"unit\": \"fahrenheit\"}"}]` | `` |
| LLM 能力 | `live_multiple_0-0-0` | `live_multiple` | `value_error:dict_key` | `[{"ChaDri_change_drink":"{\"new_preferences\": {\"size\": \"large\", \"temperature\": \"hot\", \"sweetness_le…` | `[{"ChaDri.change_drink":{"drink_id":["latte"],"new_preferences":[{"size":["large"],"milk_…` |
| LLM 能力 | `live_multiple_4-2-1` | `live_multiple` | `value_error:string` | `[{"uber_ride":"{\"loc\": \"123 Hanoi Street, Hanoi, Vietnam\", \"type\": \"plus\", \"time\": 10}"}]` | `[{"uber.ride":{"loc":["123 Hanoi Street"],"type":["plus"],"time":[10]}}]` |
| LLM 能力 | `live_multiple_9-4-1` | `live_multiple` | `multiple_function_checker:wrong_count` | `(empty tool_calls)` | `[{"HNA_WQA.search":{"keyword":["Imjin war"],"result_format":["","text"],"language":["","E…` |
| LLM 能力 | `multi_turn_base_0` | `multi_turn_base` | `multi_turn:instance_state_mismatch` | `Model instance for GorillaFileSystem does not match the state with ground truth instance. \| steps=[[[{"cd":"{…` | `[["cd(folder='document')","mkdir(dir_name='temp')","mv(source='final_report.pdf', destina…` |
| LLM 能力 | `multi_turn_base_1` | `multi_turn_base` | `multi_turn:force_terminated` | `['Model was force-terminated during inference phase. The length of the model result turns (1) does not match …` | `[["ls(a=True)"],["cd(folder='workspace')","mv(source='log.txt',destination='archive')"],[…` |
| LLM 能力 | `multi_turn_base_14` | `multi_turn_base` | `multi_turn:execution_response_mismatch` | `Model response execution results so far does not contain all the ground truth response execution results for …` | `[["cd(folder='ResearchDocs')","find(path='.',name='report.csv')"],["grep(file_name='repor…` |
| LLM 能力 | `multi_turn_base_72` | `multi_turn_base` | `multi_turn:empty_turn_model_response` | `Model response list is empty for turn 0 \| steps=[[[]],[[{"fillFuelTank":"{\"fuelAmount\": 50}"}],[{"startEngi…` | `[["fillFuelTank(fuelAmount=10.0)"],["lockDoors(unlock=False, door=['driver', 'passenger',…` |

### Qwen/Qwen3.5-0.8B · `api-bank` (`full`)

- **系统 / 基础设施错误**：`length_stops`: 2, `execution_error:KeyError`: 22, `tool_exception`: 7
- **真实 LLM 能力错误**：`execution_mismatch`: 138, `wrong_tool`: 84, `response_leaked_tool_call`: 79, `missing_tool_call`: 30, `tool_exception`: 24, `execution_error:KeyError`: 10, `fixture_argument_mismatch`: 3, `unexpected_multiple_calls`: 2

| 归因分类 | Case ID | 子类 | 错误原因 | 模型输出摘要 | 期望 / 参考 |
| --- | --- | --- | --- | --- | --- |
| LLM 能力 | `L1:AddAgenda-AddAlarm-GetUserToken-level-2-1:call:6` | `L1:call` | `wrong_tool` | `[{"id":"chatcmpl-tool-84605678f79c79b1","type":"function","function":{"name":"AddAgenda","arguments":"{\"toke…` | `api=GetUserToken` |
| LLM 能力 | `L1:AddAgenda-AddAlarm-GetUserToken-level-2-1:response:6` | `L1:response` | `response_leaked_tool_call (tool_choice=none)` | `<tool_call> <function=AddAlarm> <parameter=token> o8i7u6y5t4r3e2w1q0 </parameter> <parameter=time> 2023-09-12…` | `Natural language response without <tool_call>` |
| LLM 能力 | `L1:AddAgenda-AddAlarm-GetUserToken-level-2-1:call:8` | `L1:call` | `execution_mismatch` | `[{"id":"chatcmpl-tool-a54d1960c4f6a3b9","type":"function","function":{"name":"AddAgenda","arguments":"{\"toke…` | `api=AddAgenda` |
| 系统/服务 | `L1:AddAgenda-AddMeeting-GetUserToken-level-2-3:call:4` | `L1:call` | `execution_error:KeyError (oracle_correct=False)` | `[{"id":"chatcmpl-tool-b013a4a9963d50fc","type":"function","function":{"name":"AddMeeting","arguments":"{\"tok…` | `oracle_reason=execution_mismatch` |
| LLM 能力 | `L1:Calculator-QueryHistoryToday-level-2-1:call:6` | `L1:call` | `tool_exception` | `[{"id":"chatcmpl-tool-86605b75476db15c","type":"function","function":{"name":"QueryHistoryToday","arguments":…` | `api=QueryHistoryToday` |
| LLM 能力 | `L1:DeleteAccount-ModifyPassword-GetUserToken-level-2-4:call:6` | `L1:call` | `missing_tool_call` | `I can help you delete your account. Please provide your username and password again.` | `api=DeleteAccount` |

