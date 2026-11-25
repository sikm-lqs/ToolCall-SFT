# `SFT-v3` 高质量严格校验数据集构建报告

> 日期：2026-11-24（Asia/Shanghai） · 数据集名称：**`sft_v3`** · 配置：[configs/data/sft_v3.yaml](../../configs/data/sft_v3.yaml) · 产物目录：`data/processed/sft_v3/`

---

## 1. 重构目标与 `sft_v2_curated` 五大缺陷闭环修复

针对 `sft_v2_curated` 审计暴露出的五项实质性数据缺陷，**`sft_v3`** 采取“**宁缺毋滥、硬断言阻断、台账可追溯**”策略完成全面重构（同时保留原 `sft_v1`、`sft_v1r1_clean` 与 `sft_v2_curated` 不覆盖）：

| `sft_v2_curated` 暴露缺陷 | `sft_v3` 代码与数据层修复措施 | 实测剔除 / 修正结果 |
| --- | --- | --- |
| **1. APIGen 正文伪调用泄漏**（8 条） | 在 [parsers.py](../../src/toolcall/data/parsers.py) 新增 `_NL_PSEUDO_CALL_RE`，对任何含 `<tool_call>`、`</tool_call>`、`<function=`、`"id": "call_` 或 `"arguments": {...}` 的纯文本回复整条丢弃，并在后置校验中做硬断言阻断 | APIGen 共淘汰 **`144`** 条 `pseudo_call_in_nl`；终库泄漏 **`0`** |
| **2. ToolACE 拒调 vs 缺参互相错标** | 在 `classify_toolace_no_call(..., strict_boundary=True)` 中要求同时核验「工具不能做该任务（`_STRICT_IRR_RE`）」与「明确点出候选工具名或 `required` 缺失参数（`_STRICT_MP_RE` + 工具/必填参匹配）」，消除套话错标，冲突/模糊样本全部隔离 | ToolACE 隔离冲突 **`79`** 条 + 模糊 **`269`** 条 + 空工具 **`248`** 条 + 泄漏 **`86`** 条；保留纯 `irrelevant_refusal` **`370`** 条与纯 `missing_param` **`486`** 条 |
| **3. SmolTalk 无依据实时幻想与驼峰漏网** | 1) 彻底剔除易虚构日程的 `everyday-conversations`；<br>2) 修复 [smoltalk.py](../../src/toolcall/data/smoltalk.py) `_split_identifier_tokens()` 驼峰分词并扩充通用工具词黑名单（如 `solveMathProblem`、`Get_AI_Answer`、`Article_Summarizer`）；<br>3) 建立 SHA-256 锁定白名单 [sft_v3_smoltalk_review.json](../../configs/data/sft_v3_smoltalk_review.json) | 仅纳入 **270** 条逐条核验且 SHA-256 锁定的干净非闲聊负例（`openhermes` 183 + `smol-constraints` 44 + 限额 `metamathqa` 43） |
| **4. `τ²-Bench` 实体近重合** | 在 `build_sft_v3.py` 增加 `_load_tau2_task_entities()`，提取 `τ²` Retail（114 题）+ Airline（50 题）`tasks.json` 全部订单/用户/邮箱/航司预约号（共 `351` 个实体），命中即淘汰并置硬断言 | APIGen 共剔除 **`932`** 条共享测试实体样本，终库 `tau2_task_entity_overlap = 0` |
| **5. xLAM 参数类型错 & 重复工具名** | 在 [schema.py](../../src/toolcall/data/schema.py) `validate_record()` 增加 `strict_arg_types=True` 与 `reject_duplicate_tools=True`，拒绝 `"math.pi / 2"` 填数值、`"list(range(...))"` 填数组及同名多义工具 | xLAM 淘汰 **`1,214`** 条类型不匹配（`integer: 836`、`string: 154`、`array: 115`、`number: 85`、`boolean: 24`）与 **`260`** 条重复工具名样本 |

---

## 2. 数据清洗漏斗与严格校验机制

核心入口为 [build_sft_v3.py](../../src/toolcall/data/build_sft_v3.py)（CLI 命令 `toolcall-build-data-v3`），所有原始解析统计记录在 [manifest.json](../../data/processed/sft_v3/manifest.json)：

### 2.1 四大数据源清洗明细

| 数据源 | 原始规模 | 严格过滤 / 隔离统计 | 去重 / 长度淘汰 | 净可用池 |
| --- | ---: | --- | ---: | ---: |
| **`toolace`** | 11,300 | 非 JSON 模板 518、空工具 248、模糊边界 269、疑似文本调用泄漏 86、边界冲突 79、空回复 5、多轮文本泄漏 1 | 跨源去重 3 | **10,091**<br>（多轮 791 / 并行 4,065 / 单选 4,379 / 拒调 370 / 缺参 486） |
| **`xlam-60k`** | 60,000 | 缺失 `required` 274、重复工具名 260、**参数类型冲突 1,214**、单选要求 `len(tools) >= 2` | 去重 303 | **50,120**<br>（并行 30,468 / 多候选单选 19,652） |
| **`apigen-mt-5k`** | 5,000 | `τ²` 测试任务实体重合 **932**、含 `think` 调用 **426**、坏调用 174、**正文伪调用 144**、无 NL 收尾 59、连续 gpt 38、无交互 23 | 超 `7500` token 截断门控 **988** | **2,216**<br>（Airline 与 Retail 各 140 train / 20 val 均衡抽样） |
| **`smoltalk_irrelevant`** | 白名单 **270** | 全部通过 [sft_v3_smoltalk_review.json](../../configs/data/sft_v3_smoltalk_review.json) 精确 `(messages, tools)` SHA-256 重建校验（`0` 偏差） | 去重 0 | **270**<br>（`openhermes` 183 / `smol-constraints` 44 / `metamathqa` 43） |

### 2.2 硬阻断不变量与同域边界声明

`build_sft_v3.py` 在落盘前通过 `_assert_final_invariants()` 对全部 **5,030** 条记录（train 4,590 + val 440）执行硬阻断检查（任意 1 条失败即中止构建）：
- **`train_val_group_overlap = 0`**：基于 `normalized_user_turns`、`dedup_signature` 和 `(source, source_id)` 三键硬隔离。
- **`train_historical_val_overlap = 0`**：严格把 [sft_v1r1_clean/val.jsonl](../../data/processed/sft_v1r1_clean/val.jsonl) 的 493 条历史验证集来源全部排除在 `sft_v3/train.jsonl` 之外，保持跨版本验证独立性。
- **`eval_user_turn_overlap = 0` & `tau2_task_entity_overlap = 0`**：与 `bfcl-v4`、`api-bank`、`tau2-bench` 展示样例及 `tau2` Retail/Airline 164 道测试任务实体零重合。
- **全库格式/Schema 强校验**：5,030 条 100% 通过 `validate_record(..., strict_arg_types=True, reject_duplicate_tools=True)`、`think` 工具/调用 = `0`、ToolACE 策略句残留 = `0`、正文伪调用 = `0`。

> [!NOTE]
> **关于 `τ²-Bench` 的使用声明**：虽然 `sft_v3` 已剔除全部 932 条共享 `tasks.json` 测试实体的 APIGen 样本，但 `apigen-mt-5k` 的剩余轨迹仍运行在 `τ²-Bench` Retail/Airline 的同一套领域数据库（`db.json`）与政策文档（Policy）下。因此若后续评测 `τ²-Bench Retail/Airline`，应将其视为**同域沙盒训练效果**，不能作为域外零样本泛化结论（`Telecom` 域除外）。

---

## 3. `sft_v3` 最终产物与能力桶配比

输出目录：`data/processed/sft_v3/`

| 文件 | 条数 | 总 Token（均值 / 最大） | Assistant 监督 Token（均值 / 占比） | SHA-256（前 12 位） |
| --- | ---: | ---: | ---: | --- |
| **`train.jsonl`** | **4,590** | 5,440,445（1,185.3 / 7,495） | 814,614（177.5 / **14.97%**） | `bf44f2d059da` |
| **`val.jsonl`** | **440** | 567,113（1,288.9 / 7,466） | 75,903（172.5 / **13.38%**） | `ff53c395559e` |
| **`audit_ledger.jsonl`** | **5,030** | — | — | `bf002f152659` |
| **`qwen35_chat_template_train.jinja`** | — | — | — | `602a5dbfb1f2` |

### `train.jsonl`（4,590 条）9 大能力桶精确配比与梯度权重

| 能力大类 | 子桶名 | 行数 (占比) | Assistant Token (占比) | 构成与设计意图 |
| --- | --- | ---: | ---: | --- |
| **多轮状态与回包交互**<br>（**980 条 / 21.35%**<br>梯度占比 **61.17%**） | `toolace_multi_turn` | 700 (15.25%) | 324,097 (39.79%) | 无上采样；优先保留含 `>=2` 用户轮且自然收尾的轨迹（323 条含追问/多跳自然收尾 + 246 条单轮多跳收尾 + 131 条多跳接续调用） |
| | `apigen_multi_turn` | 280 (6.10%) | 174,178 (21.38%) | **Airline 140 + Retail 140**；下调行数把政策多轮梯度压至 ~21%（原 V2 36.3%），防单次一调策略压制并行能力 |
| **单轮并行调用**<br>（**1,350 条 / 29.41%**） | `xlam_parallel` | 750 (16.34%) | 68,142 (8.36%) | 严格类型校验通过的 xLAM 并行样本 |
| | `toolace_parallel` | 600 (13.07%) | 96,076 (11.79%) | ToolACE 复杂嵌套 schema 并行调用 |
| **单轮多候选单选**<br>（**1,350 条 / 29.41%**） | `xlam_single` | 700 (15.25%) | 33,328 (4.09%) | 全部 `len(tools) >= 2`、严格类型校验通过 |
| | `toolace_single` | 650 (14.16%) | 37,600 (4.62%) | ToolACE 复杂嵌套 schema 单调用 |
| **工具不适用拒调**<br>（**510 条 / 11.11%**<br>梯度占比 **6.84%**） | `toolace_irrelevant_refusal` | 300 (6.54%) | 13,896 (1.71%) | 双向规则清洗后的外部 API 拒答（Type 2） |
| | `smoltalk_irrelevant_tools` | 210 (4.58%) | 41,808 (5.13%) | 白名单锁定：`openhermes` **148** + `smol-constraints` **34** + 限额 `metamathqa` **28**（Type 1 直接作答，梯度高于 Clean Smoke 的 3.17%） |
| **缺参追问** | `toolace_missing_param` | 400 (8.71%) | 25,489 (3.13%) | 确认提及候选工具名或 `required` 参数的精准追问 |
| **总计** | **9 个子桶** | **4,590 (100%)** | **814,614 (100%)** | 正例 80.17% / 非调用边界 19.83%（含 1,408 条并行、3,692 条多候选） |

---

## 4. 校验、台账与渲染样例

1. **全量单元测试**：`runs/tools/bfcl/bin/python -m unittest discover -s tests -p "test_sft_v*.py" -v` **18/18 通过**（含新增 [test_sft_v3_data.py](../../tests/test_sft_v3_data.py) 验证 `strict_arg_types`、`duplicate_tool_name`、ToolACE `00275/00074` 纠偏、APIGen 伪调用/`τ²` 实体拦截与驼峰词拦截）。
2. **Manifest SHA-256 核验**：`toolcall.training.preflight.verify_data_manifest` 核验 `train.jsonl`、`val.jsonl`、`audit_ledger.jsonl`、`qwen35_chat_template_train.jinja` 全部通过。
3. **可追溯审计台账**：[audit_ledger.jsonl](../../data/processed/sft_v3/audit_ledger.jsonl) 记录了全量 5,030 条样本的 `sample_id`、`source`、`source_id`、`curated_origin`、`tool_names` 及 `content_sha256`。
4. **渲染样例抽检**（[samples/](../../data/processed/sft_v3/samples/)）：
   - [apigen_multi_turn.txt](../../data/processed/sft_v3/samples/apigen_multi_turn.txt)（5,286 tokens / 监督 603）：13 个航司工具（无 `think`），多轮询问 ID → 调 `get_reservation_details` → 读回包并按政策回复。
   - [smoltalk_irrelevant_tools.txt](../../data/processed/sft_v3/samples/smoltalk_irrelevant_tools.txt)（649 tokens / 监督 184）：挂载无关工具 `createTelecomApp`，正常回答 Python 正则表达式问题。
   - [toolace_irrelevant_refusal.txt](../../data/processed/sft_v3/samples/toolace_irrelevant_refusal.txt)（436 tokens / 监督 44）：挂载医疗图像读取工具 `getImage`，明确拒绝做肿瘤/骨密度分析。
   - [toolace_missing_param.txt](../../data/processed/sft_v3/samples/toolace_missing_param.txt)（545 tokens / 监督 44）：挂载 `streamer.get_schedule`，精准追问缺失的必填参数 `streamer_id`。

---

## 5. 上云训练参考预算

- `train.jsonl` 总计 `5.44M` tokens，按 8K `BFD packing` 约打包为 **~665 条**序列；
- 按 `bs=1, accum=8`（有效 8 序列/步），**1 epoch ≈ 83 步、2 epochs ≈ 166 步**；
- A100-40GB 上开启当前配置约 **~31 分钟**完成 2 epochs（若将 `gradient_checkpointing` 设为 `false`，预计仅需 **~24 分钟**），每 `0.25`（~42 步）留存 checkpoint 用于 BFCL 9 类 + API-Bank 对比验收。
