# 实验与分析报告索引

本目录按研究阶段分类并按两位数序号组织，所有报告均在本地生成或整理后纳入 Git 存档。

## 目录结构

| 目录 | 阶段 / 用途 | 当前文档 |
| --- | --- | --- |
| `00-archive/` | 早期探索与阶段性归档产物 | [01-bfcl-evaluation.html](00-archive/01-bfcl-evaluation.html)、[02-bfcl-budget-comparison.png](00-archive/02-bfcl-budget-comparison.png) |
| `01-baseline/` | 双模型（2B / 0.8B）64K 基线评测报告 | [01-baseline-report.md](01-baseline/01-baseline-report.md)、[01-baseline-report.html](01-baseline/01-baseline-report.html) |
| `02-data-audit/` | 训练数据与评测样本归档图鉴 | [01-sft-data-analysis.md](02-data-audit/01-sft-data-analysis.md)、[01-sft-data-analysis.html](02-data-audit/01-sft-data-analysis.html) |
| `03-sft/` | SFT 阶段实验报告（`E0 / E1 / E2` 等） | [01-e0-smoke-run.md](03-sft/01-e0-smoke-run.md) |
| `04-opd-grpo/` | OPD / Agentic RL（GRPO）阶段报告（预留） | — |

## 生成入口

- 基线报告：`toolcall-baseline-report --auto`（默认输出至 `reports/01-baseline/01-baseline-report.{md,html}`）
- 数据图鉴：`python3 scripts/data-audit/report.py`（默认输出至 `reports/02-data-audit/01-sft-data-analysis.{md,html}`）

回到[项目首页](../README.md) · 查看[科研范式指南](../AGENTS.md)。
