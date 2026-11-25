# 实验与分析报告索引

本目录按研究阶段分类并按两位数序号组织，所有报告均在本地生成或整理后纳入 Git 存档。

## 目录结构

| 目录 | 阶段 / 用途 | 当前文档 |
| --- | --- | --- |
| `00-archive/` | 早期探索与阶段性归档产物 | [01-bfcl-evaluation.html](00-archive/01-bfcl-evaluation.html)、[02-bfcl-budget-comparison.png](00-archive/02-bfcl-budget-comparison.png) |
| `01-baseline/` | 双模型（2B / 0.8B）64K 基线评测报告 | [01-baseline-report.md](01-baseline/01-baseline-report.md)、[01-baseline-report.html](01-baseline/01-baseline-report.html) |
| `02-data-audit/` | 训练数据与评测样本归档图鉴 | [01-sft-data-analysis.md](02-data-audit/01-sft-data-analysis.md)、[02-sft-v1r1-dataset-report.md](02-data-audit/02-sft-v1r1-dataset-report.md)、[03-sft-v1r1-clean-audit.md](02-data-audit/03-sft-v1r1-clean-audit.md)、[04-sft-v2-curated-dataset-report.md](02-data-audit/04-sft-v2-curated-dataset-report.md)、[05-sft-v3-dataset-report.md](02-data-audit/05-sft-v3-dataset-report.md)、[06-sft-v3-bfcl-behavior-alignment.md](02-data-audit/06-sft-v3-bfcl-behavior-alignment.md) |
| `03-sft/` | SFT 阶段实验报告（`v1r1 / v3` 等） | [01-e0-smoke-run.md](03-sft/01-e0-smoke-run.md)、[02-e0-followup.md](03-sft/02-e0-followup.md)、[03-sft-v1r1-clean-2b.md](03-sft/03-sft-v1r1-clean-2b.md)、[04-sft-v3-2b-quick-eval-report.md](03-sft/04-sft-v3-2b-quick-eval-report.md)、[05-sft-v3-2b-ckpt68-core-eval-report.md](03-sft/05-sft-v3-2b-ckpt68-core-eval-report.md)、[06-sft-v3-r16-vs-r64-report.md](03-sft/06-sft-v3-r16-vs-r64-report.md) |
| `04-opd-grpo/` | Stage 2 GRPO / Rollout 挖掘报告 | [01-grpo-single-gpu-preflight.md](04-opd-grpo/01-grpo-single-gpu-preflight.md)、[02-grpo-smoke-2b-report.md](04-opd-grpo/02-grpo-smoke-2b-report.md)、[03-grpo-baseline-rollout-mining-report.md](04-opd-grpo/03-grpo-baseline-rollout-mining-report.md)、[04-grpo-v2-smoke-colocate-report.md](04-opd-grpo/04-grpo-v2-smoke-colocate-report.md)、[05-grpo-v2-full-and-eval-report.md](04-opd-grpo/05-grpo-v2-full-and-eval-report.md)、[06-grpo-four-model-eval.html](04-opd-grpo/06-grpo-four-model-eval.html) |

## 生成入口

- 基线报告：`toolcall-baseline-report --auto`（默认输出至 `reports/01-baseline/01-baseline-report.{md,html}`）
- 数据图鉴：`python3 scripts/data-audit/report.py`（默认输出至 `reports/02-data-audit/01-sft-data-analysis.{md,html}`）

回到[项目首页](../README.md) · 查看[科研范式指南](../AGENTS.md)。
