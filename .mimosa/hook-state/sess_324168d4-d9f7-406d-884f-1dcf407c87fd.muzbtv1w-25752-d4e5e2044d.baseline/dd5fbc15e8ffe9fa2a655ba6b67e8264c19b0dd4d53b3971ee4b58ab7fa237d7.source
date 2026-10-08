"""评测报告: 汇总 JSON + Markdown 表格 + 对比图(matplotlib)。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

CONFIG_CN = {
    "naive_vector": "纯向量检索",
    "hybrid": "混合检索(+BM25+RRF)",
    "naive": "朴素RAG(无路由/单路)",
    "adaptive": "自适应(无校验)",
    "full": "完整系统",
}


def save_report(
    out_dir: Path,
    retrieval_rows: list[dict] | None,
    e2e_rows: list[dict] | None,
    per_item: list[dict] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        "# ReflexRAG 评测报告",
        "",
        f"- 时间: {ts}",
        "- 金标集: data/golden/golden.jsonl(种子题 + LLM 辅助生成, 构建脚本见 scripts/build_golden.py)",
        "- 裁判: LLM-as-judge(0~5 打分), 检索指标为纯代码计算",
        "",
    ]

    if retrieval_rows:
        lines += ["## 一、检索质量(金标条文, k=8)", "",
                  "| 配置 | n | recall@5 | recall@8 | MRR@8 | nDCG@8 | 耗时s |",
                  "|---|---|---|---|---|---|---|"]
        for r in retrieval_rows:
            lines.append(
                f"| {CONFIG_CN.get(r['config'], r['config'])} | {r['n']} | {r['recall@5']} "
                f"| {r.get('recall@8', '-')} | {r['mrr@k']} | {r['ndcg@k']} | {r['elapsed_s']} |"
            )
        lines.append("")

    if e2e_rows:
        lines += [
            "## 二、端到端质量(完整 pipeline + LLM 裁判)",
            "",
            "| 配置 | n | 路由准确率 | 拒答正确率 | 忠实性(0-5) | 正确性(0-5) | 证据召回 | 平均轮次 | p50延迟 | p95延迟 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in e2e_rows:
            lines.append(
                f"| {CONFIG_CN.get(r['config'], r['config'])} | {r['n']} "
                f"| {r['route_accuracy']:.0%} "
                f"| {f'{r["refusal_accuracy"]:.0%}' if r['refusal_accuracy'] is not None else '-'} "
                f"| {r['faithfulness_0_5'] if r['faithfulness_0_5'] is not None else '-'} "
                f"| {r['correctness_0_5'] if r['correctness_0_5'] is not None else '-'} "
                f"| {r['evidence_recall@8'] if r['evidence_recall@8'] is not None else '-'} "
                f"| {r['avg_rounds']} | {r['p50_latency_ms'] / 1000:.1f}s | {r['p95_latency_ms'] / 1000:.1f}s |"
            )
        lines += ["", "### 成本", "",
                  "| 配置 | 平均prompt tokens | 平均completion tokens | 裁判tokens(总) |",
                  "|---|---|---|---|"]
        for r in e2e_rows:
            lines.append(
                f"| {CONFIG_CN.get(r['config'], r['config'])} | {r['avg_prompt_tokens']:.0f} "
                f"| {r['avg_completion_tokens']:.0f} | {r['judge_tokens']} |"
            )
        lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {"retrieval": retrieval_rows, "e2e": e2e_rows, "generated_at": ts},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    if per_item:
        (out_dir / "per_item.json").write_text(
            json.dumps(per_item, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    if retrieval_rows or e2e_rows:
        _render_chart(out_dir / "charts.png", retrieval_rows, e2e_rows)
    return report_path


def _render_chart(out: Path, retrieval_rows: list[dict] | None, e2e_rows: list[dict] | None) -> None:
    panels = []
    if retrieval_rows:
        panels.append(("检索质量(金标条文)", ["recall@5", "recall@8", "mrr@k", "ndcg@k"]))
    if e2e_rows:
        panels.append(("端到端质量", ["route_accuracy", "faithfulness_0_5", "correctness_0_5"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, keys) in zip(axes, panels, strict=True):
        rows = retrieval_rows if title.startswith("检索") else e2e_rows
        width = 0.8 / len(rows)
        x = range(len(keys))
        for j, r in enumerate(rows):
            vals = [
                r.get(k) * 5 if (k in ("faithfulness_0_5", "correctness_0_5")) else (r.get(k) or 0)
                for k in keys
            ]
            ax.bar([xi + j * width for xi in x], vals, width,
                   label=CONFIG_CN.get(r["config"], r["config"]))
        ax.set_xticks([xi + width * (len(rows) - 1) / 2 for xi in x])
        labels = ["recall@5", "recall@8", "MRR@8", "nDCG@8"] if title.startswith("检索") \
            else ["路由准确率", "忠实性(×5归一)", "正确性(×5归一)"]
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
