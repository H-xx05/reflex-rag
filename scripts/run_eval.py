"""运行评测: python scripts/run_eval.py [--e2e-profiles naive,hybrid,full] [--e2e-limits 12,10] [--skip-e2e]

检索基准跑全部带金标条文的题; 端到端基准按分层抽样取子集(控制耗时与成本)。
报告输出到 data/eval_reports/<时间戳>/。
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reflex_rag.config import PROJECT_ROOT, get_settings
from reflex_rag.eval.dataset import GoldenItem, load_golden
from reflex_rag.eval.report import save_report
from reflex_rag.eval.runner import EvalRunner


def pick_e2e_subset(items: list[GoldenItem], n_simple: int, n_multi: int) -> list[GoldenItem]:
    """闲聊/超范围全取(便宜且关键), 单跳/多跳按固定种子抽样。"""
    rng = random.Random(2026)
    by_route: dict[str, list[GoldenItem]] = {}
    for it in items:
        by_route.setdefault(it.expected_route.value, []).append(it)
    simple = by_route.get("simple", [])
    multi = by_route.get("multi_hop", [])
    rng.shuffle(simple)
    rng.shuffle(multi)
    return (
        by_route.get("chitchat", [])
        + by_route.get("out_of_scope", [])
        + simple[:n_simple]
        + multi[:n_multi]
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-profiles", default="naive,hybrid,full")
    parser.add_argument("--e2e-limits", default="12,10",
                        help="单跳,多跳 抽样数; 闲聊与超范围全量")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    settings = get_settings()
    items = load_golden(settings.corpus_dir.parent / "golden" / "golden.jsonl")
    print(f"金标集 {len(items)} 题加载完成")

    runner = EvalRunner(settings)
    retrieval_rows, e2e_rows, per_item = None, None, None

    if not args.skip_retrieval:
        print("== 检索基准(纯代码指标) ==")
        retrieval_rows = await runner.run_retrieval_bench(items)
        for r in retrieval_rows:
            print(r)

    if not args.skip_e2e:
        n_simple, n_multi = (int(x) for x in args.e2e_limits.split(","))
        subset = pick_e2e_subset(items, n_simple, n_multi)
        profiles = [p.strip() for p in args.e2e_profiles.split(",")]
        print(f"== 端到端基准: {len(subset)} 题 × {profiles}, 并发 {args.concurrency} ==")
        t0 = time.perf_counter()
        e2e_rows = await runner.run_e2e(subset, profiles, concurrency=args.concurrency)
        print(f"端到端耗时 {time.perf_counter() - t0:.0f}s, 裁判 tokens: "
              f"{runner.judge_usage.prompt_tokens}+{runner.judge_usage.completion_tokens}")
        per_item = runner.last_e2e_items

    out_dir = PROJECT_ROOT / "data" / "eval_reports" / time.strftime("%Y%m%d-%H%M%S")
    report = save_report(out_dir, retrieval_rows, e2e_rows, per_item)
    print(f"\n报告已写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
