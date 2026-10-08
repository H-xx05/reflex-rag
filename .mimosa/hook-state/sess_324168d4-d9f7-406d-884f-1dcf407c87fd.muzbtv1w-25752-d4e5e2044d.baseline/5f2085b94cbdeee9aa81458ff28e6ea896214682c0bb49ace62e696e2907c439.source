"""命令行问答: python scripts/ask.py "试用期最长多久" [--trace] [--json]

不带参数进入交互模式。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reflex_rag.runtime import build_runtime


async def ask_once(question: str, *, show_trace: bool, as_json: bool) -> None:
    runtime = build_runtime()

    def on_delta(text: str) -> None:
        print(text, end="", flush=True)

    async def async_delta(text: str) -> None:
        on_delta(text)

    result = await runtime.pipeline.answer(question, on_delta=async_delta)
    print("\n")
    if as_json:
        print(result.model_dump_json(indent=2))
        return
    print("─" * 60)
    print(f"路由: {result.route.value} | 轮次: {result.rounds} | 用时: {result.total_latency_ms}ms")
    print(f"tokens: {result.usage.prompt_tokens}+{result.usage.completion_tokens} | LLM调用: {result.usage.llm_calls}")
    if result.citations:
        print("引用:")
        for c in result.citations:
            print(f"  [{c.n}] 《{c.title}》{c.chapter} {'、'.join(c.articles)}")
    if result.verification and result.verification.verdict != "pass":
        print(f"⚠ 校验警告: {result.verification.model_dump()}")
    if show_trace:
        print("\nTrace:")
        for ev in result.trace:
            print(f"  {ev.step:<12} {ev.latency_ms:>6}ms  {json.dumps(ev.data, ensure_ascii=False)[:110]}")


async def repl(*, show_trace: bool) -> None:
    runtime = build_runtime()
    print(f"ReflexRAG 就绪。语料: {len(runtime.corpus_titles)} 部法规。输入问题, exit 退出。\n")
    while True:
        try:
            q = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        async def async_delta(text: str) -> None:
            print(text, end="", flush=True)
        result = await runtime.pipeline.answer(q, on_delta=async_delta)
        print("\n")
        print(f"路由: {result.route.value} | 轮次: {result.rounds} | 用时: {result.total_latency_ms}ms")
        if show_trace:
            for ev in result.trace:
                print(f"  {ev.step:<12} {ev.latency_ms:>6}ms  {json.dumps(ev.data, ensure_ascii=False)[:110]}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="ReflexRAG 命令行问答")
    parser.add_argument("question", nargs="?", help="问题; 省略则进入交互模式")
    parser.add_argument("--trace", action="store_true", help="打印阶段 Trace")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON 结果")
    args = parser.parse_args()
    if args.question:
        asyncio.run(ask_once(args.question, show_trace=args.trace, as_json=args.json))
    else:
        asyncio.run(repl(show_trace=args.trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
