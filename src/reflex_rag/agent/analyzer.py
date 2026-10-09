"""查询分析器: 一次 LLM 调用完成路由决策 + 查询规范化 + (多跳时)查询分解。"""

from __future__ import annotations

import time

from reflex_rag.agent.prompts import ANALYZER_SYSTEM
from reflex_rag.agent.schemas import AnalysisResult, Route
from reflex_rag.llm.client import LLMClient, Usage, parse_json_dict

_VALID_ROUTES = {r.value for r in Route}


class QueryAnalyzer:
    def __init__(self, llm: LLMClient, corpus_titles: list[str]) -> None:
        self.llm = llm
        self.system = ANALYZER_SYSTEM.format(
            corpus_overview="\n".join(f"- 《{t}》" for t in corpus_titles)
        )

    async def analyze(
        self, question: str, usage: Usage, history: list[dict[str, str]] | None = None
    ) -> tuple[AnalysisResult, int]:
        """返回 (分析结果, 本步耗时 ms)。LLM 失败时回退到 simple 单轮检索, 保证可用性。

        history: 最近几轮对话, 用于指代消解("那试用期呢" -> 完整问题)。
        """
        t0 = time.perf_counter()
        user_content = question
        if history:
            lines = "\n".join(f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}" for m in history)
            user_content = f"【对话历史(仅供消解指代, 不要回答历史中的问题)】\n{lines}\n\n【当前问题】\n{question}"
        try:
            resp = await self.llm.chat(
                [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            usage.add(resp.usage)
            raw = parse_json_dict(resp.text)
        except Exception:  # noqa: BLE001 — 分析失败不能阻断服务, 降级为简单检索
            return (
                AnalysisResult(
                    route=Route.simple,
                    normalized_query=question,
                    reason="分析失败, 降级为单轮检索",
                ),
                int((time.perf_counter() - t0) * 1000),
            )

        route = raw.get("route")
        route = route if route in _VALID_ROUTES else Route.simple.value
        norm = str(raw.get("normalized_query") or question).strip() or question
        sub_queries = [
            q.strip() for q in raw.get("sub_queries") or [] if str(q).strip()
        ][:4]
        if route == Route.multi_hop.value and not sub_queries:
            sub_queries = [norm]
        result = AnalysisResult(
            route=Route(route),
            normalized_query=norm,
            reason=str(raw.get("reason") or ""),
            sub_queries=sub_queries,
        )
        return result, int((time.perf_counter() - t0) * 1000)
