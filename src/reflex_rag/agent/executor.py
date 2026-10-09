"""检索执行器: 多轮迭代检索循环 —— 检索 → 充分性判定 → 不足则补充检索。

这是系统的"自反思"核心: 不盲目信任单轮召回, 用 LLM 判定证据是否覆盖问题要点,
缺口驱动下一轮查询(由充分性判定一并向模型要), 直到足够或达到轮数上限。
"""

from __future__ import annotations

import time

from reflex_rag.agent.prompts import SUFFICIENCY_PROMPT
from reflex_rag.agent.schemas import AnalysisResult, Route, SufficiencyVerdict, TraceEvent
from reflex_rag.config import Settings
from reflex_rag.llm.client import LLMClient, Usage, parse_json_dict
from reflex_rag.retrieval.hybrid import HybridRetriever
from reflex_rag.schemas import RetrievedChunk


class RetrievalExecutor:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient, settings: Settings) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    async def run(
        self,
        question: str,
        analysis: AnalysisResult,
        usage: Usage,
        *,
        judge_enabled: bool = True,
    ) -> tuple[list[RetrievedChunk], int, list[TraceEvent]]:
        """返回 (最终证据, 实际轮数, 过程事件)。judge_enabled=False 时退化为单轮检索。"""
        events: list[TraceEvent] = []
        max_extra = (
            self.settings.max_extra_rounds_simple
            if analysis.route == Route.simple
            else self.settings.max_extra_rounds_multi
        )

        queries = analysis.sub_queries or [analysis.normalized_query]
        t0 = time.perf_counter()
        evidence = await self.retriever.retrieve(queries, rerank_query=question)
        seen = {rc.chunk_id for rc in evidence}
        events.append(
            TraceEvent(
                step="retrieve",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                data={"round": 1, "queries": queries, "hits": len(evidence)},
            )
        )
        rounds = 1
        if not judge_enabled:
            return evidence, rounds, events
        while rounds <= max_extra:
            t1 = time.perf_counter()
            verdict = await self._judge(question, evidence, usage)
            events.append(
                TraceEvent(
                    step="sufficiency",
                    latency_ms=int((time.perf_counter() - t1) * 1000),
                    data={
                        "round": rounds,
                        "sufficient": verdict.sufficient,
                        "missing_aspects": verdict.missing_aspects,
                    },
                )
            )
            followups = [q for q in verdict.followup_queries if q.strip()][:2]
            if verdict.sufficient or not followups:
                break

            t2 = time.perf_counter()
            new_hits = await self.retriever.retrieve(followups, rerank_query=question)
            added = [rc for rc in new_hits if rc.chunk_id not in seen]
            for rc in added:
                seen.add(rc.chunk_id)
            events.append(
                TraceEvent(
                    step="retrieve",
                    latency_ms=int((time.perf_counter() - t2) * 1000),
                    data={"round": rounds + 1, "queries": followups, "hits": len(added)},
                )
            )
            if not added:
                break
            evidence.extend(added)
            rounds += 1

        return evidence, rounds, events

    async def _judge(
        self, question: str, evidence: list[RetrievedChunk], usage: Usage
    ) -> SufficiencyVerdict:
        digest = "\n".join(
            f"[{i}] 《{rc.title}》{rc.chapter}{'、'.join(rc.articles)} | {rc.text[:160]}"
            for i, rc in enumerate(evidence[:12], 1)
        )
        try:
            resp = await self.llm.chat(
                [
                    {"role": "user", "content": SUFFICIENCY_PROMPT.format(
                        question=question, evidence_digest=digest
                    )},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            usage.add(resp.usage)
            raw = parse_json_dict(resp.text)
            return SufficiencyVerdict(
                sufficient=bool(raw.get("sufficient")),
                missing_aspects=[str(x) for x in raw.get("missing_aspects") or []][:4],
                followup_queries=[str(x) for x in raw.get("followup_queries") or []][:3],
            )
        except Exception:  # noqa: BLE001 — 判定失败视为已充分, 不阻塞主流程
            return SufficiencyVerdict(sufficient=True)
