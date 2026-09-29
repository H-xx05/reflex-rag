"""AgentPipeline: 把 分析 → 检索循环 → 生成 → 审查 串成一条可观测的流水线。

设计要点:
- 每个阶段都产生 TraceEvent(步骤/耗时/关键中间量), 通过 on_event 回调实时外发,
  API 层转成 SSE 推给前端做"思考过程"可视化;
- on_delta 回调透传生成器流式增量, 供 API/CLI 做打字机效果;
- 所有 LLM 用量聚合进 AnswerResult.usage, 评测时统计成本;
- EvalProfile 控制消融: 朴素基线关掉路由/混合检索/补检/校验, 供评测对比。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from reflex_rag.agent.analyzer import QueryAnalyzer
from reflex_rag.agent.critic import AnswerCritic
from reflex_rag.agent.executor import RetrievalExecutor
from reflex_rag.agent.generator import AnswerGenerator, extract_citations
from reflex_rag.agent.prompts import OUT_OF_SCOPE_REPLY
from reflex_rag.agent.schemas import AnalysisResult, AnswerResult, CriticResult, Route, TraceEvent, TraceUsage
from reflex_rag.config import EvalProfile, Settings
from reflex_rag.llm.client import LLMClient, Usage

OnEvent = Callable[[TraceEvent], Awaitable[None]]
OnDelta = Callable[[str], Awaitable[None]]


class AgentPipeline:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        analyzer: QueryAnalyzer,
        executor: RetrievalExecutor,
        generator: AnswerGenerator,
        critic: AnswerCritic,
        corpus_titles: list[str],
        profile: EvalProfile | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.analyzer = analyzer
        self.executor = executor
        self.generator = generator
        self.critic = critic
        self.corpus_titles = corpus_titles
        self.profile = profile or EvalProfile(name="full")

    async def answer(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        on_event: OnEvent | None = None,
        on_delta: OnDelta | None = None,
    ) -> AnswerResult:
        """history: 最近几轮对话 [{role, content}], 供路由分析做指代消解。"""
        t0 = time.perf_counter()
        usage = Usage()
        events: list[TraceEvent] = []

        async def emit(event: TraceEvent) -> None:
            events.append(event)
            if on_event:
                await on_event(event)

        # 1. 查询分析(路由); 朴素模式跳过, 恒按 simple 单轮处理
        if self.profile.use_analyzer:
            analysis, latency = await self.analyzer.analyze(
                question, usage, history=(history or [])[-4:]
            )
        else:
            analysis, latency = (
                AnalysisResult(
                    route=Route.simple, normalized_query=question, reason="朴素模式: 不做路由"
                ),
                0,
            )
        await emit(
            TraceEvent(
                step="analyze",
                latency_ms=latency,
                data={
                    "route": analysis.route.value,
                    "reason": analysis.reason,
                    "sub_queries": analysis.sub_queries,
                },
            )
        )

        answer = ""
        citations: list = []
        verification: CriticResult | None = None
        evidence: list = []
        rounds = 0

        if analysis.route == Route.out_of_scope:
            answer = OUT_OF_SCOPE_REPLY.format(
                corpus_list="\n".join(f"- 《{t}》" for t in self.corpus_titles)
            )
            if on_delta:
                await on_delta(answer)
        elif analysis.route == Route.chitchat:
            t1 = time.perf_counter()
            chunks = []
            msgs = [
                {
                    "role": "system",
                    "content": "你是劳动法问答助手'问律', 简短友好地回应寒暄, 并引导用户提出劳动法问题, 30字以内。",
                }
            ]
            if history:
                msgs += history[-2:]
            msgs.append({"role": "user", "content": question})
            async for delta in self.llm.chat_stream(msgs, temperature=0.5, max_tokens=120):
                chunks.append(delta)
                if on_delta:
                    await on_delta(delta)
            answer = "".join(chunks)
            usage.completion_tokens += len(answer) // 2
            usage.llm_calls += 1
            await emit(
                TraceEvent(
                    step="generate", latency_ms=int((time.perf_counter() - t1) * 1000),
                    data={"mode": "chitchat_direct"},
                )
            )
        else:
            # 2. 迭代检索(含充分性自反思; 朴素/混合模式退化为单轮)
            evidence, rounds, retrieve_events = await self.executor.run(
                question, analysis, usage, judge_enabled=self.profile.use_sufficiency
            )
            for ev in retrieve_events:
                await emit(ev)

            # 3. 流式生成
            t2 = time.perf_counter()
            chunks = []
            async for delta in self.generator.stream(question, evidence):
                chunks.append(delta)
                if on_delta:
                    await on_delta(delta)
            answer = "".join(chunks)
            usage.completion_tokens += len(answer) // 2
            usage.llm_calls += 1
            await emit(
                TraceEvent(
                    step="generate",
                    latency_ms=int((time.perf_counter() - t2) * 1000),
                    data={"chars": len(answer)},
                )
            )
            citations = extract_citations(answer, evidence)

            # 4. 引用与忠实性自校验
            if self.settings.critic_enabled and self.profile.use_critic:
                verification, latency = await self.critic.verify(question, answer, evidence, usage)
                await emit(
                    TraceEvent(
                        step="verify", latency_ms=latency,
                        data=verification.model_dump(),
                    )
                )
                # 引用编号越界属实质缺陷, 触发一次修正重生成
                if verification.unknown_citations:
                    await emit(
                        TraceEvent(
                            step="generate",
                            data={"regenerate": True, "reason": "引用编号越界, 修正后重新生成"},
                        )
                    )
                    answer = await self._regenerate_with_fix(
                        question, evidence, verification, on_delta, usage
                    )
                    citations = extract_citations(answer, evidence)

        return AnswerResult(
            question=question,
            answer=answer,
            citations=citations,
            route=analysis.route,
            sub_queries=analysis.sub_queries,
            rounds=rounds,
            evidence=evidence,
            verification=verification,
            trace=events,
            usage=TraceUsage(**usage.__dict__),
            total_latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    async def _regenerate_with_fix(
        self,
        question: str,
        evidence: list,
        verification: CriticResult,
        on_delta: OnDelta | None,
        usage: Usage,
    ) -> str:
        fix = f"引用编号越界: {verification.unknown_citations}。只允许引用 1~{len(evidence)} 的编号。"
        chunks = []
        async for delta in self.generator.stream(question, evidence, extra_instruction=fix):
            chunks.append(delta)
            if on_delta:
                await on_delta(delta)
        usage.llm_calls += 1
        usage.completion_tokens += len("".join(chunks)) // 2
        return "".join(chunks)
