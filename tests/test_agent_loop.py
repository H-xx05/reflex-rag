"""Agent 循环 mock 测试: 不触网, 覆盖路由降级 / 迭代检索轮数 / 引用校验。"""

from __future__ import annotations

import pytest

from reflex_rag.agent.analyzer import QueryAnalyzer
from reflex_rag.agent.critic import AnswerCritic
from reflex_rag.agent.executor import RetrievalExecutor
from reflex_rag.agent.schemas import AnalysisResult, Route
from reflex_rag.config import get_settings
from reflex_rag.llm.client import LLMResponse, Usage
from reflex_rag.schemas import RetrievedChunk


class FakeLLM:
    """按序吐出预设回复的假 LLM。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.i = 0
        self.prompts: list[str] = []

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        self.prompts.append(messages[-1]["content"])
        text = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return LLMResponse(
            text=text, usage=Usage(1, 1, 1), latency_ms=1
        )


class FakeRetriever:
    """每个查询返回一个以其命名的确定性块, 便于断言合并结果。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def retrieve(self, queries: list[str], *, rerank_query=None, top_k=None):
        self.calls.append(list(queries))
        out = []
        for q in queries:
            out.append(
                RetrievedChunk(chunk_id=f"chunk-{q}", doc_id="d", title="t", text=f"text {q}")
            )
        return out


@pytest.fixture
def settings():
    return get_settings()


async def test_analyzer_parses_multi_hop():
    llm = FakeLLM(['{"route":"multi_hop","normalized_query":"拖欠工资维权","reason":"r",'
                   '"sub_queries":["工资支付规定","仲裁程序"]}'])
    analyzer = QueryAnalyzer(llm, ["测试法"])  # type: ignore[arg-type]
    result, _ = await analyzer.analyze("公司欠我工资怎么办", Usage())
    assert result.route == Route.multi_hop
    assert result.sub_queries == ["工资支付规定", "仲裁程序"]


async def test_analyzer_falls_back_on_garbage():
    llm = FakeLLM(["这不是JSON"])
    analyzer = QueryAnalyzer(llm, ["测试法"])  # type: ignore[arg-type]
    result, _ = await analyzer.analyze("试用期最长多久", Usage())
    assert result.route == Route.simple
    assert result.normalized_query == "试用期最长多久"
    assert "降级" in result.reason


async def test_executor_iterates_until_sufficient(settings):
    retriever = FakeRetriever()
    llm = FakeLLM([
        '{"sufficient": false, "missing_aspects": ["时效"], "followup_queries": ["仲裁时效"]}',
        '{"sufficient": true, "missing_aspects": [], "followup_queries": []}',
    ])
    executor = RetrievalExecutor(retriever, llm, settings)  # type: ignore[arg-type]
    analysis = AnalysisResult(
        route=Route.multi_hop, normalized_query="原问题", sub_queries=["子查询A", "子查询B"]
    )
    evidence, rounds, events = await executor.run("原问题", analysis, Usage())

    assert rounds == 2
    assert retriever.calls[0] == ["子查询A", "子查询B"]
    assert retriever.calls[1] == ["仲裁时效"]
    assert {rc.chunk_id for rc in evidence} == {"chunk-子查询A", "chunk-子查询B", "chunk-仲裁时效"}
    assert [e.step for e in events].count("sufficiency") == 2


async def test_executor_single_round_when_judge_disabled(settings):
    retriever = FakeRetriever()
    llm = FakeLLM([])
    executor = RetrievalExecutor(retriever, llm, settings)  # type: ignore[arg-type]
    analysis = AnalysisResult(route=Route.multi_hop, normalized_query="原问题", sub_queries=["A"])
    evidence, rounds, events = await executor.run("原问题", analysis, Usage(), judge_enabled=False)

    assert rounds == 1
    assert len(retriever.calls) == 1
    assert not any(e.step == "sufficiency" for e in events)
    assert {rc.chunk_id for rc in evidence} == {"chunk-A"}


async def test_critic_flags_unknown_citations():
    evidence = [
        RetrievedChunk(chunk_id=f"c{i}", doc_id="d", title="t", text="x") for i in (1, 2)
    ]
    llm = FakeLLM(['{"unsupported_claims": [], "overall": "pass"}'])
    critic = AnswerCritic(llm)  # type: ignore[arg-type]
    result, _ = await critic.verify("q", "依据[1]和[3]…", evidence, Usage())
    assert result.unknown_citations == ["3"]
    assert result.verdict == "warn"
