"""Agent 层数据模型:路由决策、迭代检索中间态、校验结果、全链路 Trace。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from reflex_rag.schemas import RetrievedChunk


class Route(str, Enum):
    chitchat = "chitchat"  # 闲聊/寒暄, 免检索直答
    simple = "simple"  # 单跳事实型, 一轮检索
    multi_hop = "multi_hop"  # 多跳/综合判断, 查询分解 + 迭代检索
    out_of_scope = "out_of_scope"  # 超出语料领域, 礼貌拒答


class AnalysisResult(BaseModel):
    route: Route
    normalized_query: str = ""
    reason: str = ""
    sub_queries: list[str] = Field(default_factory=list)


class SufficiencyVerdict(BaseModel):
    sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)
    followup_queries: list[str] = Field(default_factory=list)


class CriticResult(BaseModel):
    unknown_citations: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    verdict: str = "pass"  # pass | warn


class Citation(BaseModel):
    n: int
    chunk_id: str
    title: str
    chapter: str = ""
    articles: list[str] = Field(default_factory=list)
    publish_date: str = ""
    source_path: str = ""


class TraceEvent(BaseModel):
    step: str  # analyze | retrieve | sufficiency | generate | verify
    latency_ms: int = 0
    data: dict = Field(default_factory=dict)


class TraceUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0


class AnswerResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: Route
    sub_queries: list[str] = Field(default_factory=list)
    rounds: int = 0
    evidence: list[RetrievedChunk] = Field(default_factory=list, exclude=True)
    verification: CriticResult | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    usage: TraceUsage = Field(default_factory=TraceUsage)
    total_latency_ms: int = 0
