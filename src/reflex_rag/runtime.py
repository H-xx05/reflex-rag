"""组合根: 一次性构建全部组件, 供 API / CLI / 评测复用。"""

from __future__ import annotations

from dataclasses import dataclass

from reflex_rag.agent.analyzer import QueryAnalyzer
from reflex_rag.agent.critic import AnswerCritic
from reflex_rag.agent.executor import RetrievalExecutor
from reflex_rag.agent.generator import AnswerGenerator
from reflex_rag.agent.pipeline import AgentPipeline
from reflex_rag.config import Settings, get_settings
from reflex_rag.embeddings.openai_compat import EmbeddingClient
from reflex_rag.llm.client import LLMClient
from reflex_rag.retrieval.bm25 import BM25Index
from reflex_rag.retrieval.hybrid import HybridRetriever
from reflex_rag.retrieval.reranker import build_reranker
from reflex_rag.retrieval.vector_store import VectorStore


@dataclass
class Runtime:
    settings: Settings
    llm: LLMClient
    embedding: EmbeddingClient
    retriever: HybridRetriever
    pipeline: AgentPipeline
    corpus_titles: list[str]


def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY, 请复制 .env.example 为 .env 并填写")

    bm25 = BM25Index.load(settings.index_dir / "bm25.pkl")
    corpus_titles = sorted({c.title for c in bm25.chunks})

    llm = LLMClient(settings)
    embedding = EmbeddingClient(settings)
    store = VectorStore(settings.index_dir / "qdrant", settings.qdrant_collection)
    retriever = HybridRetriever(
        settings, embedding, store, bm25, build_reranker(settings)
    )

    analyzer = QueryAnalyzer(llm, corpus_titles)
    executor = RetrievalExecutor(retriever, llm, settings)
    generator = AnswerGenerator(llm)
    critic = AnswerCritic(llm)
    pipeline = AgentPipeline(settings, llm, analyzer, executor, generator, critic, corpus_titles)
    return Runtime(
        settings=settings,
        llm=llm,
        embedding=embedding,
        retriever=retriever,
        pipeline=pipeline,
        corpus_titles=corpus_titles,
    )
