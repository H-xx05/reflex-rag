"""组合根: 一次性构建全部组件, 供 API / CLI / 评测复用。

评测场景可为不同 EvalProfile 注入共享的 llm/embedding/索引单例,
避免本地 Qdrant 存储被多个客户端同时打开。
"""

from __future__ import annotations

from dataclasses import dataclass

from reflex_rag.agent.analyzer import QueryAnalyzer
from reflex_rag.agent.critic import AnswerCritic
from reflex_rag.agent.executor import RetrievalExecutor
from reflex_rag.agent.generator import AnswerGenerator
from reflex_rag.agent.pipeline import AgentPipeline
from reflex_rag.config import EvalProfile, Settings, get_settings
from reflex_rag.embeddings.openai_compat import EmbeddingClient
from reflex_rag.llm.client import LLMClient
from reflex_rag.retrieval.bm25 import BM25Index
from reflex_rag.retrieval.hybrid import HybridRetriever
from reflex_rag.retrieval.reranker import BaseReranker, build_reranker
from reflex_rag.retrieval.vector_store import VectorStore


@dataclass
class Runtime:
    settings: Settings
    profile: EvalProfile
    llm: LLMClient
    embedding: EmbeddingClient
    retriever: HybridRetriever
    pipeline: AgentPipeline
    corpus_titles: list[str]


def build_runtime(
    settings: Settings | None = None,
    profile: EvalProfile | None = None,
    *,
    llm: LLMClient | None = None,
    embedding: EmbeddingClient | None = None,
    store: VectorStore | None = None,
    bm25: BM25Index | None = None,
    reranker: BaseReranker | None = None,
) -> Runtime:
    settings = settings or get_settings()
    profile = profile or EvalProfile(name="full")
    if llm is None and not settings.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY: 请复制 .env.example 为 .env 并填入你的 API key")
    if bm25 is None and not (settings.index_dir / "bm25.pkl").exists():
        raise RuntimeError(
            "尚未建立索引: 请先运行 python scripts/fetch_corpus.py && python scripts/ingest.py"
        )

    llm = llm or LLMClient(settings)
    embedding = embedding or EmbeddingClient(settings)
    store = store or VectorStore(
        settings.index_dir / "qdrant", settings.qdrant_collection, url=settings.qdrant_url or None
    )
    bm25 = bm25 or BM25Index.load(settings.index_dir / "bm25.pkl")
    corpus_titles = sorted({c.title for c in bm25.chunks})

    retriever = HybridRetriever(
        settings,
        embedding,
        store,
        bm25,
        reranker or build_reranker(settings),
        use_bm25=profile.use_hybrid,
    )

    analyzer = QueryAnalyzer(llm, corpus_titles)
    executor = RetrievalExecutor(retriever, llm, settings)
    generator = AnswerGenerator(llm)
    critic = AnswerCritic(llm)
    pipeline = AgentPipeline(
        settings, llm, analyzer, executor, generator, critic, corpus_titles, profile=profile
    )
    return Runtime(
        settings=settings,
        profile=profile,
        llm=llm,
        embedding=embedding,
        retriever=retriever,
        pipeline=pipeline,
        corpus_titles=corpus_titles,
    )
