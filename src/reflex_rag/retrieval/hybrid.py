"""混合检索编排: 多查询 → 向量+BM25 双路 → RRF 融合 → 可选 rerank。

同一问题会被拆成多个子查询(多跳场景), 每个子查询独立走双路召回;
跨查询再做一次 RRF 汇总, 被多个子查询同时命中的块会排到前面。
"""

from __future__ import annotations

import asyncio

from reflex_rag.config import Settings
from reflex_rag.embeddings.openai_compat import EmbeddingClient
from reflex_rag.retrieval.bm25 import BM25Index
from reflex_rag.retrieval.fusion import rrf_fuse
from reflex_rag.retrieval.reranker import BaseReranker
from reflex_rag.retrieval.vector_store import VectorStore
from reflex_rag.schemas import RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedding: EmbeddingClient,
        vector_store: VectorStore,
        bm25: BM25Index,
        reranker: BaseReranker,
    ) -> None:
        self.settings = settings
        self.embedding = embedding
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker

    async def retrieve(
        self,
        queries: list[str],
        *,
        rerank_query: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """queries: 本轮要执行的全部查询(原始问题或子查询); 返回去重后的最终证据。"""
        top_k = top_k or self.settings.final_top_k
        k = self.settings.retrieve_top_k
        vec_lists, bm25_lists = await asyncio.gather(
            asyncio.gather(*[self._vector_search(q, k) for q in queries]),
            asyncio.gather(*[self._bm25_search(q, k) for q in queries]),
        )

        # 第一步: 每个查询内部, 向量路 + BM25 路 RRF 融合
        per_query: list[list[RetrievedChunk]] = []
        for vres, bres in zip(vec_lists, bm25_lists, strict=True):
            pool: dict[str, RetrievedChunk] = {}
            for rc in vres + bres:
                _merge_hit(pool, rc)
            fused = rrf_fuse(
                [[rc.chunk_id for rc in vres], [rc.chunk_id for rc in bres]],
                k=self.settings.rrf_k,
            )
            fused_chunks = []
            for cid, score in fused:
                rc = pool[cid]
                rc.score = score
                fused_chunks.append(rc)
            per_query.append(fused_chunks)

        # 第二步: 跨查询 RRF 汇总(多子查询共同命中 → 权威块)
        cross = rrf_fuse([[c.chunk_id for c in lst] for lst in per_query], k=self.settings.rrf_k)
        cross_score = dict(cross)
        merged_pool: dict[str, RetrievedChunk] = {}
        for lst in per_query:
            for rc in lst:
                _merge_hit(merged_pool, rc)
        fused_order = [merged_pool[cid] for cid, _ in cross if cid in merged_pool]
        for rc in fused_order:
            rc.score = cross_score[rc.chunk_id]

        candidates = fused_order[: max(top_k * 2, 16)]
        return await self.reranker.rerank(rerank_query or queries[0], candidates, top_k)

    async def _vector_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        vec = await self.embedding.embed_query(query)
        hits = await self.vector_store.search(vec, top_k)
        return [
            RetrievedChunk(**c.model_dump(), score=score, sources=[f"vec:{query}"])
            for c, score in hits
        ]

    async def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        hits = await asyncio.to_thread(self.bm25.search, query, top_k)
        return [
            RetrievedChunk(**c.model_dump(), score=score, sources=[f"bm25:{query}"])
            for c, score in hits
        ]


def _merge_hit(pool: dict[str, RetrievedChunk], rc: RetrievedChunk) -> None:
    """把命中块并入池, 保留首次对象并合并命中路径信息。"""
    if rc.chunk_id in pool:
        for s in rc.sources:
            if s not in pool[rc.chunk_id].sources:
                pool[rc.chunk_id].sources.append(s)
    else:
        pool[rc.chunk_id] = rc
