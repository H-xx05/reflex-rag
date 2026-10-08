"""可插拔 Reranker: SiliconFlow / Jina 兼容的 /rerank 接口; 未启用时为恒等直通。"""

from __future__ import annotations

import httpx

from reflex_rag.config import Settings
from reflex_rag.schemas import RetrievedChunk


class BaseReranker:
    async def rerank(self, query: str, docs: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return docs[:top_k]


class APIReranker(BaseReranker):
    """POST {base}/rerank  {"model": ..., "query": ..., "documents": [...]}"""

    def __init__(self, settings: Settings) -> None:
        self._url = settings.rerank_base_url.rstrip("/") + "/rerank"
        self._headers = {"Authorization": f"Bearer {settings.rerank_api_key}"}
        self._model = settings.rerank_model

    async def rerank(self, query: str, docs: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not docs:
            return []
        payload = {
            "model": self._model,
            "query": query,
            "documents": [d.text for d in docs],
            "top_n": len(docs),
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self._url, json=payload, headers=self._headers)
            resp.raise_for_status()
            results = resp.json()["results"]
        out: list[RetrievedChunk] = []
        for r in results:
            chunk = docs[r["index"]].model_copy(deep=True)
            chunk.score = float(r["relevance_score"])
            out.append(chunk)
        return out[:top_k]


def build_reranker(settings: Settings) -> BaseReranker:
    if settings.rerank_enabled and settings.rerank_base_url and settings.rerank_model:
        return APIReranker(settings)
    return BaseReranker()
