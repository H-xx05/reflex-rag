"""Qdrant 封装:本地嵌入模式运行(数据落盘), 后续可无缝切到服务端部署。"""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import AsyncQdrantClient, models

from reflex_rag.schemas import Chunk


def _uuid(chunk_id: str) -> str:
    """chunk_id -> 确定性 UUID, 重建索引时可覆盖旧点。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reflex-rag/{chunk_id}"))


class VectorStore:
    """本地嵌入模式(path)或服务端模式(url)。服务端模式支持多客户端并发访问。"""

    def __init__(self, path: Path, collection: str, *, url: str | None = None) -> None:
        self.collection = collection
        if url:
            self.client = AsyncQdrantClient(url=url)
        else:
            self.client = AsyncQdrantClient(path=str(path))

    async def ensure_collection(self, dim: int, *, rebuild: bool = False) -> None:
        if rebuild and await self.client.collection_exists(self.collection):
            await self.client.delete_collection(self.collection)
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )

    async def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        points = [
            models.PointStruct(
                id=_uuid(c.chunk_id),
                vector=v,
                payload=c.model_dump(),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        await self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    async def search(self, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        resp = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            (Chunk(**p.payload), p.score) for p in resp.points if p.payload
        ]
