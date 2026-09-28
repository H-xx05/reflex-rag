"""OpenAI 兼容 /embeddings 接口客户端(适配 SiliconFlow / 通义 / 智谱等)。"""

from __future__ import annotations

import asyncio
import random
import time

from openai import APIConnectionError, AsyncOpenAI, RateLimitError

from reflex_rag.config import Settings


class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.embed_base, api_key=settings.embed_key, timeout=60.0, max_retries=0
        )
        self._dim: int | None = None

    @property
    def model(self) -> str:
        return self.settings.embed_model

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = await self._client.embeddings.create(
                    model=self.settings.embed_model, input=texts
                )
                return [item.embedding for item in resp.data]
            except (RateLimitError, APIConnectionError) as exc:
                last_exc = exc
                await asyncio.sleep(min(2**attempt + random.random(), 15))
        raise RuntimeError(f"Embedding 调用失败: {last_exc}") from last_exc

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入, 自动分批 + 限速, 保持输入顺序。"""
        out: list[list[float]] = []
        batch = self.settings.embed_batch_size
        for i in range(0, len(texts), batch):
            out.extend(await self._embed(texts[i : i + batch]))
            if i + batch < len(texts):
                await asyncio.sleep(0.2)
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def dimension(self) -> int:
        """探测向量维度(建集合时需要), 结果缓存。"""
        if self._dim is None:
            t0 = time.perf_counter()
            self._dim = len(await self.embed_query("维度探测"))
        return self._dim
