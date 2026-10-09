"""OpenAI 兼容 /embeddings 接口客户端(适配 SiliconFlow / 通义 / 智谱等)。

各供应商对单请求条数/长度上限不一(通义 ≤20 条), 这里做了自适应降批:
超限错误自动把批量减半重试, 换供应商无需改代码。
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Awaitable, Callable

from openai import APIConnectionError, AsyncOpenAI, BadRequestError, RateLimitError

from reflex_rag.config import Settings

_BATCH_HINT = re.compile(r"batch|contents|too many|exceed|limit", re.IGNORECASE)


async def embed_in_batches(
    texts: list[str],
    call: Callable[[list[str]], Awaitable[list[list[float]]]],
    initial_size: int,
    *,
    throttle_s: float = 0.2,
) -> list[list[float]]:
    """按批嵌入并保持输入顺序; 遇到批量/长度超限的 400 错误自动减半重试。

    call: 单次网络调用(一批文本 -> 一批向量), 由调用方注入, 便于测试替换。
    """
    out: list[list[float]] = []
    i, size = 0, max(1, initial_size)
    while i < len(texts):
        try:
            out.extend(await call(texts[i : i + size]))
            i += size
            if i < len(texts):
                await asyncio.sleep(throttle_s)
        except BadRequestError as exc:
            if size == 1 or not _BATCH_HINT.search(str(exc)):
                raise
            size = max(1, size // 2)
            print(
                f"[warn] Embedding 批量被供应商拒绝, 自动降批至 {size} 条/次"
                f"(可在 .env 调大 embed_batch_size): {str(exc)[:120]}"
            )
    return out


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
        return await embed_in_batches(
            texts, self._embed, self.settings.embed_batch_size
        )

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def dimension(self) -> int:
        """探测向量维度(建集合时需要), 结果缓存。"""
        if self._dim is None:
            self._dim = len(await self.embed_query("维度探测"))
        return self._dim
