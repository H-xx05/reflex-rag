"""Embedding 自适应降批逻辑测试(不触网, 用假供应商)。"""

from __future__ import annotations

import httpx
import pytest
from openai import BadRequestError

from reflex_rag.embeddings.openai_compat import embed_in_batches


def _api_400(message: str) -> BadRequestError:
    req = httpx.Request("POST", "https://fake/v1/embeddings")
    resp = httpx.Response(400, json={"error": {"message": message}}, request=req)
    return BadRequestError(f"Error code: 400 - {message}", response=resp, body=None)


@pytest.mark.asyncio
async def test_embed_in_batches_shrinks_on_batch_limit(capsys):
    """模拟通义: 单请求 >20 条必 400; 断言自动降批、顺序保持、全部成功。"""
    calls: list[int] = []

    async def call(batch: list[str]) -> list[list[float]]:
        calls.append(len(batch))
        if len(batch) > 20:
            raise _api_400("batch size is invalid, it should not be larger than 20.: input.contents")
        return [[float(int(t[1:]))] for t in batch]  # 向量里编码文本下标

    texts = [f"t{i}" for i in range(53)]
    vecs = await embed_in_batches(texts, call, initial_size=64)

    assert len(vecs) == 53
    assert [v[0] for v in vecs] == list(range(53))  # 顺序与输入一致
    assert calls[0] == 53 and calls[1] == 32  # 首批全量尝试, 400 后减半
    assert max(calls[2:]) <= 20  # 之后所有成功批次都 ≤ 上限
    assert "[warn]" in capsys.readouterr().out  # 降批有提示


@pytest.mark.asyncio
async def test_embed_in_batches_reraises_unrelated_400():
    """与批量无关的 400(如模型名错误)必须原样抛出, 不能被降批掩盖。"""

    async def call(batch: list[str]) -> list[list[float]]:
        raise _api_400("InvalidParameter: model not found")

    with pytest.raises(BadRequestError, match="model not found"):
        await embed_in_batches(["a", "b"], call, initial_size=10)
