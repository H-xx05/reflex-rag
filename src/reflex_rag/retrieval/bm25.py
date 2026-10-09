"""jieba 分词 + BM25 关键词检索, 与向量检索互补(法条编号/术语等精确匹配)。

法条场景里很多查询含精确字面(如 "竞业限制" "第82条"), 纯向量召回容易漏,
BM25 是性价比最高的补路。索引与语料块一起 pickle 落盘。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from reflex_rag.schemas import Chunk

jieba.initialize()


def _tokenize(text: str) -> list[str]:
    # search 模式会额外吐出短粒度切分, 对 "未签订劳动合同" 这类连写查询更友好
    return [t for t in jieba.lcut_for_search(text) if t.strip()]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        corpus = [_tokenize(f"{c.title} {c.chapter} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_k] if s > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as f:
            obj: BM25Index = pickle.load(f)
        return obj
