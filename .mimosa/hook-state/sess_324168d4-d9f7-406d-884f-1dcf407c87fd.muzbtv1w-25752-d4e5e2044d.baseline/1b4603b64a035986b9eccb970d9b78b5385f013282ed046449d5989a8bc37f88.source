"""RRF(Reciprocal Rank Fusion):多路/多查询结果融合。

只用排名不用原始分数, 免去 BM25 分与余弦相似度的量纲对齐问题。
"""


def rrf_fuse(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """输入若干条 doc_id 排名列表, 返回融合后的 (doc_id, score) 降序列表。"""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for w, ranked in zip(weights, ranked_lists, strict=True):
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
