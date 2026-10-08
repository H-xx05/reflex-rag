"""检索指标: 基于 chunk_id 金标, 纯代码计算, 零 LLM 成本。"""

from __future__ import annotations


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def mrr_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    for i, cid in enumerate(ranked[:k], 1):
        if cid in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    """二值增益 nDCG; 金标多条时衡量整体排序质量。"""
    if not gold:
        return 0.0
    gold_set = set(gold)
    dcg = sum(1.0 / _log2(i + 1) for i, cid in enumerate(ranked[:k], 1) if cid in gold_set)
    ideal = sum(1.0 / _log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def _log2(n: int) -> float:
    from math import log2

    return log2(n)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]
