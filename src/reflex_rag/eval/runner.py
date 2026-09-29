"""评测执行器: 共享重资源单例, 按 EvalProfile 构建待测系统, 跑两类基准。

- 检索基准(零生成成本): naive_vector(纯向量) vs hybrid(向量+BM25+RRF),
  金标 chunk 上算 recall / MRR / nDCG;
- 端到端基准: 完整 pipeline 逐题作答 + LLM 裁判打分, 对比 naive/hybrid/full
  在路由准确率、拒答正确率、忠实性、正确性、延迟、成本上的差异。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from reflex_rag.agent.schemas import Route
from reflex_rag.config import PROFILES, EvalProfile, Settings, get_settings
from reflex_rag.embeddings.openai_compat import EmbeddingClient
from reflex_rag.eval.dataset import GoldenItem
from reflex_rag.eval.judge import AnswerJudge
from reflex_rag.eval.metrics import mean, mrr_at_k, ndcg_at_k, percentile, recall_at_k
from reflex_rag.llm.client import LLMClient, Usage
from reflex_rag.retrieval.bm25 import BM25Index
from reflex_rag.retrieval.vector_store import VectorStore
from reflex_rag.runtime import Runtime, build_runtime
from reflex_rag.schemas import RetrievedChunk

RETRIEVAL_CONFIGS = ("naive_vector", "hybrid")  # 检索层消融


@dataclass
class E2EItemResult:
    id: str
    expected_route: str
    pred_route: str
    route_ok: bool
    refusal_ok: bool | None = None
    faithfulness: int | None = None
    correctness: int | None = None
    rounds: int = 0
    n_citations: int = 0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    evidence_recall: float | None = None
    answer: str = ""
    issues: list[str] = field(default_factory=list)


def evidence_digest(evidence: list[RetrievedChunk]) -> str:
    return "\n".join(
        f"[{i}] 《{rc.title}》{rc.chapter}{'、'.join(rc.articles)} | {rc.text[:150]}"
        for i, rc in enumerate(evidence[:10], 1)
    )


class EvalRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # 重资源共享单例
        self.llm = LLMClient(self.settings)
        self.embedding = EmbeddingClient(self.settings)
        # 本地 Qdrant 是独占文件锁, Web 服务运行时会挡住评测进程。
        # 评测对索引只读(摄取后不变), 复制一份快照即可安全解耦。
        src = self.settings.index_dir / "qdrant"
        snapshot = self.settings.index_dir / "qdrant_eval_snapshot"
        if snapshot.exists():
            shutil.rmtree(snapshot)
        # .lock 被运行中的服务持有, 快照跳过它(Qdrant 打开时会自建)
        shutil.copytree(src, snapshot, ignore=shutil.ignore_patterns(".lock"))
        self.store = VectorStore(snapshot, self.settings.qdrant_collection)
        self.bm25 = BM25Index.load(self.settings.index_dir / "bm25.pkl")
        self.judge = AnswerJudge(self.llm)
        self.judge_usage = Usage()
        self.last_e2e_items: list[dict] = []  # 最近一次 run_e2e 的逐题明细

    def build_runtime(self, profile: EvalProfile) -> Runtime:
        return build_runtime(
            self.settings,
            profile,
            llm=self.llm,
            embedding=self.embedding,
            store=self.store,
            bm25=self.bm25,
        )

    # ---------- 检索基准 ----------

    async def run_retrieval_bench(self, items: list[GoldenItem], *, k: int = 8) -> list[dict]:
        gold_items = [it for it in items if it.gold_chunk_ids]
        rows = []
        for name in RETRIEVAL_CONFIGS:
            profile = PROFILES["naive" if name == "naive_vector" else "hybrid"]
            retriever = self.build_runtime(profile).retriever
            recalls5, recalls_k, mrrs, ndcgs = [], [], [], []
            t0 = time.perf_counter()
            for it in gold_items:
                hits = await retriever.retrieve([it.question], top_k=k)
                ranked = [h.chunk_id for h in hits]
                recalls5.append(recall_at_k(ranked, it.gold_chunk_ids, 5))
                recalls_k.append(recall_at_k(ranked, it.gold_chunk_ids, k))
                mrrs.append(mrr_at_k(ranked, it.gold_chunk_ids, k))
                ndcgs.append(ndcg_at_k(ranked, it.gold_chunk_ids, k))
            rows.append(
                {
                    "config": name,
                    "n": len(gold_items),
                    "recall@5": round(mean(recalls5), 4),
                    f"recall@{k}": round(mean(recalls_k), 4),
                    "mrr@k": round(mean(mrrs), 4),
                    "ndcg@k": round(mean(ndcgs), 4),
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                }
            )
        return rows

    # ---------- 端到端基准 ----------

    async def run_e2e(
        self,
        items: list[GoldenItem],
        profile_names: list[str],
        *,
        concurrency: int = 4,
        checkpoint_dir: Path | None = None,
    ) -> list[dict]:
        sem = asyncio.Semaphore(concurrency)
        rows = []
        for pname in profile_names:
            profile = PROFILES[pname]
            rt = self.build_runtime(profile)
            print(f"[e2e] 配置 {pname} 开始, {len(items)} 题…")

            async def one(item: GoldenItem) -> E2EItemResult:
                async with sem:
                    t0 = time.perf_counter()
                    usage = Usage()
                    res = await rt.pipeline.answer(item.question)
                    rec = E2EItemResult(
                        id=item.id,
                        expected_route=item.expected_route.value,
                        pred_route=res.route.value,
                        route_ok=res.route == item.expected_route,
                        rounds=res.rounds,
                        n_citations=len(res.citations),
                        latency_ms=res.total_latency_ms,
                        prompt_tokens=res.usage.prompt_tokens,
                        completion_tokens=res.usage.completion_tokens,
                        answer=res.answer,
                    )
                    usage.prompt_tokens += res.usage.prompt_tokens
                    usage.completion_tokens += res.usage.completion_tokens

                    if item.gold_chunk_ids:
                        rec.evidence_recall = round(
                            recall_at_k([rc.chunk_id for rc in res.evidence], item.gold_chunk_ids, 8), 4
                        )
                    if item.expected_route in (Route.simple, Route.multi_hop):
                        jr = await self.judge.judge_answer(
                            item.question, item.gold_answer, evidence_digest(res.evidence),
                            res.answer, self.judge_usage,
                        )
                        rec.faithfulness, rec.correctness = jr.faithfulness, jr.correctness
                        rec.issues = jr.issues
                    elif item.expected_route == Route.out_of_scope:
                        jr = await self.judge.judge_refusal(item.question, res.answer, self.judge_usage)
                        rec.refusal_ok = jr.is_refusal
                    return rec

            results: list[E2EItemResult] = await asyncio.gather(*[one(it) for it in items])
            rows.append(self._aggregate(pname, results))
            self.last_e2e_items.extend({**asdict(r), "config": pname} for r in results)
            # 每个配置跑完立即落盘, 防止后续配置异常导致整轮白跑
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                (checkpoint_dir / "per_item.json").write_text(
                    json.dumps(self.last_e2e_items, ensure_ascii=False, indent=1), encoding="utf-8"
                )
            print(f"[e2e] 配置 {pname} 完成: {rows[-1]['route_accuracy']:.0%} 路由准确率")
        return rows

    def _aggregate(self, pname: str, rs: list[E2EItemResult]) -> dict:
        oos = [r for r in rs if r.expected_route == Route.out_of_scope.value]
        qa = [r for r in rs if r.expected_route in (Route.simple.value, Route.multi_hop.value)]
        ref_ok = [r.refusal_ok for r in oos if r.refusal_ok is not None]
        faith = [r.faithfulness for r in qa if r.faithfulness is not None and r.faithfulness >= 0]
        corr = [r.correctness for r in qa if r.correctness is not None and r.correctness >= 0]
        ev = [r.evidence_recall for r in qa if r.evidence_recall is not None]
        lat = [float(r.latency_ms) for r in rs]
        return {
            "config": pname,
            "n": len(rs),
            "route_accuracy": round(mean([float(r.route_ok) for r in rs]), 4),
            "refusal_accuracy": round(mean([float(x) for x in ref_ok]), 4) if ref_ok else None,
            "faithfulness_0_5": round(mean([float(x) for x in faith]), 2) if faith else None,
            "correctness_0_5": round(mean([float(x) for x in corr]), 2) if corr else None,
            "evidence_recall@8": round(mean(ev), 4) if ev else None,
            "avg_rounds": round(mean([float(r.rounds) for r in rs]), 2),
            "avg_citations": round(mean([float(r.n_citations) for r in rs]), 2),
            "p50_latency_ms": int(percentile(lat, 50)),
            "p95_latency_ms": int(percentile(lat, 95)),
            "avg_prompt_tokens": round(mean([float(r.prompt_tokens) for r in rs]), 0),
            "avg_completion_tokens": round(mean([float(r.completion_tokens) for r in rs]), 0),
            "judge_tokens": self.judge_usage.prompt_tokens + self.judge_usage.completion_tokens,
        }
