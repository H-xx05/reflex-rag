"""摄取管道: 语料 -> 结构感知分块 -> 嵌入 -> Qdrant 入库 + BM25 索引落盘。"""

from __future__ import annotations

import json
import time

from reflex_rag.config import Settings
from reflex_rag.embeddings.openai_compat import EmbeddingClient
from reflex_rag.ingestion.chunker import chunk_document
from reflex_rag.ingestion.loaders import load_corpus
from reflex_rag.retrieval.bm25 import BM25Index
from reflex_rag.retrieval.vector_store import VectorStore
from reflex_rag.schemas import Chunk


async def ingest(settings: Settings, *, rebuild: bool = False) -> dict:
    t0 = time.perf_counter()
    docs = load_corpus(settings.corpus_dir)
    chunks: list[Chunk] = [c for doc in docs for c in chunk_document(doc)]
    print(f"语料 {len(docs)} 篇 -> 分块 {len(chunks)} 个, 平均 {sum(map(lambda c: len(c.text), chunks))//len(chunks)} 字/块")  # noqa: E501

    embedding = EmbeddingClient(settings)
    dim = await embedding.dimension()
    print(f"嵌入模型 {embedding.model}, 维度 {dim}")

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store = VectorStore(settings.index_dir / "qdrant", settings.qdrant_collection)
    await store.ensure_collection(dim, rebuild=rebuild)

    t1 = time.perf_counter()
    vectors = await embedding.embed_documents([c.text for c in chunks])
    await store.upsert_chunks(chunks, vectors)
    print(f"向量入库 {len(chunks)} 条, 耗时 {time.perf_counter() - t1:.1f}s")

    BM25Index(chunks).save(settings.index_dir / "bm25.pkl")
    (settings.index_dir / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    stats = {
        "docs": len(docs),
        "chunks": len(chunks),
        "dim": dim,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(f"摄取完成: {stats}")
    return stats
