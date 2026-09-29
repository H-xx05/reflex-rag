"""FastAPI 服务: SSE 流式问答接口, 阶段事件与答案增量实时推送。

事件流(每行 data: {json}\n\n):
    stage  — Agent 阶段事件(analyze/retrieve/sufficiency/generate/verify)
    delta  — 答案文本增量
    final  — 完整结果(引用/路由/轮次/用量/耗时)
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from reflex_rag.agent.schemas import AnswerResult, TraceEvent
from reflex_rag.runtime import Runtime, build_runtime

STATIC_DIR = Path(__file__).parent / "static"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []  # 最近对话, 供指代消解; 服务端只取末尾若干条


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime: Runtime = build_runtime()
    yield


app = FastAPI(title="ReflexRAG", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": app.version}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    runtime: Runtime = app.state.runtime

    async def event_stream():
        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=256)

        async def on_event(ev: TraceEvent) -> None:
            await queue.put({"type": "stage", "step": ev.step, **ev.model_dump()})

        async def on_delta(text: str) -> None:
            await queue.put({"type": "delta", "text": text})

        def final_payload(result: AnswerResult) -> dict:
            return {
                "type": "final",
                "answer": result.answer,
                "citations": [c.model_dump() for c in result.citations],
                "route": result.route.value,
                "sub_queries": result.sub_queries,
                "rounds": result.rounds,
                "verification": result.verification.model_dump() if result.verification else None,
                "trace": [e.model_dump() for e in result.trace],
                "usage": result.usage.model_dump(),
                "total_latency_ms": result.total_latency_ms,
                "evidence_preview": [
                    {
                        "chunk_id": rc.chunk_id,
                        "title": rc.title,
                        "chapter": rc.chapter,
                        "articles": rc.articles,
                        "text": rc.text[:280],
                    }
                    for rc in result.evidence[:12]
                ],
            }

        async def run() -> None:
            try:
                result = await runtime.pipeline.answer(
                    req.question,
                    history=[m.model_dump() for m in req.history[-6:]],
                    on_event=on_event,
                    on_delta=on_delta,
                )
                await queue.put(final_payload(result))
            except Exception as exc:  # noqa: BLE001 — 错误也走事件流, 前端统一渲染
                await queue.put({"type": "error", "message": str(exc)[:300]})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while (item := await queue.get()) is not None:
                yield _sse(item)
        finally:
            task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
