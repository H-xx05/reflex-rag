"""金标数据集模型与读写(JSONL)。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from reflex_rag.agent.schemas import Route


class GoldenItem(BaseModel):
    id: str
    question: str
    expected_route: Route
    gold_chunk_ids: list[str] = Field(default_factory=list)  # 闲聊/超范围为空
    gold_answer: str = ""
    source: str = "seed"  # seed(人工) | llm(辅助生成)
    meta: dict = Field(default_factory=dict)


def load_golden(path: Path) -> list[GoldenItem]:
    items = [
        GoldenItem(**json.loads(ln))
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not items:
        raise FileNotFoundError(f"{path} 为空, 先运行 scripts/build_golden.py")
    return items


def save_golden(items: list[GoldenItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")
