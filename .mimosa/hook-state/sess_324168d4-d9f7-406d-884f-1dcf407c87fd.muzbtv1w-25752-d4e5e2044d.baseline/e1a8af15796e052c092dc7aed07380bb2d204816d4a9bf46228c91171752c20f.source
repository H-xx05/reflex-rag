"""语料加载:解析 markdown frontmatter 元信息。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str = ""
    publish_date: str = ""
    body: str = ""
    meta: dict = field(default_factory=dict)


def load_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = raw
    m = _FRONT.match(raw)
    if m:
        try:
            meta = json.loads(m.group(1))
        except json.JSONDecodeError:
            meta = {}
        body = raw[m.end() :]
    doc_id = meta.get("title") or path.stem
    return Document(
        doc_id=doc_id,
        title=meta.get("title") or path.stem,
        source_path=meta.get("source_path", str(path.name)),
        publish_date=meta.get("publish_date", ""),
        body=body,
        meta=meta,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    docs = [load_markdown(p) for p in sorted(corpus_dir.glob("*.md"))]
    if not docs:
        raise FileNotFoundError(f"{corpus_dir} 下没有语料, 先运行 scripts/fetch_corpus.py")
    return docs
