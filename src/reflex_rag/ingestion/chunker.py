"""结构感知分块:法律法规是高度结构化文本, 按「条文」切分而非固定窗口。

- 原子单元: 一条(第X条) —— 引用溯源的最小语义单位, 永不跨块截断;
- 连续条文按目标块长(约 500 字)打包, 避免单条检索结果过碎;
- 章节(第X章/第X节)与法规名进入 chunk 元数据, 供引用展示与元数据过滤;
- 开头题注(公布日期/主席令号)与文末附录各自成块。
"""

from __future__ import annotations

import re

from reflex_rag.ingestion.loaders import Document
from reflex_rag.schemas import Chunk

_ARTICLE = re.compile(r"^第[零一二三四五六七八九十百千零两]+条")
_HEADING = re.compile(r"^#{1,4}\s*(第[零一二三四五六七八九十百千零两]+[章节].*)$")
_TARGET = 520
_MAX = 900
_INFO_END = "<!-- INFO END -->"


def chunk_document(doc: Document) -> list[Chunk]:
    lines = [
        ln.rstrip()
        for ln in doc.body.replace(_INFO_END, "").splitlines()
        if ln.strip() and not ln.strip().startswith("# ")  # 主标题行由元数据承担
    ]

    preamble: list[str] = []
    chapter = ""
    # [(章, 条号, 条文本)]
    articles: list[tuple[str, str]] = []
    appendix: list[str] = []

    cur_article_no = ""
    cur_article_lines: list[str] = []
    seen_article = False
    in_appendix = False

    for ln in lines:
        if in_appendix:
            appendix.append(ln)
            continue
        hm = _HEADING.match(ln)
        if hm:
            chapter = hm.group(1).strip()
            continue
        if ln.startswith("#"):  # 附录等非章节标题, 之后的内容整体归入附录
            if seen_article:
                in_appendix = True
                appendix.append(ln.lstrip("# ").strip())
            continue
        if ln.startswith("附录") and seen_article:
            in_appendix = True
            if cur_article_no:
                articles.append((chapter, cur_article_no, "\n".join(cur_article_lines)))
                cur_article_no, cur_article_lines = "", []
            appendix.append(ln)
            continue
        am = _ARTICLE.match(ln)
        if am:
            if cur_article_no:
                articles.append((chapter, cur_article_no, "\n".join(cur_article_lines)))
            cur_article_no, cur_article_lines = am.group(0), [ln]
            seen_article = True
            continue
        if seen_article:
            cur_article_lines.append(ln)
        else:
            preamble.append(ln)

    if cur_article_no:
        articles.append((chapter, cur_article_no, "\n".join(cur_article_lines)))

    chunks: list[Chunk] = []
    idx = 0

    def emit(chapter_: str, art_list: list[tuple[str, str]], extra_head: str = "") -> None:
        nonlocal idx
        text = "\n".join([extra_head, *[a[2] for a in art_list]]).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{idx:03d}",
                doc_id=doc.doc_id,
                title=doc.title,
                chapter=chapter_,
                articles=[a[1] for a in art_list],
                text=text,
                source_path=doc.source_path,
                publish_date=doc.publish_date,
            )
        )
        idx += 1

    if preamble:
        text = "\n".join(preamble).strip()
        if len(text) > 40:  # 太短的题注没有检索价值
            emit("", [], extra_head=text)

    buf: list[tuple[str, str]] = []
    buf_len = 0
    buf_chapter = ""
    for art_chapter, no, body in articles:
        if buf and (buf_len + len(body) > _TARGET or art_chapter != buf_chapter):
            emit(buf_chapter, buf)
            buf, buf_len = [], 0
        buf.append((art_chapter, no, body))
        buf_len += len(body)
        buf_chapter = art_chapter
        if buf_len >= _MAX:
            emit(buf_chapter, buf)
            buf, buf_len = [], 0
    if buf:
        emit(buf_chapter, buf)

    appendix_text = "\n".join(appendix).strip()
    if appendix_text:
        emit("附录", [], extra_head=appendix_text)

    return chunks
