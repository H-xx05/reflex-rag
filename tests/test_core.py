"""核心离线逻辑单元测试: 分块 / RRF / JSON解析 / 引用提取。"""

from __future__ import annotations

import pytest

from reflex_rag.ingestion.chunker import chunk_document
from reflex_rag.ingestion.loaders import Document
from reflex_rag.llm.client import parse_json_dict
from reflex_rag.retrieval.fusion import rrf_fuse
from reflex_rag.agent.generator import extract_citations
from reflex_rag.schemas import RetrievedChunk


def _doc(body: str) -> Document:
    return Document(doc_id="测试法", title="测试法", body=body)


def test_chunker_keeps_articles_atomic():
    body = """# 测试法

2020年1月1日 施行

<!-- INFO END -->

## 第一章 总则

第一条 这是第一条的内容。

第二条 这是第二条的内容, 内容稍长一些, 用来验证条文不被截断。

## 第二章 附则

第三条 这是第三条。
"""
    chunks = chunk_document(_doc(body))
    # 题注太短(<40字)不成块; 章节变更时断块, 同章内连续条文打包
    assert len(chunks) == 2
    assert chunks[0].chapter == "第一章 总则"
    assert chunks[0].articles == ["第一条"]
    assert chunks[1].chapter == "第二章 附则"
    assert chunks[1].articles == ["第二条", "第三条"]
    # 条文完整: 不出现半个条文
    assert "第二条的内容" in chunks[1].text
    assert chunks[1].text.count("第") >= 3


def test_chunker_preamble_and_appendix():
    body = """# 法

<!-- INFO END -->

第一条 正文。

附录:
一、附录条目一
二、附录条目二
"""
    chunks = chunk_document(_doc(body))
    assert chunks[0].articles == ["第一条"]
    assert chunks[-1].chapter == "附录"
    assert "附录条目一" in chunks[-1].text


def test_rrf_prefers_consensus():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"], ["b", "e"]])
    assert fused[0][0] == "b"


def test_parse_json_dict_variants():
    assert parse_json_dict('{"a": 1}') == {"a": 1}
    assert parse_json_dict('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_dict('前置说明 {"a": {"b": 2}} 后缀') == {"a": {"b": 2}}
    with pytest.raises(Exception):
        parse_json_dict("完全不是json")


def test_extract_citations_maps_and_bounds():
    ev = [
        RetrievedChunk(chunk_id=f"c{i}", doc_id="d", title="t", text="x")
        for i in range(1, 4)
    ]
    ans = "依据 [1] 和 [3], 另有 [9] 越界, 重复 [1]。"
    cites = extract_citations(ans, ev)
    assert [c.n for c in cites] == [1, 3]  # 越界被忽略, 去重且升序
    assert cites[0].chunk_id == "c1"
