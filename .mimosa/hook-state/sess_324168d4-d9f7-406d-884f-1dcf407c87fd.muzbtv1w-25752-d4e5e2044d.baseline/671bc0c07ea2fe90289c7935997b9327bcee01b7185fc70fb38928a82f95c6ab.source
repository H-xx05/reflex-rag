"""共享数据模型:Chunk 是索引与检索的基本单元。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str = Field(description="全局唯一, 形如 {doc_id}:{序号}")
    doc_id: str
    title: str = Field(description="所属法律法规名")
    chapter: str = Field(default="", description="所在章节, 如 '第三章 劳动合同的订立'")
    articles: list[str] = Field(default_factory=list, description="包含的条文号, 如 ['第十条']")
    text: str
    source_path: str = ""
    publish_date: str = ""


class RetrievedChunk(Chunk):
    """检索命中结果: score 为最终排序分(RRF 或 rerank 分), sources 记录命中路径。"""

    score: float = 0.0
    sources: list[str] = Field(default_factory=list, description="命中该块的查询/通道列表")
