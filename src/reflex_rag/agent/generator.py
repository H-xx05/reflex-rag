"""带引用的答案生成器: 流式输出, 引用编号在生成后做确定性提取与映射。"""

from __future__ import annotations

import re
from typing import Any, AsyncIterator

from reflex_rag.agent.prompts import GENERATOR_SYSTEM
from reflex_rag.agent.schemas import Citation
from reflex_rag.llm.client import LLMClient
from reflex_rag.schemas import RetrievedChunk

_CITE = re.compile(r"\[(\d{1,2})\]")


def build_evidence_block(evidence: list[RetrievedChunk]) -> str:
    parts = []
    for i, rc in enumerate(evidence, 1):
        art = "、".join(rc.articles) if rc.articles else ""
        head = f"《{rc.title}》"
        if rc.chapter:
            head += f" {rc.chapter}"
        if art:
            head += f" {art}"
        if rc.publish_date:
            head += f"(发布 {rc.publish_date})"
        parts.append(f"[{i}] {head}\n{rc.text}")
    return "\n\n".join(parts)


def extract_citations(answer: str, evidence: list[RetrievedChunk]) -> list[Citation]:
    """确定性提取引用编号 -> 证据对象; 编号越界(幻觉引用)直接忽略, 由审查器另行报告。"""
    cited: list[Citation] = []
    seen: set[int] = set()
    for m in _CITE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(evidence) and n not in seen:
            seen.add(n)
            rc = evidence[n - 1]
            cited.append(
                Citation(
                    n=n,
                    chunk_id=rc.chunk_id,
                    title=rc.title,
                    chapter=rc.chapter,
                    articles=rc.articles,
                    publish_date=rc.publish_date,
                    source_path=rc.source_path,
                )
            )
    return sorted(cited, key=lambda c: c.n)


class AnswerGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def stream(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        *,
        extra_instruction: str = "",
        usage: Any = None,
    ) -> AsyncIterator[str]:
        """流式生成答案; 迭代器耗尽后调用方可用 extract_citations 做引用映射。

        usage: 可选的 Usage 累积器, 回填流式调用的真实 token 用量。
        """
        user = (
            f"【用户问题】\n{question}\n\n【检索证据】\n{build_evidence_block(evidence)}"
        )
        if extra_instruction:
            user += f"\n\n【注意】上一版回答存在以下问题, 请修正后重新回答:\n{extra_instruction}"
        return self._stream(user, usage)

    async def _stream(self, user: str, usage: Any = None) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": user},
        ]
        async for delta in self.llm.chat_stream(messages, temperature=0.3, usage=usage):
            yield delta
