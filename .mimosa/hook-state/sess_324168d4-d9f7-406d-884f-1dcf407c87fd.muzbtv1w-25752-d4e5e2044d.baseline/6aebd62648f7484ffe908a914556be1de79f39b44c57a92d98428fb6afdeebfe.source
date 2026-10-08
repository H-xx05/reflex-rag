"""答案审查器: 确定性引用核对(代码, 零成本) + 忠实性抽查(LLM)。"""

from __future__ import annotations

import re
import time

from reflex_rag.agent.generator import _CITE
from reflex_rag.agent.prompts import CRITIC_PROMPT
from reflex_rag.agent.schemas import CriticResult
from reflex_rag.llm.client import LLMClient, Usage, parse_json_dict
from reflex_rag.schemas import RetrievedChunk


class AnswerCritic:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def verify(
        self, question: str, answer: str, evidence: list[RetrievedChunk], usage: Usage
    ) -> tuple[CriticResult, int]:
        """返回 (校验结果, 耗时 ms)。LLM 失败时只保留确定性检查结果。"""
        t0 = time.perf_counter()

        # 第一级: 引用编号越界检查(纯代码)
        cited_nums = {int(m.group(1)) for m in re.finditer(_CITE.pattern, answer)}
        unknown = sorted(n for n in cited_nums if not 1 <= n <= len(evidence))

        # 第二级: 陈述忠实性(LLM 抽查)
        unsupported: list[str] = []
        try:
            digest = "\n".join(
                f"[{i}] 《{rc.title}》{rc.chapter}{'、'.join(rc.articles)} | {rc.text[:200]}"
                for i, rc in enumerate(evidence[:12], 1)
            )
            resp = await self.llm.chat(
                [
                    {
                        "role": "user",
                        "content": CRITIC_PROMPT.format(
                            question=question, answer=answer, evidence_digest=digest
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=400,
            )
            usage.add(resp.usage)
            raw = parse_json_dict(resp.text)
            unsupported = [str(x) for x in raw.get("unsupported_claims") or []][:5]
        except Exception:  # noqa: BLE001 — 审查失败不阻塞, 保留第一级结果
            pass

        verdict = "pass" if not unknown and not unsupported else "warn"
        return (
            CriticResult(
                unknown_citations=[str(n) for n in unknown],
                unsupported_claims=unsupported,
                verdict=verdict,
            ),
            int((time.perf_counter() - t0) * 1000),
        )
