"""LLM-as-judge: 答案忠实性/正确性打分 + 超范围问题拒答判定。

打分制(0~5)而非二值, 对小样本更稳健; 每个被评 item 只花 1 次廉价 LLM 调用。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from reflex_rag.llm.client import LLMClient, Usage, parse_json_dict

JUDGE_PROMPT = """你是严格的评测裁判。根据【金标答案】和【检索证据】给【系统回答】打分(0~5 整数):

- faithfulness 忠实性: 回答中的法律论断是否都能被证据支持, 编造条文/数字/时限记 0 分;
- correctness 正确性: 与金标答案相比, 关键结论(数字/时限/条件/程序)是否正确完整;
- 5 分 = 全部正确且忠实; 3 分 = 主结论正确但有次要缺漏; 1 分 = 关键结论错误; 0 分 = 编造或答非所问。

输出严格 JSON: {{"faithfulness": n, "correctness": n, "issues": ["主要扣分点"]}}

【用户问题】{question}
【金标答案】{gold_answer}
【检索证据】{evidence_digest}
【系统回答】{answer}"""

REFUSAL_PROMPT = """判断下面的助手回答是否为「恰当拒答」: 即明确说明问题超出其资料范围/不予作答,
或仅建议转介专业人士而没有给出实质性的法律结论。

用户问题: {question}
助手回答: {answer}

输出严格 JSON: {{"is_refusal": true/false, "reason": "一句话"}}"""


class JudgeResult(BaseModel):
    faithfulness: int = 0
    correctness: int = 0
    issues: list[str] = Field(default_factory=list)


class RefusalResult(BaseModel):
    is_refusal: bool = False
    reason: str = ""


class AnswerJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def judge_answer(
        self,
        question: str,
        gold_answer: str,
        evidence_digest: str,
        answer: str,
        usage: Usage,
    ) -> JudgeResult:
        try:
            resp = await self.llm.chat(
                [{"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, gold_answer=gold_answer,
                    evidence_digest=evidence_digest[:3000], answer=answer[:3000],
                )}],
                temperature=0.0,
                max_tokens=300,
            )
            usage.add(resp.usage)
            raw = parse_json_dict(resp.text)
            return JudgeResult(
                faithfulness=max(0, min(5, int(raw.get("faithfulness", 0)))),
                correctness=max(0, min(5, int(raw.get("correctness", 0)))),
                issues=[str(x) for x in raw.get("issues") or []][:4],
            )
        except Exception:  # noqa: BLE001 — 裁判失败计 0 并标注, 不中断评测
            return JudgeResult(faithfulness=-1, correctness=-1, issues=["judge_error"])

    async def judge_refusal(self, question: str, answer: str, usage: Usage) -> RefusalResult:
        try:
            resp = await self.llm.chat(
                [{"role": "user", "content": REFUSAL_PROMPT.format(
                    question=question, answer=answer[:1500]
                )}],
                temperature=0.0,
                max_tokens=150,
            )
            usage.add(resp.usage)
            raw = parse_json_dict(resp.text)
            return RefusalResult(is_refusal=bool(raw.get("is_refusal")), reason=str(raw.get("reason", "")))
        except Exception:  # noqa: BLE001
            return RefusalResult(is_refusal=False, reason="judge_error")
