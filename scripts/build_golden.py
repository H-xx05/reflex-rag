"""构建金标评测集 data/golden/golden.jsonl。

构成:
- 种子题(人工写死): 闲聊 10 + 超范围 12 —— 无需金标条文;
- 单跳题(LLM 辅助): 从各法规抽样条文反向出题, 金标=该条文块;
- 多跳题(LLM 辅助): 从两部相关法规各取一块, 要求问题必须同时用到两者, 金标=两块。

固定随机种子保证数据集可复现(评测回归的前提)。
用法: python scripts/build_golden.py [--single 30] [--multi 22]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reflex_rag.config import get_settings
from reflex_rag.eval.dataset import GoldenItem, save_golden
from reflex_rag.llm.client import LLMClient, Usage, parse_json_dict
from reflex_rag.schemas import Chunk

SEEDS_DIR = Path(__file__).resolve().parent.parent / "data"

CHITCHAT = [
    "你好", "在吗?", "谢谢你啦", "你是谁?能做什么?", "今天天气真好",
    "帮我写一份周报", "讲个笑话吧", "晚安", "你用的什么模型呀", "嗨, 吃饭了吗",
]

OUT_OF_SCOPE = [
    "离婚时房子怎么分?",
    "故意伤人会被判几年?",
    "交通事故对方全责, 我该怎么索赔?",
    "我想注册一个商标, 流程是什么?",
    "个税专项附加扣除怎么填报?",
    "公司想上市需要满足什么条件?",
    "猫把邻居家沙发抓坏了, 我要赔吗?",
    "公司拖欠我8千块工资, 你预测一下我仲裁能赢吗?",
    "比特币挖矿在中国违法吗?",
    "出租车绕路该怎么投诉?",
    "邻居装修太吵扰民怎么办?",
    "我股票被套牢了, 该怎么办?",
]

# 多跳组合: 这两部法规之间存在高频联动问题
MULTI_COMBOS = [
    ("中华人民共和国劳动合同法", "劳动合同法实施条例"),
    ("中华人民共和国劳动合同法", "中华人民共和国劳动争议调解仲裁法"),
    ("中华人民共和国劳动法", "中华人民共和国劳动合同法"),
    ("中华人民共和国社会保险法", "工伤保险条例"),
    ("中华人民共和国劳动合同法", "职工带薪年休假条例"),
    ("中华人民共和国劳动法", "国务院关于职工工作时间的规定"),
    ("中华人民共和国劳动合同法", "最高法劳动争议司法解释(一)"),
    ("中华人民共和国劳动争议调解仲裁法", "最高法劳动争议司法解释(一)"),
    ("中华人民共和国劳动合同法", "女职工劳动保护特别规定"),
    ("工伤保险条例", "最高法劳动争议司法解释(一)"),
    ("中华人民共和国劳动合同法", "中华人民共和国社会保险法"),
]

SINGLE_PROMPT = """你是评测集构建专家。根据下面的法条原文, 站在劳动者视角设计一个中文问题。

要求:
1. 问题能主要依据这段法条回答, 不要出现"根据第X条"之类的提示;
2. 口语自然, 像真实用户会问的;
3. 标准答案控制在 150 字内, 必须包含法条中的关键数字/时限/比例。

【法条】《{title}》{chapter}{articles}
{text}

输出严格 JSON: {{"question": "...", "answer": "..."}}"""

MULTI_PROMPT = """你是评测集构建专家。下面是两部不同法规中的法条, 请设计一个中文问题,
要求: 完整回答必须同时用到这两段法条(如: 一部规定情形/责任, 另一部规定程序/时限/计算)。
不要出现"根据第X条"之类的提示; 口语自然; 标准答案 200 字内。

【法条A】《{title_a}》{chapter_a}{articles_a}
{text_a}

【法条B】《{title_b}》{chapter_b}{articles_b}
{text_b}

输出严格 JSON: {{"question": "...", "answer": "..."}}"""


def load_chunks() -> list[Chunk]:
    settings = get_settings()
    raw = json.loads((settings.index_dir / "chunks.json").read_text(encoding="utf-8"))
    return [Chunk(**c) for c in raw]


def by_doc(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    out: dict[str, list[Chunk]] = {}
    for c in chunks:
        out.setdefault(c.title, []).append(c)
    return out


def fmt_head(c: Chunk) -> str:
    art = "、".join(c.articles) if c.articles else ""
    head = f"{c.chapter} " if c.chapter else ""
    return head + (art + "\n" if art else "\n")


async def gen_single(llm: LLMClient, usage: Usage, c: Chunk, sem: asyncio.Semaphore, idx: int) -> GoldenItem | None:
    async with sem:
        for _ in range(2):
            try:
                resp = await llm.chat(
                    [{"role": "user", "content": SINGLE_PROMPT.format(
                        title=c.title, chapter=c.chapter + " " if c.chapter else "",
                        articles="、".join(c.articles), text=c.text,
                    )}],
                    temperature=0.7, max_tokens=400,
                )
                usage.add(resp.usage)
                raw = parse_json_dict(resp.text)
                q, a = str(raw["question"]).strip(), str(raw["answer"]).strip()
                if q and a:
                    return GoldenItem(
                        id=f"simple-{idx:03d}", question=q,
                        expected_route="simple", gold_chunk_ids=[c.chunk_id],
                        gold_answer=a, source="llm",
                        meta={"title": c.title, "articles": c.articles},
                    )
            except Exception:  # noqa: BLE001
                continue
        return None


async def gen_multi(llm: LLMClient, usage: Usage, a: Chunk, b: Chunk, sem: asyncio.Semaphore, idx: int) -> GoldenItem | None:
    async with sem:
        for _ in range(2):
            try:
                resp = await llm.chat(
                    [{"role": "user", "content": MULTI_PROMPT.format(
                        title_a=a.title, chapter_a=a.chapter + " " if a.chapter else "",
                        articles_a="、".join(a.articles), text_a=a.text,
                        title_b=b.title, chapter_b=b.chapter + " " if b.chapter else "",
                        articles_b="、".join(b.articles), text_b=b.text,
                    )}],
                    temperature=0.7, max_tokens=500,
                )
                usage.add(resp.usage)
                raw = parse_json_dict(resp.text)
                q, ans = str(raw["question"]).strip(), str(raw["answer"]).strip()
                if q and ans:
                    return GoldenItem(
                        id=f"multi-{idx:03d}", question=q,
                        expected_route="multi_hop", gold_chunk_ids=[a.chunk_id, b.chunk_id],
                        gold_answer=ans, source="llm",
                        meta={"titles": [a.title, b.title]},
                    )
            except Exception:  # noqa: BLE001
                continue
        return None


def pick(chunks: list[Chunk], lo: int, hi: int) -> list[Chunk]:
    pool = [c for c in chunks if lo <= len(c.text) <= hi and c.articles]
    return pool


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=int, default=30)
    parser.add_argument("--multi", type=int, default=22)
    args = parser.parse_args()

    settings = get_settings()
    llm = LLMClient(settings)
    usage = Usage()
    rng = random.Random(42)
    chunks = load_chunks()
    docs = by_doc(chunks)
    items: list[GoldenItem] = []
    sem = asyncio.Semaphore(4)

    # 种子题
    for i, q in enumerate(CHITCHAT):
        items.append(GoldenItem(id=f"chit-{i:03d}", question=q, expected_route="chitchat", source="seed"))
    for i, q in enumerate(OUT_OF_SCOPE):
        items.append(GoldenItem(id=f"oos-{i:03d}", question=q, expected_route="out_of_scope", source="seed"))

    # 单跳: 大法各 3 块, 其余各 2 块, 总量截到 --single
    big = {"中华人民共和国劳动合同法", "中华人民共和国劳动法", "中华人民共和国社会保险法", "工伤保险条例"}
    single_tasks = []
    for title, cs in sorted(docs.items()):
        pool = pick(cs, 100, 700)
        rng.shuffle(pool)
        for c in pool[: 3 if title in big else 2]:
            single_tasks.append(gen_single(llm, usage, c, sem, len(single_tasks)))
    singles = [x for x in await asyncio.gather(*single_tasks) if x]
    rng.shuffle(singles)
    items += singles[: args.single]
    print(f"单跳题生成 {len(singles)} -> 保留 {min(args.single, len(singles))}")

    # 多跳: 每个组合抽 2 对
    multi_tasks = []
    idx = 0
    for title_a, title_b in MULTI_COMBOS:
        pa, pb = pick(docs.get(title_a, []), 120, 600), pick(docs.get(title_b, []), 120, 600)
        if not pa or not pb:
            print(f"[warn] 组合缺料: {title_a} + {title_b}")
            continue
        for _ in range(2):
            a, b = rng.choice(pa), rng.choice(pb)
            idx += 1
            multi_tasks.append(gen_multi(llm, usage, a, b, sem, idx))
    multis = [x for x in await asyncio.gather(*multi_tasks) if x]
    items += multis[: args.multi]
    print(f"多跳题生成 {len(multis)} -> 保留 {min(args.multi, len(multis))}")

    out = settings.corpus_dir.parent / "golden" / "golden.jsonl"
    save_golden(items, out)
    dist: dict[str, int] = {}
    for it in items:
        dist[it.expected_route.value] = dist.get(it.expected_route.value, 0) + 1
    print(f"\n金标集写入 {out}: 共 {len(items)} 题, 分布 {dist}")
    print(f"构建用 tokens: {usage.prompt_tokens}+{usage.completion_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
