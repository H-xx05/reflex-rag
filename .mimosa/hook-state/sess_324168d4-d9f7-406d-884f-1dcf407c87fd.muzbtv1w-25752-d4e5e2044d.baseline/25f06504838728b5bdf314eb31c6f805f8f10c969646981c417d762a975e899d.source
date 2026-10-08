"""从 LawRefBook/Laws 开源仓库抓取劳动法领域语料到 data/corpus/。

用法:
    python scripts/fetch_corpus.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/LawRefBook/Laws/master/"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"

# (repo 内路径, 归一化文件名)
DOCS: list[tuple[str, str]] = [
    ("社会法/劳动法(2018-12-29).md", "中华人民共和国劳动法.md"),
    ("社会法/劳动合同法(2012-12-28).md", "中华人民共和国劳动合同法.md"),
    ("社会法/社会保险法(2018-12-29).md", "中华人民共和国社会保险法.md"),
    ("社会法/就业促进法(2015-04-24).md", "中华人民共和国就业促进法.md"),
    ("诉讼与非诉讼程序法/劳动争议调解仲裁法(2007-12-29).md", "中华人民共和国劳动争议调解仲裁法.md"),
    ("行政法规/劳动合同法实施条例(2008-09-18).md", "劳动合同法实施条例.md"),
    ("行政法规/工伤保险条例(2010-12-20).md", "工伤保险条例.md"),
    ("行政法规/女职工劳动保护特别规定(2012-04-28).md", "女职工劳动保护特别规定.md"),
    ("行政法规/职工带薪年休假条例(2007-12-14).md", "职工带薪年休假条例.md"),
    ("行政法规/劳动保障监察条例(2004-11-01).md", "劳动保障监察条例.md"),
    ("行政法规/国务院关于职工工作时间的规定(1995-03-25).md", "国务院关于职工工作时间的规定.md"),
    (
        "司法解释/最高人民法院关于审理劳动争议案件适用法律问题的解释（一）(2020-12-29).md",
        "最高法劳动争议司法解释(一).md",
    ),
    (
        "司法解释/最高人民法院关于审理劳动争议案件适用法律问题的解释（二）(2025-07-31).md",
        "最高法劳动争议司法解释(二).md",
    ),
]


def fetch(repo_path: str) -> str:
    url = RAW_BASE + urllib.parse.quote(repo_path)
    req = urllib.request.Request(url, headers={"User-Agent": "reflex-rag-corpus-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def extract_date(name: str) -> str:
    m = re.search(r"\((\d{4}-\d{2}-\d{2})\)", name)
    return m.group(1) if m else ""


def main() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for repo_path, out_name in DOCS:
        out_path = CORPUS_DIR / out_name
        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[skip] {out_name} 已存在")
            ok += 1
            continue
        try:
            text = fetch(repo_path)
            # 统一 frontmatter,记录元信息供摄取管道使用
            front = {
                "title": out_path.stem,
                "source": "LawRefBook/Laws",
                "source_path": repo_path,
                "publish_date": extract_date(repo_path),
            }
            body = text.lstrip("\ufeff")
            if not body.startswith("---"):
                body = "---\n" + json.dumps(front, ensure_ascii=False) + "\n---\n\n" + body
            out_path.write_text(body, encoding="utf-8")
            print(f"[ok]   {out_name}  ({len(body)} chars)")
            ok += 1
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            failed.append((repo_path, str(exc)))
            print(f"[fail] {repo_path}: {exc}")
    print(f"\n完成 {ok}/{len(DOCS)}")
    if failed:
        print("失败项:", *failed, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
