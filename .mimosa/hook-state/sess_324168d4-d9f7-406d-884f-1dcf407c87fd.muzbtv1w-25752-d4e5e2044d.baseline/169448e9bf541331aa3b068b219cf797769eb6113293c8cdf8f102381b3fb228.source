"""摄取入口: python scripts/ingest.py [--rebuild]"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reflex_rag.config import get_settings
from reflex_rag.ingestion.pipeline import ingest


def main() -> int:
    parser = argparse.ArgumentParser(description="语料摄取: 分块 + 向量入库 + BM25 索引")
    parser.add_argument("--rebuild", action="store_true", help="清空集合并重建")
    args = parser.parse_args()
    stats = asyncio.run(ingest(get_settings(), rebuild=args.rebuild))
    return 0 if stats["chunks"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
