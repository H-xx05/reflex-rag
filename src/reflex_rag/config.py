"""全局配置:所有可调参数集中在 .env / 环境变量,代码内提供类型安全默认值。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ----
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # ---- Embedding(留空则复用 LLM 的连接信息) ----
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    # 各家 /embeddings 单请求条数上限不同: 通义 DashScope ≤20, SiliconFlow 宽松得多。
    # 默认取保守值 10, 客户端还会在超限时自动降批, 用 SiliconFlow 可在 .env 调大。
    embed_batch_size: int = 10

    # ---- 检索 ----
    retrieve_top_k: int = 24  # 向量/BM25 每路召回条数
    rrf_k: int = 60  # RRF 平滑常数
    final_top_k: int = 8  # 送给生成器的证据条数
    rerank_enabled: bool = False
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""

    # ---- Agent ----
    max_extra_rounds_simple: int = 1  # 简单题最多补检轮数
    max_extra_rounds_multi: int = 2  # 多跳题最多补检轮数
    critic_enabled: bool = True  # 引用/忠实度自校验
    analyzer_temperature: float = 0.0

    # ---- 存储 ----
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    qdrant_collection: str = "reflex_rag"

    @property
    def embed_base(self) -> str:
        return self.embed_base_url or self.llm_base_url

    @property
    def embed_key(self) -> str:
        return self.embed_api_key or self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()


@dataclass(frozen=True)
class EvalProfile:
    """系统消融开关: 评测流水线用它构建「朴素 RAG → 完整系统」的对比配置。"""

    name: str
    use_analyzer: bool = True  # False: 不做路由, 恒按 simple 单轮处理
    use_hybrid: bool = True  # False: 仅向量检索(无 BM25/RRF 融合)
    use_sufficiency: bool = True  # False: 检索一轮即生成, 无补检循环
    use_critic: bool = True  # False: 跳过引用/忠实性校验


#: 评测配置矩阵: 从"LangChain+向量库读文档"式基线到完整系统
PROFILES: dict[str, EvalProfile] = {
    "naive": EvalProfile("naive", use_analyzer=False, use_hybrid=False, use_sufficiency=False, use_critic=False),
    "hybrid": EvalProfile("hybrid", use_analyzer=False, use_sufficiency=False, use_critic=False),
    "adaptive": EvalProfile("adaptive", use_critic=False),
    "full": EvalProfile("full"),
}
