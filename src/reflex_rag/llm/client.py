"""异步 LLM 客户端:OpenAI 兼容协议,内置指数退避重试、JSON 模式与流式输出。

所有调用都返回 token 用量,由上层聚合进 Trace,评测时据此统计成本。
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import APIConnectionError, AsyncOpenAI, RateLimitError

from reflex_rag.config import Settings

_RETRYABLE = (RateLimitError, APIConnectionError, asyncio.TimeoutError)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.llm_calls += other.llm_calls


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=60.0,
            max_retries=0,  # 重试自己控制,便于记录
        )
        self.extra_body: dict[str, Any] | None = None
        if settings.llm_extra_body_json.strip():
            self.extra_body = json.loads(settings.llm_extra_body_json)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 3,
    ) -> LLMResponse:
        """普通补全,失败按指数退避重试。"""
        start = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    temperature=self.settings.llm_temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.settings.llm_max_tokens,
                    extra_body=self.extra_body,
                )
                latency = int((time.perf_counter() - start) * 1000)
                usage = Usage(
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                    llm_calls=1,
                )
                text = resp.choices[0].message.content or ""
                return LLMResponse(text=text, usage=usage, latency_ms=latency)
            except _RETRYABLE as exc:
                last_exc = exc
                await asyncio.sleep(min(2**attempt + random.random(), 8))
        raise LLMError(f"LLM 调用失败(重试 {retries} 次): {last_exc}") from last_exc

    async def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """要求模型输出 JSON 并解析;剥离代码围栏,解析失败抛出 LLMError。"""
        resp = await self.chat(messages, **kwargs)
        return parse_json_dict(resp.text)

    async def chat_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """流式补全,逐段 yield 增量文本。

        传入 usage=Usage 实例时, 从流的 usage 尾帧回填真实 token 用量;
        供应商不回传时由调用方自行估算。异常时已重试过则直接抛出。
        """
        usage: Usage | None = kwargs.pop("usage", None)
        for attempt in range(2):
            try:
                stream = await self._client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    temperature=(
                        self.settings.llm_temperature
                        if kwargs.get("temperature") is None
                        else kwargs["temperature"]
                    ),
                    max_tokens=kwargs.get("max_tokens") or self.settings.llm_max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=self.extra_body,
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    elif chunk.usage is not None and usage is not None:
                        usage.add(
                            Usage(
                                prompt_tokens=chunk.usage.prompt_tokens or 0,
                                completion_tokens=chunk.usage.completion_tokens or 0,
                                llm_calls=1,
                            )
                        )
                return
            except _RETRYABLE:
                if attempt == 1:
                    raise
                await asyncio.sleep(1.5)


def parse_json_dict(text: str) -> dict[str, Any]:
    """宽容解析:优先取 ```json 围栏,否则找第一个平衡的 {...}。"""
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"无法解析模型 JSON 输出: {text[:200]!r}")


def extract_json_array(text: str) -> list[Any]:
    """同上,针对数组输出。"""
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError(f"无法解析模型 JSON 数组输出: {text[:200]!r}")
