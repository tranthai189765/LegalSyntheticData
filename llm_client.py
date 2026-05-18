"""
Async LLM client wrapping the OpenAI-compatible FPT Cloud API.

All agents share a single client instance (singleton pattern).
Rate limiting is handled via asyncio.Semaphore (LLM_CONCURRENCY).
Retries use tenacity with exponential backoff.
"""

import asyncio
import json
import re
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config import FPT_API_KEY, FPT_BASE_URL, FPT_MODEL, LLM_CONCURRENCY, LLM_TEMPERATURE


def _extract_json(raw: str) -> dict:
    """
    Try several strategies to extract a JSON object from LLM output.
    The model sometimes wraps the JSON in thinking text or markdown.
    """
    raw = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    stripped = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    stripped = re.sub(r"\s*```$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Find the last {...} block (model may prepend thinking text)
    matches = list(re.finditer(r"\{[\s\S]*\}", raw))
    for m in reversed(matches):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid JSON found in response (first 200 chars): {raw[:200]}")


class LLMClient:
    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=FPT_API_KEY,
            base_url=FPT_BASE_URL,
        )
        self._semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call(
        self,
        messages: list[dict],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = 2048,
        response_format: str = "text",   # "text" | "json"
    ) -> str:
        async with self._semaphore:
            kwargs: dict[str, Any] = {
                "model": FPT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}

            resp = await self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            # gpt-oss-120b: content holds the final answer, reasoning_content
            # holds the chain-of-thought. Only use content.
            return msg.content or ""

    async def chat(
        self,
        system: str,
        user: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = 2048,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        return await self._call(messages, temperature=temperature, max_tokens=max_tokens)

    async def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict:
        """Call LLM expecting a JSON response; returns parsed dict.

        Does NOT pass response_format=json because gpt-oss-120b (reasoning model)
        may not support it and sometimes outputs thinking before the JSON object.
        Instead we use robust JSON extraction.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        # Use text mode (no response_format) - extract JSON ourselves
        raw = await self._call(messages, temperature=temperature, max_tokens=max_tokens)
        try:
            return _extract_json(raw)
        except ValueError as exc:
            logger.warning(f"JSON parse failed; raw={raw[:200]!r}")
            raise ValueError(f"LLM did not return valid JSON: {exc}") from exc


# Module-level singleton
_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
