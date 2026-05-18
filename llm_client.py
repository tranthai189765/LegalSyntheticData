"""
Async LLM client wrapping the OpenAI-compatible API.

Supports both:
  • FPT Cloud  (gpt-oss-120b  – reasoning model, CoT in reasoning_content)
  • OpenAI API (gpt-4o / gpt-4o-mini – standard model, supports json_object)

Active provider is chosen automatically from config (OPENAI_API_KEY takes
priority; set ACTIVE_PROVIDER=fpt to force FPT).
"""

import json
import re
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    ACTIVE_API_KEY, ACTIVE_BASE_URL, ACTIVE_MODEL,
    IS_REASONING_MODEL, LLM_CONCURRENCY, LLM_TEMPERATURE,
)

import asyncio


def _extract_json(raw: str) -> dict:
    """
    Multi-strategy JSON extraction.
    Needed for reasoning models that may prepend thinking text before the JSON.
    """
    raw = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
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
            api_key=ACTIVE_API_KEY,
            base_url=ACTIVE_BASE_URL,
        )
        self._semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
        self._is_reasoning = IS_REASONING_MODEL
        logger.info(
            f"LLMClient: model={ACTIVE_MODEL}  base_url={ACTIVE_BASE_URL}  "
            f"reasoning_model={self._is_reasoning}"
        )

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
        json_mode: bool = False,
    ) -> str:
        async with self._semaphore:
            kwargs: dict[str, Any] = {
                "model": ACTIVE_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            # Only non-reasoning models reliably support response_format
            if json_mode and not self._is_reasoning:
                kwargs["response_format"] = {"type": "json_object"}

            resp = await self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
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
        """
        Call LLM expecting a JSON response; returns parsed dict.

        For standard models (gpt-4o): uses response_format=json_object for
        reliable output.
        For reasoning models (gpt-oss-120b): relies on robust JSON extraction
        since response_format may not be supported.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        raw = await self._call(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
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
