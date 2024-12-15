import asyncio
import os
from typing import AsyncIterator, Union

import google.generativeai as genai
import numpy as np
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

from .utils import locate_json_string_body_from_string, logger


def log_retry_attempt(retry_state):
    """リトライ前にログを出力する"""
    exception = retry_state.outcome.exception()
    logger.warning(
        f"リトライを実行します。(試行回数: {retry_state.attempt_number}, "
        f"待機時間: {retry_state.next_action.sleep} 秒, "
        f"例外: {exception})"
    )


class GeminiRateLimiter:
    def __init__(self, max_concurrent_calls=5):
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()


# グローバルなレートリミッターインスタンスを作成
rate_limiter = GeminiRateLimiter(max_concurrent_calls=5)  # 同時実行数を5に制限


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random(min=1, max=10),
    retry=retry_if_exception_type(ResourceExhausted),
    before_sleep=log_retry_attempt,
)
async def gemini_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    api_key=None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    model = genai.GenerativeModel(model)

    messages = []
    if system_prompt:
        messages.append({"role": "user", "parts": [system_prompt]})
    for history_message in history_messages:
        messages.append(
            {"role": history_message["role"], "parts": [history_message["content"]]}
        )
    messages.append({"role": "user", "parts": [prompt]})

    # Add logging
    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug("Full context:")

    stream = True if kwargs.get("stream") else False
    if stream:
        """cannot cache stream response"""
        response = model.generate_content_async(contents=messages, **kwargs)

        async def inner():
            async for chunk in await response:
                yield chunk.text

        return inner()
    else:
        response = await model.generate_content_async(contents=messages, **kwargs)
        return response.text


async def gemini_complete(
    prompt, system_prompt=None, history_messages=[], api_key=None, **kwargs
) -> str:
    kwargs.pop("hashing_kv", None)
    keyword_extraction = kwargs.pop("keyword_extraction", None)

    # Gemini APIリソース制限回避のため、同時実行数を制限
    async with rate_limiter:
        result = await gemini_complete_if_cache(
            "models/gemini-1.5-flash-latest",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            **kwargs,
        )

    if keyword_extraction:
        return locate_json_string_body_from_string(result)
    return result


async def gemini_embedding(
    texts: list[str], embed_model="models/text-embedding-004", api_key=None, **kwargs
) -> np.ndarray:
    # APIキーの設定
    if api_key:
        genai.configure(api_key=api_key)
    elif "GOOGLE_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        raise ValueError(
            "APIキーが必要です。api_key引数として渡すか、GOOGLE_API_KEY環境変数を設定してください。"
        )

    result = genai.embed_content(model=embed_model, content=texts)
    return np.array(result["embedding"])
