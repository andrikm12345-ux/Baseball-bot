from __future__ import annotations

from typing import List, Dict

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings


_TAVILY_URL = "https://api.tavily.com/search"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
async def _post(body: dict) -> dict:
    async with aiohttp.ClientSession() as s:
        async with s.post(_TAVILY_URL, json=body, timeout=20) as r:
            if r.status != 200:
                text = await r.text()
                logger.warning(f"Tavily {r.status}: {text[:200]}")
                r.raise_for_status()
            return await r.json()


async def tavily_search(query: str, days: int = 7, max_results: int = 5) -> List[Dict[str, str]]:
    """Returns [{title, content}] or [] if no key/error."""
    key = settings.tavily_api_key
    if not key:
        return []
    body = {
        "api_key": key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "days": days,
        "include_answer": False,
    }
    try:
        data = await _post(body)
    except Exception as e:
        logger.warning(f"Tavily search failed for {query!r}: {e}")
        return []
    return [
        {"title": r.get("title", ""), "content": r.get("content", "")[:500]}
        for r in data.get("results", [])
    ]
