"""LLM-generated short analytical commentaries for signals.

Supports two backends:
  1. Native Anthropic API (default) — POST {anthropic}/v1/messages with x-api-key.
  2. OpenAI-compatible proxy (NeuroAPI, OpenRouter, vLLM, ...) — POST
     {LLM_BASE_URL}/chat/completions with Authorization: Bearer ....

Switching is automatic: set LLM_BASE_URL to a proxy and the OpenAI path is used.
Leave it empty and we hit Anthropic directly. Both paths take the same prompt
and return the same commentary string. Output is cached per (match, market, pick)
so we don't pay for the same answer twice.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Optional

import aiohttp
from loguru import logger

from src.config import settings


_cache: Dict[str, str] = {}


_PROMPT_TEMPLATE = """Ты футбольный аналитик. На основе этих данных напиши КОРОТКИЙ (2-3 предложения, до 280 символов) аналитический комментарий на русском про прогноз модели. Без воды, без советов "ставьте/не ставьте", только суть: на чём основан прогноз и какой главный риск.

Матч: {home} vs {away}
Лига: {competition}
Прогноз модели: {market} → {pick}
Вероятность модели: {prob:.0%}
{odds_block}

Статистика (доступная до матча):
- Elo: {home} {home_elo:.0f}, {away} {away_elo:.0f}
- Форма (PPG за 10 матчей): {home} {home_form:.2f}, {away} {away_form:.2f}
- Голы дома ({home}): забивает {home_gf:.2f}, пропускает {home_ga:.2f} за матч
- Голы в гостях ({away}): забивает {away_gf:.2f}, пропускает {away_ga:.2f} за матч
- H2H: домашняя команда выигрывает в {h2h_wr:.0%} случаев, средний тотал {h2h_g:.2f}

Отвечай ОДНИМ абзацем, без эмодзи и markdown."""


def _resolve_provider() -> tuple[str, str, str, str]:
    """Returns (mode, url, api_key, model). mode is 'openai' or 'anthropic'."""
    model = settings.llm_model or "claude-sonnet-4-6"
    if settings.llm_base_url:
        base = settings.llm_base_url.rstrip("/")
        if not base.endswith("/v1") and "/v1/" not in base:
            base = base + "/v1"
        url = base + "/chat/completions"
        key = settings.llm_api_key or settings.anthropic_api_key
        return "openai", url, key, model
    return "anthropic", "https://api.anthropic.com/v1/messages", settings.anthropic_api_key, model


async def _call_openai(url: str, key: str, model: str, prompt: str, max_tokens: int = 350) -> Optional[str]:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=body, headers=headers, timeout=60) as r:
            if r.status != 200:
                text = await r.text()
                logger.warning(f"LLM OpenAI-proxy {r.status} ({model}): {text[:300]}")
                return None
            data = await r.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"LLM OpenAI-proxy unexpected response shape: {e}; data={str(data)[:300]}")
        return None


async def _call_anthropic(url: str, key: str, model: str, prompt: str, max_tokens: int = 350) -> Optional[str]:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=body, headers=headers, timeout=60) as r:
            if r.status != 200:
                text = await r.text()
                logger.warning(f"Anthropic API {r.status} ({model}): {text[:300]}")
                return None
            data = await r.json()
    return "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    ).strip()


async def call_llm(prompt: str, max_tokens: int = 350) -> Optional[str]:
    mode, url, key, model = _resolve_provider()
    if not key:
        return None
    try:
        if mode == "openai":
            return await _call_openai(url, key, model, prompt, max_tokens=max_tokens)
        return await _call_anthropic(url, key, model, prompt, max_tokens=max_tokens)
    except asyncio.TimeoutError:
        logger.warning("LLM timeout")
        return None
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


async def generate_commentary(
    *,
    match_id: int,
    home: str,
    away: str,
    competition: str,
    market: str,
    pick: str,
    prob: float,
    book_odds: float,
    edge: float,
    features: dict,
) -> Optional[str]:
    mode, url, key, model = _resolve_provider()
    if not key:
        logger.info("LLM commentary skipped: no API key set (LLM_API_KEY / ANTHROPIC_API_KEY)")
        return None

    cache_key = f"{match_id}:{market}:{pick}"
    if cache_key in _cache:
        return _cache[cache_key]

    odds_block = (
        f"Кф букмекера: {book_odds:.2f}, edge модели: {edge*100:.1f}%"
        if book_odds and book_odds > 1.0
        else "Котировки букмекера недоступны."
    )
    prompt = _PROMPT_TEMPLATE.format(
        home=home, away=away, competition=competition,
        market=market, pick=pick, prob=prob,
        odds_block=odds_block,
        home_elo=features.get("home_elo", 1500),
        away_elo=features.get("away_elo", 1500),
        home_form=features.get("home_form_pts", 1.5),
        away_form=features.get("away_form_pts", 1.5),
        home_gf=features.get("home_gf_home_avg", features.get("home_gf_avg", 1.3)),
        home_ga=features.get("home_ga_home_avg", features.get("home_ga_avg", 1.3)),
        away_gf=features.get("away_gf_away_avg", features.get("away_gf_avg", 1.0)),
        away_ga=features.get("away_ga_away_avg", features.get("away_ga_avg", 1.4)),
        h2h_wr=features.get("h2h_home_winrate", 0.5),
        h2h_g=features.get("h2h_avg_goals", 2.6),
    )

    try:
        if mode == "openai":
            text = await _call_openai(url, key, model, prompt)
        else:
            text = await _call_anthropic(url, key, model, prompt)
    except asyncio.TimeoutError:
        logger.warning("LLM timeout")
        return None
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None

    if text:
        _cache[cache_key] = text
        logger.info(f"LLM commentary OK ({mode}, {model}) match {match_id}: {len(text)} chars")
        return text
    logger.warning(f"LLM returned empty text for match {match_id}")
    return None
