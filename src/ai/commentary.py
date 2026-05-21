"""Claude-generated short analytical commentaries for signals.

We feed the model the structured stats we already computed (form, Elo, etc.)
and ask for a 2-3 sentence commentary in Russian. Output is cached per
(match_id, market, pick) so we don't pay for the same answer twice.
"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, Optional

import aiohttp
from loguru import logger

from src.config import settings


API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-opus-4-7"

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
    if not settings.anthropic_api_key:
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
    body = {
        "model": MODEL,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(API_URL, json=body, headers=headers, timeout=30) as r:
                if r.status != 200:
                    text = await r.text()
                    logger.warning(f"Claude API {r.status}: {text[:200]}")
                    return None
                data = await r.json()
        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        ).strip()
        if text:
            _cache[cache_key] = text
            return text
    except asyncio.TimeoutError:
        logger.warning("Claude API timeout")
    except Exception as e:
        logger.warning(f"Claude API failed: {e}")
    return None
