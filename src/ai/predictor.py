"""AI-сторона: Claude получает реальные котировки букмекера со всеми линиями
(тотал/фора) и рыночными no-vig вероятностями, и выбирает ОДНУ ставку с
максимальным положительным расхождением (своя вероятность − рыночная).
XGBoost не используется.
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, Optional

from loguru import logger

from src.ai.commentary import call_llm
from src.config import settings
from src.data.web_search import tavily_search


_CACHE_TTL_SEC = 12 * 3600
_cache: Dict[int, tuple[float, dict]] = {}

_SYSTEM = "Ты профессиональный футбольный аналитик. Отвечай ТОЛЬКО валидным JSON, без markdown и без текста вокруг."

# Допустимые пики по рынкам
_VALID_PICKS = {
    "1X2": {"HOME", "DRAW", "AWAY"},
    "TOTAL": {"OVER", "UNDER"},
    "HANDICAP": {"HOME", "AWAY"},
}


async def _db_cache_get(match_id: int) -> Optional[dict]:
    from datetime import datetime, timedelta
    from src.data.database import AiPrediction, SessionLocal

    cutoff = datetime.utcnow() - timedelta(seconds=_CACHE_TTL_SEC)
    try:
        async with SessionLocal() as session:
            row = await session.get(AiPrediction, match_id)
            if row is None or row.created_at < cutoff:
                return None
            return json.loads(row.payload)
    except Exception as e:
        logger.warning(f"ai db cache read failed for {match_id}: {e}")
        return None


async def _db_cache_put(match_id: int, payload: dict) -> None:
    from datetime import datetime
    from src.data.database import AiPrediction, SessionLocal

    try:
        async with SessionLocal() as session:
            existing = await session.get(AiPrediction, match_id)
            data = json.dumps(payload)
            if existing is None:
                session.add(AiPrediction(
                    match_id=match_id, created_at=datetime.utcnow(), payload=data
                ))
            else:
                existing.created_at = datetime.utcnow()
                existing.payload = data
            await session.commit()
    except Exception as e:
        logger.warning(f"ai db cache write failed for {match_id}: {e}")


_PROMPT = """Матч: {home} vs {away} ({competition}).

КОТИРОВКИ БУКМЕКЕРА (с рыночной no-vig вероятностью каждого исхода):
{odds_table}

СВЕЖИЕ ДАННЫЕ ИЗ СЕТИ (травмы, составы, мотивация):
{web_block}

ЗАДАЧА (пройди мысленно, в ответ — только итог):
1. Оцени СВОЮ истинную вероятность для КАЖДОЙ котировки выше.
2. Для каждой посчитай расхождение = (твоя вероятность) − (рыночная no-vig вероятность).
3. Выбери РОВНО ОДНУ котировку с МАКСИМАЛЬНЫМ ПОЛОЖИТЕЛЬНЫМ расхождением.
4. Ставь только если: расхождение положительное, кэф ≥ {min_odds}, твоя уверенность ≥ 0.56.
   Если подходящей ставки нет — верни confidence 0.50 (это значит «пропуск»).

Рынки и пики:
- 1X2: pick = HOME | DRAW | AWAY, line = null
- TOTAL: pick = OVER | UNDER, line = число (напр. 2.5)
- HANDICAP: pick = HOME | AWAY, line = гандикап хозяев (напр. -0.5)

Верни СТРОГО JSON:
{{"market": "1X2|TOTAL|HANDICAP", "pick": "...", "line": число или null, "confidence": float 0..1, "reasoning": "1-2 предложения почему есть расхождение"}}"""


def _format_web(results: list[dict]) -> str:
    if not results:
        return "(нет свежих данных)"
    lines = []
    for r in results[:5]:
        title = r.get("title", "").strip()
        content = r.get("content", "").strip()
        lines.append(f"• {title}: {content[:250]}")
    return "\n".join(lines)


def build_odds_table(odds: dict) -> str:
    """Render the bookmaker odds (all lines) + no-vig probabilities as text."""
    lines: list[str] = []
    ml = odds.get("ml")
    if ml and ml.get("p_home") is not None:
        lines.append(
            f"1X2: П1 {ml['home']:.2f} ({ml['p_home']:.0%}) | "
            f"X {ml['draw']:.2f} ({ml['p_draw']:.0%}) | "
            f"П2 {ml['away']:.2f} ({ml['p_away']:.0%})"
        )
    for t in odds.get("totals", []):
        if t.get("over") and t.get("under") and t.get("over_novig") is not None:
            lines.append(
                f"ТОТАЛ {t['point']}: Б {t['over']:.2f} ({t['over_novig']:.0%}) | "
                f"М {t['under']:.2f} ({t['under_novig']:.0%})"
            )
    for h in odds.get("handicaps", []):
        if h.get("home") and h.get("away") and h.get("home_novig") is not None:
            lines.append(
                f"ФОРА {h['point']:+}: П1 {h['home']:.2f} ({h['home_novig']:.0%}) | "
                f"П2 {h['away']:.2f} ({h['away_novig']:.0%})"
            )
    return "\n".join(lines) if lines else "(котировки недоступны)"


def _parse_json_strict(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _validate_pick(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    market = d.get("market")
    pick = d.get("pick")
    conf = d.get("confidence")
    if market not in _VALID_PICKS:
        logger.warning(f"ai_predict: bad market {market!r}")
        return False
    if pick not in _VALID_PICKS[market]:
        logger.warning(f"ai_predict: bad pick {pick!r} for {market}")
        return False
    if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
        logger.warning(f"ai_predict: bad confidence {conf!r}")
        return False
    line = d.get("line")
    if market == "1X2":
        d["line"] = None
    else:
        if not isinstance(line, (int, float)):
            logger.warning(f"ai_predict: {market} requires numeric line, got {line!r}")
            return False
        d["line"] = float(line)
    return True


async def ai_predict(
    *,
    match_id: int,
    home: str,
    away: str,
    competition: str,
    odds: dict,
) -> Optional[dict]:
    """Returns {market, pick, line, confidence, reasoning} or None.

    Claude picks one bet by maximum positive divergence vs the no-vig market.
    """
    now = time.time()
    cached = _cache.get(match_id)
    if cached and now - cached[0] < _CACHE_TTL_SEC:
        return cached[1]

    db_cached = await _db_cache_get(match_id)
    if db_cached is not None:
        _cache[match_id] = (now, db_cached)
        return db_cached

    web_results = await tavily_search(
        f"{home} vs {away} team news injuries lineup", days=7
    )
    prompt = _PROMPT.format(
        home=home, away=away, competition=competition,
        odds_table=build_odds_table(odds),
        web_block=_format_web(web_results),
        min_odds=settings.min_odds,
    )

    raw = await call_llm(prompt, max_tokens=700, system=_SYSTEM, temperature=0.0)
    if not raw:
        logger.warning(f"ai_predict({match_id}): empty LLM response")
        return None

    parsed = _parse_json_strict(raw)
    if not parsed or not _validate_pick(parsed):
        logger.warning(f"ai_predict({match_id}): invalid JSON/pick; raw={raw[:300]}")
        return None

    _cache[match_id] = (now, parsed)
    await _db_cache_put(match_id, parsed)
    logger.info(
        f"ai_predict({match_id}): {parsed['market']}/{parsed['pick']} "
        f"line={parsed['line']} conf={parsed['confidence']:.2f}"
    )
    return parsed
