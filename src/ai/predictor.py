"""AI-сторона ансамбля: Claude через NeuroAPI оценивает матч по веб-данным и
статистике, возвращает собственные вероятности. Используется в pipeline для
смешивания с XGBoost.
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, Optional

from loguru import logger

from src.ai.commentary import call_llm, call_llm_with_web_search
_CACHE_TTL_SEC = 12 * 3600
_cache: Dict[int, tuple[float, dict]] = {}


async def _db_cache_get(match_id: int) -> Optional[dict]:
    """Return cached AI prediction from DB if not expired."""
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


_PROMPT = """Ты — элитный футбольный аналитик. 15+ лет в лайв-анализе. Глубокое понимание xG (Opta, StatsBomb), математики линий. Ты ищешь РАСХОЖДЕНИЕ между истинной вероятностью и рынком.

ЦЕЛЬ: максимальный value в матче {home} vs {away} ({competition}).

ЖЁСТКОЕ ПРАВИЛО: взвешивай каждую цифру. Никакой отсебятины не из данных. Если данные противоречат — выбери весомый аргумент.

АЛГОРИТМ (7 ступеней):
1. ЦИФРОВОЙ ФУНДАМЕНТ: форма за 6 туров, xG/xGA портрет, фильтр «решающие матчи», стандарты, последние 15 минут.
2. КАДРОВАЯ ТЕКТОНИКА: критические потери (вратарь, бомбардир), глубина скамейки, изоляции на флангах.
3. H2H: только 2-3 года; если контекст изменился (мотивация, тренер, состав) — вес тренда −50%.
4. ФАКТОР X: судья (пенальти/ЖК), погода, дерби, психология. Мотивация: обычная (0%), повышенная (+5%), экстремальная (+10-15%). В плей-офф с минимальным преимуществом вероятность ничьи +10-15%.
5. РЫНОЧНЫЙ РАЗРЕЗ: вычисти маржу, сравни истинные вероятности с рыночными, найди расхождение ≥6%.
6. СБОРКА: соедини ступени 1-5, найди противоречия, проверь экстремальный сценарий («последний бой», слом H2H).
7. ИТОГ: ищем валуй >6% при кэфе ≥2.50; иначе >8% при ≥1.80; иначе >10% при ≥1.50.

ДАННЫЕ ОТ ML-МОДЕЛИ (XGBoost):
P(дом) = {p_home:.0%}, P(ничья) = {p_draw:.0%}, P(гости) = {p_away:.0%}
P(тотал>2.5) = {p_over25:.0%}, P(обе забьют) = {p_btts:.0%}
P(дом забьёт >0.5/1.5/2.5) = {p_home_over05:.0%} / {p_home_over15:.0%} / {p_home_over25:.0%}
P(гости забьют >0.5/1.5/2.5) = {p_away_over05:.0%} / {p_away_over15:.0%} / {p_away_over25:.0%}

ИСТОРИЧЕСКИЙ КОНТЕКСТ:
Elo {home_elo:.0f} vs {away_elo:.0f}
Форма (PPG за 10): {home_form:.2f} vs {away_form:.2f}

СВЕЖИЕ ДАННЫЕ: используй web_search чтобы найти за последние 7 дней:
travmы, дисквалификации, состав на ближайший матч, мотивацию (борьба за топ-4 / еврокубки / вылет), погоду, заявления тренеров. Делай не больше 2-3 поисков.

ЗАДАЧА: пройди все 7 ступеней мысленно. Сформируй СВОИ независимые вероятности на 11 рынках. Если данных мало — пометь предположения, но не уходи в фантазии.

Верни СТРОГО JSON, без markdown:
{{"p_home": float, "p_draw": float, "p_away": float, "p_over25": float, "p_btts": float, "p_home_over05": float, "p_home_over15": float, "p_home_over25": float, "p_away_over05": float, "p_away_over15": float, "p_away_over25": float, "reasoning": "1-2 предложения с главным расхождением"}}"""


def _parse_json_strict(raw: str) -> Optional[dict]:
    if not raw:
        return None
    # 1. Try direct
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    # 2. Try extracting from ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. Try first {...} block
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _validate_probs(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    for key in ("p_home", "p_draw", "p_away", "p_over25", "p_btts"):
        v = d.get(key)
        if not isinstance(v, (int, float)):
            return False
        if v < 0 or v > 1.0:
            return False
    s = d["p_home"] + d["p_draw"] + d["p_away"]
    if s <= 0:
        return False
    # Normalize 1x2 to sum to 1
    d["p_home"] = d["p_home"] / s
    d["p_draw"] = d["p_draw"] / s
    d["p_away"] = d["p_away"] / s
    # ITB probs are optional; if present, clamp to [0,1]; otherwise drop.
    for key in (
        "p_home_over05", "p_home_over15", "p_home_over25",
        "p_away_over05", "p_away_over15", "p_away_over25",
    ):
        v = d.get(key)
        if isinstance(v, (int, float)) and 0 <= v <= 1:
            continue
        d.pop(key, None)
    return True


async def ai_predict(
    *,
    match_id: int,
    home: str,
    away: str,
    competition: str,
    ml_probs: dict,
    features: dict,
) -> Optional[dict]:
    """Returns {p_home, p_draw, p_away, p_over25, p_btts, reasoning} or None."""
    now = time.time()
    cached = _cache.get(match_id)
    if cached and now - cached[0] < _CACHE_TTL_SEC:
        return cached[1]

    db_cached = await _db_cache_get(match_id)
    if db_cached is not None:
        _cache[match_id] = (now, db_cached)
        return db_cached

    prompt = _PROMPT.format(
        home=home,
        away=away,
        competition=competition,
        p_home=ml_probs.get("p_home", 0.33),
        p_draw=ml_probs.get("p_draw", 0.33),
        p_away=ml_probs.get("p_away", 0.33),
        p_over25=ml_probs.get("p_over25", 0.5),
        p_btts=ml_probs.get("p_btts", 0.5),
        p_home_over05=ml_probs.get("p_home_over05", 0.7),
        p_home_over15=ml_probs.get("p_home_over15", 0.4),
        p_home_over25=ml_probs.get("p_home_over25", 0.2),
        p_away_over05=ml_probs.get("p_away_over05", 0.6),
        p_away_over15=ml_probs.get("p_away_over15", 0.3),
        p_away_over25=ml_probs.get("p_away_over25", 0.15),
        home_elo=features.get("home_elo", 1500),
        away_elo=features.get("away_elo", 1500),
        home_form=features.get("home_form_pts", 1.5),
        away_form=features.get("away_form_pts", 1.5),
    )

    raw = await call_llm_with_web_search(prompt, max_tokens=1500)
    if not raw:
        # Fall back to plain LLM call (no web search) — e.g. if no direct
        # Anthropic key is configured and only a proxy is available.
        raw = await call_llm(prompt, max_tokens=900)
    if not raw:
        logger.warning(f"ai_predict({match_id}): empty LLM response")
        return None

    parsed = _parse_json_strict(raw)
    if not parsed or not _validate_probs(parsed):
        logger.warning(f"ai_predict({match_id}): invalid JSON or probs: {raw[:200]}")
        return None

    _cache[match_id] = (now, parsed)
    await _db_cache_put(match_id, parsed)
    logger.info(
        f"ai_predict({match_id}): home={parsed['p_home']:.2f} "
        f"draw={parsed['p_draw']:.2f} away={parsed['p_away']:.2f} "
        f"ou={parsed['p_over25']:.2f} btts={parsed['p_btts']:.2f}"
    )
    return parsed
