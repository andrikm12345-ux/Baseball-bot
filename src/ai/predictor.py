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

from src.ai.commentary import call_llm
from src.data.web_search import tavily_search


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

ЖЁСТКИЕ ПРАВИЛА:
- ОБЯЗАН пройти ВСЕ 7 ступеней. Каждая ступень = отдельное поле в JSON-ответе с КОНКРЕТНЫМ выводом по данным, а не общими словами.
- Никакой отсебятины: каждое утверждение должно опираться либо на цифры из блока ДАННЫЕ ОТ ML, либо на блок СВЕЖИЕ ДАННЫЕ. Если в блоках инфы нет — пиши "данных недостаточно" по этой ступени, НЕ выдумывай.
- Если данные противоречат — выбери весомый аргумент и явно укажи противоречие.
- Ответ без любой из ступеней = НЕВАЛИДНЫЙ, лучше совсем не отвечать.

АЛГОРИТМ (7 ступеней):
1. ЦИФРОВОЙ ФУНДАМЕНТ: форма за 6 туров, xG/xGA портрет, фильтр «решающие матчи», стандарты, последние 15 минут.
2. КАДРОВАЯ ТЕКТОНИКА: критические потери (вратарь, бомбардир), глубина скамейки, изоляции на флангах.
3. H2H: только 2-3 года; если контекст изменился (мотивация, тренер, состав) — вес тренда −50%.
4. ФАКТОР X: судья (пенальти/ЖК), погода, дерби, психология. Мотивация: обычная (0%), повышенная (+5%), экстремальная (+10-15%). В плей-офф с минимальным преимуществом вероятность ничьи +10-15%.
5. РЫНОЧНЫЙ РАЗРЕЗ: сравни своё ощущение с ML-вероятностями, найди расхождение ≥6%.
6. СБОРКА: соедини ступени 1-5, найди противоречия, проверь экстремальный сценарий («последний бой», слом H2H).
7. ИТОГ: финальные вероятности по всем 11 рынкам. Если data sufficient — даёшь конкретные числа; если нет — повторяешь ML с минимальной коррекцией.

ДАННЫЕ ОТ ML-МОДЕЛИ (XGBoost):
P(дом) = {p_home:.0%}, P(ничья) = {p_draw:.0%}, P(гости) = {p_away:.0%}
P(тотал>2.5) = {p_over25:.0%}, P(обе забьют) = {p_btts:.0%}
P(дом забьёт >0.5/1.5/2.5) = {p_home_over05:.0%} / {p_home_over15:.0%} / {p_home_over25:.0%}
P(гости забьют >0.5/1.5/2.5) = {p_away_over05:.0%} / {p_away_over15:.0%} / {p_away_over25:.0%}

ИСТОРИЧЕСКИЙ КОНТЕКСТ:
Elo {home_elo:.0f} vs {away_elo:.0f}
Форма (PPG за 10): {home_form:.2f} vs {away_form:.2f}

СВЕЖИЕ ДАННЫЕ ИЗ СЕТИ:
{web_block}

Верни СТРОГО JSON, без markdown и без текста до или после JSON:
{{"step1_numerics": "1-2 предложения по форме/xG/решающим", "step2_squad": "1-2 предложения по составу и потерям", "step3_h2h": "1-2 предложения по очным", "step4_factor_x": "1-2 предложения по судье/погоде/мотивации", "step5_market_gap": "1-2 предложения о расхождении со своей оценкой", "step6_synthesis": "1-2 предложения о противоречиях и финальной логике", "step7_verdict": "главный пик и почему", "p_home": float, "p_draw": float, "p_away": float, "p_over25": float, "p_btts": float, "p_home_over05": float, "p_home_over15": float, "p_home_over25": float, "p_away_over05": float, "p_away_over15": float, "p_away_over25": float, "reasoning": "1-2 предложения с главным расхождением для показа в TG"}}"""


def _format_web(results: list[dict]) -> str:
    if not results:
        return "(нет свежих данных)"
    lines = []
    for r in results[:5]:
        title = r.get("title", "").strip()
        content = r.get("content", "").strip()
        lines.append(f"• {title}: {content[:250]}")
    return "\n".join(lines)


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


_REQUIRED_STEPS = (
    "step1_numerics", "step2_squad", "step3_h2h", "step4_factor_x",
    "step5_market_gap", "step6_synthesis", "step7_verdict",
)


def _validate_probs(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    # Each of the 7 reasoning steps must be present and non-empty — that's the
    # hard contract with the model. Skipping a step = invalid answer.
    for step in _REQUIRED_STEPS:
        v = d.get(step)
        if not isinstance(v, str) or not v.strip():
            logger.warning(f"ai_predict: missing/empty step '{step}'")
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

    web_results = await tavily_search(
        f"{home} vs {away} team news injuries lineup", days=7
    )
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
        web_block=_format_web(web_results),
    )

    # 7-step JSON ≈ 250-400 output tokens above the probability block. Headroom
    # at 1500 prevents truncation mid-JSON that would void validation.
    raw = await call_llm(prompt, max_tokens=1500)
    if not raw:
        logger.warning(f"ai_predict({match_id}): empty LLM response")
        return None

    parsed = _parse_json_strict(raw)
    if not parsed or not _validate_probs(parsed):
        logger.warning(f"ai_predict({match_id}): invalid JSON or missing steps; raw={raw[:300]}")
        return None
    # Trace the chain-of-thought into logs so we can audit whether Claude
    # actually went through every step or just paraphrased.
    logger.debug(
        f"ai_predict({match_id}) steps:\n"
        + "\n".join(f"  · {k}: {parsed.get(k, '')[:160]}" for k in _REQUIRED_STEPS)
    )

    _cache[match_id] = (now, parsed)
    await _db_cache_put(match_id, parsed)
    logger.info(
        f"ai_predict({match_id}): home={parsed['p_home']:.2f} "
        f"draw={parsed['p_draw']:.2f} away={parsed['p_away']:.2f} "
        f"ou={parsed['p_over25']:.2f} btts={parsed['p_btts']:.2f}"
    )
    return parsed
