"""End-to-end pipeline: ingest → odds → Claude analysis → emit signals."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger
from sqlalchemy import select

from src.ai.predictor import ai_predict
from src.bot.formatters import format_signal
from src.bot.handlers import broadcast_signal
from src.config import settings
from src.data.database import AiPrediction, Match, SessionLocal, Signal as SignalRow, Team
from src.data.football_api import FootballDataClient
from src.data.ingest import ingest_history, ingest_upcoming
from src.data.odds_api import OddsApiClient, fetch_odds_for_matches
from src.data.settings_store import get_bool
from src.signals.generator import Signal, _kelly
from src.signals.tracker import settle_pending


# Legacy ITB markets — kept only so the startup purge in main.py / handlers.py
# can still target old rows. Never generated anymore.
DISABLED_MARKETS: set[str] = {
    "HOME_OVER05", "HOME_OVER15", "HOME_OVER25",
    "AWAY_OVER05", "AWAY_OVER15", "AWAY_OVER25",
}


async def bootstrap_history(seasons: List[int] | None = None) -> int:
    """Initial load of historical data — call once after first deploy."""
    if seasons is None:
        # last three full seasons by default
        this_year = datetime.utcnow().year
        seasons = [this_year - 3, this_year - 2, this_year - 1]
    client = FootballDataClient()
    try:
        n = await ingest_history(client, settings.competitions, seasons)
        return n
    finally:
        await client.close()


async def refresh_upcoming(days: int = 7) -> int:
    client = FootballDataClient()
    try:
        return await ingest_upcoming(client, settings.competitions, days_ahead=days)
    finally:
        await client.close()


async def _store_signals(signals: List[Signal]) -> List[SignalRow]:
    """Persist signals — one per match (hard dedup across the whole table)."""
    stored: List[SignalRow] = []
    async with SessionLocal() as session:
        for s in signals:
            exists = (await session.execute(
                select(SignalRow).where(SignalRow.match_id == s.match_id)
            )).first()
            if exists is not None:
                continue
            row = SignalRow(
                match_id=s.match_id, market=s.market, pick=s.pick, line=s.line,
                is_ai_ensemble=True,
                model_prob=s.model_prob, fair_odds=s.fair_odds,
                book_odds=s.book_odds, edge=s.edge, confidence=s.confidence,
                stake_units=s.stake_units,
            )
            session.add(row)
            stored.append(row)
        await session.commit()
    return stored


async def generate_and_broadcast(bot, hours: int = 4) -> int:
    """AI-only generation. Claude sees real bookmaker lines (with no-vig market
    probabilities) for each upcoming match and picks one bet by max positive
    divergence. The admin '🧠 AI' toggle is the master on/off.
    """
    if not await get_bool("ai_ensemble_enabled", False):
        logger.info("Signal generation paused (AI toggle OFF)")
        return 0

    now = datetime.utcnow()
    horizon = now + timedelta(hours=hours)
    async with SessionLocal() as session:
        upcoming = list((await session.execute(
            select(Match).where(
                Match.status != "FINISHED",
                Match.utc_date >= now,
                Match.utc_date <= horizon,
            )
        )).scalars())
        if not upcoming:
            logger.info(f"No upcoming matches in the next {hours}h")
            return 0
        # Skip matches that already have a signal
        already = {
            int(m) for m in (await session.execute(
                select(SignalRow.match_id)
                .where(SignalRow.match_id.in_([m.id for m in upcoming]))
                .distinct()
            )).scalars().all()
        }
        match_meta: dict[int, tuple[str, str, str, datetime]] = {}
        for m in upcoming:
            if m.id in already:
                continue
            home = await session.get(Team, m.home_team_id)
            away = await session.get(Team, m.away_team_id)
            if home and away:
                match_meta[m.id] = (home.name, away.name, m.competition, m.utc_date)
    if already:
        logger.info(f"AI loop: skipping {len(already)} match(es) already signalled")
    if not match_meta:
        return 0

    # Pull real odds (all lines) for the candidate matches
    odds_map: dict[int, dict] = {}
    if settings.odds_api_key:
        odds_client = OddsApiClient(settings.odds_api_key)
        try:
            tuples = [
                (mid, comp, home, away, ko)
                for mid, (home, away, comp, ko) in match_meta.items()
            ]
            odds_map = await fetch_odds_for_matches(odds_client, tuples)
        finally:
            await odds_client.close()
        logger.info(f"Pulled odds for {len(odds_map)}/{len(match_meta)} matches")

    signals: List[Signal] = []
    for mid, (home_name, away_name, comp, _ko) in match_meta.items():
        odds = odds_map.get(mid)
        if not odds:
            continue  # no real lines → nothing to bet on
        ai = await ai_predict(
            match_id=mid, home=home_name, away=away_name, competition=comp, odds=odds,
        )
        if not ai:
            continue
        sig = _ai_to_signal(mid, ai, odds)
        if sig is not None:
            signals.append(sig)

    new_rows = await _store_signals(signals)
    sent = 0
    if new_rows and bot:
        async with SessionLocal() as session:
            for row in new_rows:
                match = await session.get(Match, row.match_id)
                if not match:
                    continue
                home = await session.get(Team, match.home_team_id)
                away = await session.get(Team, match.away_team_id)
                cached_ai = await session.get(AiPrediction, row.match_id)
                ai_comment = None
                if cached_ai:
                    try:
                        ai_comment = json.loads(cached_ai.payload).get("reasoning")
                    except Exception:
                        ai_comment = None
                if ai_comment:
                    stored = await session.get(SignalRow, row.id)
                    if stored:
                        stored.commentary = ai_comment
                        await session.commit()
                text = format_signal(row, match, home, away, ai_comment)
                sent += await broadcast_signal(bot, text)
    logger.info(
        f"AI loop: {len(match_meta)} analysed, {len(signals)} signals, {sent} broadcast"
    )
    return len(new_rows)


def _find_line_odds(ai: dict, odds: dict) -> Optional[tuple[float, float]]:
    """Return (book_odds, market_novig_prob) for Claude's chosen market/pick/line.

    None if the chosen line/pick is not present in the bookmaker data.
    """
    market, pick, line = ai["market"], ai["pick"], ai.get("line")
    if market == "1X2":
        ml = odds.get("ml")
        if not ml or ml.get("p_home") is None:
            return None
        side = {"HOME": ("home", "p_home"), "DRAW": ("draw", "p_draw"),
                "AWAY": ("away", "p_away")}[pick]
        o, p = ml.get(side[0]), ml.get(side[1])
        return (o, p) if o and p is not None else None
    if market == "TOTAL":
        for t in odds.get("totals", []):
            if line is not None and abs(t["point"] - line) < 0.01:
                if pick == "OVER" and t.get("over") and t.get("over_novig") is not None:
                    return t["over"], t["over_novig"]
                if pick == "UNDER" and t.get("under") and t.get("under_novig") is not None:
                    return t["under"], t["under_novig"]
        return None
    if market == "HANDICAP":
        for h in odds.get("handicaps", []):
            if line is not None and abs(h["point"] - line) < 0.01:
                if pick == "HOME" and h.get("home") and h.get("home_novig") is not None:
                    return h["home"], h["home_novig"]
                if pick == "AWAY" and h.get("away") and h.get("away_novig") is not None:
                    return h["away"], h["away_novig"]
        return None
    return None


def _ai_to_signal(match_id: int, ai: dict, odds: dict) -> Optional[Signal]:
    """Build a Signal from Claude's pick if it clears edge + odds thresholds.

    Edge = Claude confidence − market no-vig probability (NOT conf*odds-1).
    """
    conf = float(ai["confidence"])
    if conf < settings.min_confidence:
        # Claude is passing the match (it returns ~0.50 when no value)
        return None
    found = _find_line_odds(ai, odds)
    if not found:
        logger.info(f"_ai_to_signal({match_id}): chosen line not in book data — reject")
        return None
    book, market_prob = found
    edge = conf - market_prob
    if edge < settings.min_edge:
        return None
    if book < settings.min_odds or book > settings.max_odds:
        return None
    fair = 1.0 / max(conf, 1e-6)
    stake = _kelly(conf, book)
    return Signal(
        match_id=match_id,
        market=ai["market"], pick=ai["pick"], line=ai.get("line"),
        model_prob=conf, fair_odds=fair,
        book_odds=float(book), edge=float(edge),
        confidence=conf, stake_units=float(round(stake, 2)),
        is_value=True,
    )


async def daily_cycle(bot) -> None:
    """Refresh data, settle results, run AI analysis, broadcast."""
    logger.info("Daily cycle start")
    await refresh_upcoming(days=7)
    await settle_pending()
    await generate_and_broadcast(bot)


async def daily_stats_broadcast(bot) -> None:
    """Once a day: settle any late finishers, then push a stats digest
    (yesterday split by type + running totals) to EVERY allowed user,
    regardless of their notifications toggle (stats are not 'live spam')."""
    from src.bot.formatters import MSK_OFFSET, format_daily_digest, msk_now
    from src.bot.handlers import broadcast_signal
    from src.signals.tracker import RoiStats, roi_stats

    logger.info("Daily stats broadcast: start")
    await refresh_upcoming(days=2)
    settled_now = await settle_pending()
    if settled_now:
        logger.info(f"Daily stats broadcast: settled {settled_now} late signals")

    now_msk = msk_now()
    yesterday_msk_date = (now_msk - timedelta(days=1)).date()
    start_msk = datetime.combine(yesterday_msk_date, datetime.min.time())
    end_msk = start_msk + timedelta(days=1)
    since_utc = start_msk - MSK_OFFSET
    until_utc = end_msk - MSK_OFFSET

    async with SessionLocal() as session:
        rows = list((await session.execute(
            select(SignalRow).where(
                SignalRow.settled.is_(True),
                SignalRow.created_at >= since_utc,
                SignalRow.created_at < until_utc,
            )
        )).scalars())

    def _calc(subset: list) -> RoiStats:
        # Exclude pushes (won is None) from the ratios
        subset = [r for r in subset if r.won is not None]
        if not subset:
            return RoiStats(0, 0, 0, 0, 0, 0.0, 0.0)
        staked = sum(r.stake_units for r in subset)
        profit = sum(r.profit_units or 0.0 for r in subset)
        won = sum(1 for r in subset if r.won)
        returned = sum(
            r.stake_units * r.book_odds if (r.won and r.book_odds > 1.0) else 0.0
            for r in subset
        )
        return RoiStats(
            n_settled=len(subset), n_won=won,
            staked=staked, returned=returned, profit=profit,
            roi=(profit / staked * 100.0) if staked > 0 else 0.0,
            hit_rate=(won / len(subset) * 100.0) if subset else 0.0,
        )

    y_total = _calc(rows)
    y_by_market = {m: _calc([r for r in rows if r.market == m])
                   for m in ("1X2", "TOTAL", "HANDICAP")}

    total = await roi_stats()
    text = format_daily_digest(
        yesterday_total=y_total,
        yesterday_by_market=y_by_market,
        total=total,
        date_label=yesterday_msk_date.strftime("%d.%m.%Y"),
    )
    sent = await broadcast_signal(bot, text, respect_notifications=False)
    logger.info(f"Daily stats broadcast: sent to {sent} subscribers (notifications bypassed)")
    logger.info("Daily cycle done")
