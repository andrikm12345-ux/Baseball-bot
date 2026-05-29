"""End-to-end pipeline: ingest → odds → Claude analysis → emit signals."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import select

from src.ai.predictor import ai_predict
from src.bot.formatters import format_signal
from src.bot.handlers import broadcast_signal
from src.config import settings
from src.data.database import AiPrediction, Match, SessionLocal, Signal as SignalRow, Team
from src.data.features import build_inference_features
from src.data.football_api import FootballDataClient
from src.data.ingest import ingest_history, ingest_upcoming
from src.data.odds_api import OddsApiClient, fetch_odds_for_matches
from src.signals.generator import Signal
from src.signals.tracker import settle_pending


# ITB markets are never generated in AI-only mode, but the constant is still
# referenced by the legacy-signal purge in main.py / handlers.py.
DISABLED_MARKETS: set[str] = {
    "HOME_OVER05", "HOME_OVER15", "HOME_OVER25",
    "AWAY_OVER05", "AWAY_OVER15", "AWAY_OVER25",
}


async def _load_matches_df() -> pd.DataFrame:
    async with SessionLocal() as session:
        rows = (await session.execute(select(Match))).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "id": m.id,
            "utc_date": m.utc_date,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "competition": m.competition,
            "status": m.status,
        }
        for m in rows
    ])


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


async def _store_signals(
    signals: List[Signal], ai_match_ids: set[int] | None = None
) -> List[SignalRow]:
    """Persist signals, de-duplicating by (match, market, pick)."""
    stored: List[SignalRow] = []
    ai_ids = ai_match_ids or set()
    async with SessionLocal() as session:
        for s in signals:
            if s.market in DISABLED_MARKETS:
                continue
            # Hard dedup: one signal per match across the whole signals table.
            # The earlier (match, market, pick) tuple let us publish two bets
            # on the same fixture (e.g. 1X2 first hour, OU25 the next) which
            # the user pushed back on as 'placing two bets on one game'.
            exists = (await session.execute(
                select(SignalRow).where(SignalRow.match_id == s.match_id)
            )).first()
            if exists is not None:
                continue
            row = SignalRow(
                match_id=s.match_id, market=s.market, pick=s.pick,
                is_ai_ensemble=(s.match_id in ai_ids),
                model_prob=s.model_prob, fair_odds=s.fair_odds,
                book_odds=s.book_odds, edge=s.edge, confidence=s.confidence,
                stake_units=s.stake_units,
            )
            session.add(row)
            stored.append(row)
        await session.commit()
    return stored


async def generate_and_broadcast(bot) -> int:
    """AI-only signal generation. Claude analyses every upcoming match;
    we publish if the recommended pick has bookmaker odds >= settings.min_odds.
    The admin '🧠 AI' toggle acts as the master on/off for signal generation.
    """
    from src.data.settings_store import get_bool
    if not await get_bool("ai_ensemble_enabled", False):
        logger.info("Signal generation paused (AI toggle OFF)")
        return 0
    df = await _load_matches_df()
    if df.empty:
        return 0
    finished = df[df["status"] == "FINISHED"].copy()
    now = datetime.utcnow()
    horizon = now + timedelta(hours=4)
    upcoming = df[
        (df["status"] != "FINISHED")
        & (df["utc_date"] >= now)
        & (df["utc_date"] <= horizon)
    ].copy()
    if upcoming.empty:
        logger.info("No upcoming matches in the next 4 hours")
        return 0
    feats = build_inference_features(upcoming, finished)
    if feats.empty:
        return 0

    # Pull odds for all candidate matches in one batch
    odds_map: dict[int, dict] = {}
    if settings.odds_api_key:
        odds_client = OddsApiClient(settings.odds_api_key)
        try:
            async with SessionLocal() as session:
                tuples = []
                for mid in feats["match_id"].tolist():
                    match = await session.get(Match, int(mid))
                    if not match:
                        continue
                    home = await session.get(Team, match.home_team_id)
                    away = await session.get(Team, match.away_team_id)
                    if home and away:
                        tuples.append((match.id, match.competition, home.name, away.name, match.utc_date))
            odds_map = await fetch_odds_for_matches(odds_client, tuples)
        finally:
            await odds_client.close()
        logger.info(f"Pulled odds for {sum(1 for v in odds_map.values() if v)}/{len(feats)} matches")

    # Skip matches we already published any signal on
    async with SessionLocal() as session:
        already = {
            int(m) for m in (await session.execute(
                select(SignalRow.match_id)
                .where(SignalRow.match_id.in_([int(x) for x in feats["match_id"]]))
                .distinct()
            )).scalars().all()
        }
        match_meta: dict[int, tuple[str, str, str]] = {}
        for mid in feats["match_id"].tolist():
            if int(mid) in already:
                continue
            match = await session.get(Match, int(mid))
            if not match:
                continue
            home = await session.get(Team, match.home_team_id)
            away = await session.get(Team, match.away_team_id)
            if home and away:
                match_meta[int(mid)] = (home.name, away.name, match.competition)
    if already:
        logger.info(f"AI loop: skipping {len(already)} match(es) already signalled")

    signals: List[Signal] = []
    feats_by_id = {int(r["match_id"]): r.to_dict() for _, r in feats.iterrows()}
    for mid, (home_name, away_name, comp) in match_meta.items():
        ai = await ai_predict(
            match_id=mid, home=home_name, away=away_name, competition=comp,
            features=feats_by_id.get(mid, {}),
        )
        if not ai:
            continue
        sig = _ai_to_signal(mid, ai, odds_map.get(mid) or {})
        if sig is not None:
            signals.append(sig)

    new_rows = await _store_signals(signals, ai_match_ids={s.match_id for s in signals})
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
        f"AI loop: {len(match_meta)} matches analysed, "
        f"{len(signals)} signals, {sent} broadcast"
    )
    return len(new_rows)


def _ai_to_signal(match_id: int, ai: dict, odds: dict) -> Optional[Signal]:
    """Pick the best market for a match given Claude's probabilities and the
    bookmaker line. Requires book odds >= settings.min_odds; otherwise None.
    """
    candidates = []
    p_home, p_draw, p_away = ai.get("p_home", 0), ai.get("p_draw", 0), ai.get("p_away", 0)
    best_1x2 = max(
        [("HOME", p_home, odds.get("odds_home", 0.0)),
         ("DRAW", p_draw, odds.get("odds_draw", 0.0)),
         ("AWAY", p_away, odds.get("odds_away", 0.0))],
        key=lambda x: x[1],
    )
    candidates.append(("1X2", *best_1x2))

    p_over = ai.get("p_over25", 0.5)
    if p_over >= 0.5:
        candidates.append(("OU25", "OVER", p_over, odds.get("odds_over25", 0.0)))
    else:
        candidates.append(("OU25", "UNDER", 1 - p_over, odds.get("odds_under25", 0.0)))

    p_btts = ai.get("p_btts", 0.5)
    if p_btts >= 0.5:
        candidates.append(("BTTS", "YES", p_btts, odds.get("odds_btts_yes", 0.0)))
    else:
        candidates.append(("BTTS", "NO", 1 - p_btts, odds.get("odds_btts_no", 0.0)))

    # Require a valid bookmaker line at min_odds or above
    valid = [c for c in candidates if c[3] and c[3] >= settings.min_odds and c[3] <= settings.max_odds]
    if not valid:
        return None
    # No value/edge logic — pick the market Claude is MOST CONFIDENT in among
    # those with acceptable odds. Edge is still computed for the stats display.
    market, pick, prob, book = max(valid, key=lambda c: c[2])
    edge = prob * book - 1.0
    fair = 1.0 / max(prob, 1e-6)
    from src.signals.generator import _kelly
    stake = _kelly(prob, book)
    return Signal(
        match_id=match_id,
        market=market, pick=pick,
        model_prob=float(prob), fair_odds=float(fair),
        book_odds=float(book), edge=float(edge),
        confidence=float(prob), stake_units=float(round(stake, 2)),
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
    y_model = _calc([r for r in rows if not r.book_odds or r.book_odds <= 1.0])
    y_value = _calc([r for r in rows if r.book_odds and r.book_odds > 1.0])
    y_ai = _calc([r for r in rows if getattr(r, "is_ai_ensemble", False)])

    total = await roi_stats(only_value=None)
    text = format_daily_digest(
        yesterday_total=y_total,
        yesterday_model=y_model,
        yesterday_value=y_value,
        yesterday_ai=y_ai,
        total=total,
        date_label=yesterday_msk_date.strftime("%d.%m.%Y"),
    )
    sent = await broadcast_signal(bot, text, respect_notifications=False)
    logger.info(f"Daily stats broadcast: sent to {sent} subscribers (notifications bypassed)")
    logger.info("Daily cycle done")
