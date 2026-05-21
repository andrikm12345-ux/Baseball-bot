"""End-to-end pipeline: ingest → features → train → predict → emit signals."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pandas as pd
from loguru import logger
from sqlalchemy import select

from src.bot.formatters import format_signal
from src.bot.handlers import broadcast_signal
from src.config import settings
from src.data.database import Match, SessionLocal, Signal as SignalRow, Team
from src.data.features import build_features, build_inference_features
from src.data.football_api import FootballDataClient
from src.data.ingest import ingest_history, ingest_upcoming
from src.ml.predict import Predictor
from src.ml.train import train_all
from src.signals.generator import Signal, generate
from src.signals.tracker import settle_pending


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


async def train_models() -> None:
    df = await _load_matches_df()
    if df.empty:
        logger.warning("No matches in DB — skip training")
        return
    finished = df[df["status"] == "FINISHED"].copy()
    if len(finished) < 200:
        logger.warning(f"Only {len(finished)} finished matches — not training yet")
        return
    features = build_features(finished)
    paths = train_all(features)
    logger.info(f"Models saved: {paths}")


async def _store_signals(signals: List[Signal]) -> List[SignalRow]:
    """Persist signals, de-duplicating by (match, market, pick)."""
    stored: List[SignalRow] = []
    async with SessionLocal() as session:
        for s in signals:
            exists = (await session.execute(
                select(SignalRow).where(
                    SignalRow.match_id == s.match_id,
                    SignalRow.market == s.market,
                    SignalRow.pick == s.pick,
                )
            )).scalar_one_or_none()
            if exists is not None:
                continue
            row = SignalRow(
                match_id=s.match_id, market=s.market, pick=s.pick,
                model_prob=s.model_prob, fair_odds=s.fair_odds,
                book_odds=s.book_odds, edge=s.edge, confidence=s.confidence,
                stake_units=s.stake_units,
            )
            session.add(row)
            stored.append(row)
        await session.commit()
    return stored


async def generate_and_broadcast(bot) -> int:
    """Generate signals for upcoming matches and broadcast new ones."""
    predictor = Predictor()
    if not predictor.ready:
        logger.warning("Models not ready — skip signal generation")
        return 0
    df = await _load_matches_df()
    if df.empty:
        return 0
    finished = df[df["status"] == "FINISHED"].copy()
    now = datetime.utcnow()
    horizon = now + timedelta(days=7)
    upcoming = df[
        (df["status"] != "FINISHED")
        & (df["utc_date"] >= now)
        & (df["utc_date"] <= horizon)
    ].copy()
    if upcoming.empty:
        logger.info("No upcoming matches in the next 7 days")
        return 0
    feats = build_inference_features(upcoming, finished)
    if feats.empty:
        return 0
    preds = predictor.predict(feats)
    signals = generate(preds)
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
                text = format_signal(row, match, home, away)
                sent += await broadcast_signal(bot, text)
    logger.info(f"Generated {len(new_rows)} new signals, broadcast {sent} messages")
    return len(new_rows)


async def daily_cycle(bot) -> None:
    """Refresh data, train if needed, generate signals, settle results."""
    logger.info("Daily cycle start")
    await refresh_upcoming(days=7)
    await settle_pending()
    await train_models()
    await generate_and_broadcast(bot)
    logger.info("Daily cycle done")
