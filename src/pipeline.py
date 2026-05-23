"""End-to-end pipeline: ingest → features → train → predict → emit signals."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pandas as pd
from loguru import logger
from sqlalchemy import select

from src.ai.commentary import generate_commentary
from src.ai.predictor import ai_predict
from src.bot.formatters import format_signal
from src.bot.handlers import broadcast_signal
from src.config import settings
from src.data.database import Match, SessionLocal, Signal as SignalRow, Team
from src.data.features import build_features, build_inference_features
from src.data.football_api import FootballDataClient
from src.data.ingest import ingest_history, ingest_upcoming
from src.data.odds_api import OddsApiClient, fetch_odds_for_matches
from src.data.settings_store import get_bool
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


async def train_models(bot=None) -> None:
    df = await _load_matches_df()
    if df.empty:
        logger.warning("No matches in DB — skip training")
        return
    finished = df[df["status"] == "FINISHED"].copy()
    if len(finished) < 200:
        logger.warning(f"Only {len(finished)} finished matches — not training yet")
        return
    features = build_features(finished)
    result = train_all(features)
    logger.info(f"Models saved: {result['paths']}")
    if bot:
        await _notify_admins_training(bot, result["metrics"])


async def _notify_admins_training(bot, metrics: dict) -> None:
    from src.bot.formatters import format_training_report
    text = format_training_report(metrics)
    for admin_id in settings.admin_ids:
        try:
            await bot.send_message(admin_id, text, parse_mode="HTML")
        except Exception as e:
            logger.warning(f"train notify to {admin_id} failed: {e}")


DISABLED_MARKETS: set[str] = {
    "HOME_OVER05", "HOME_OVER15", "HOME_OVER25",
    "AWAY_OVER05", "AWAY_OVER15", "AWAY_OVER25",
}


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
    preds = predictor.predict(feats)

    # Attach bookmaker odds if we have an Odds API key
    if settings.odds_api_key:
        odds_client = OddsApiClient(settings.odds_api_key)
        try:
            async with SessionLocal() as session:
                tuples = []
                for mid in preds["match_id"].tolist():
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
        for col in [
            "odds_home", "odds_draw", "odds_away",
            "odds_over25", "odds_under25",
            "odds_btts_yes", "odds_btts_no",
        ]:
            preds[col] = preds["match_id"].map(lambda m: (odds_map.get(int(m)) or {}).get(col, 0.0))
        logger.info(f"Attached odds to {sum(1 for v in odds_map.values() if v)}/{len(preds)} matches")

    ai_match_ids: set[int] = set()
    if await get_bool("ai_ensemble_enabled", False):
        preds, ai_match_ids = await _apply_ai_ensemble(preds, feats)

    preds["_ai_applied"] = preds["match_id"].isin(ai_match_ids)
    signals = generate(preds)
    new_rows = await _store_signals(signals, ai_match_ids=ai_match_ids)
    sent = 0
    ai_on = await get_bool("ai_ensemble_enabled", False)
    if new_rows and bot:
        feats_by_id = {int(r["match_id"]): r.to_dict() for _, r in feats.iterrows()}
        async with SessionLocal() as session:
            for row in new_rows:
                match = await session.get(Match, row.match_id)
                if not match:
                    continue
                home = await session.get(Team, match.home_team_id)
                away = await session.get(Team, match.away_team_id)
                ai_comment = None
                if ai_on:
                    ai_comment = await generate_commentary(
                        match_id=row.match_id,
                        home=home.name, away=away.name,
                        competition=match.competition,
                        market=row.market, pick=row.pick,
                        prob=row.model_prob,
                        book_odds=row.book_odds, edge=row.edge,
                        features=feats_by_id.get(row.match_id, {}),
                    )
                    if ai_comment:
                        stored = await session.get(SignalRow, row.id)
                        if stored:
                            stored.commentary = ai_comment
                            await session.commit()
                text = format_signal(row, match, home, away, ai_comment)
                sent += await broadcast_signal(bot, text)
    logger.info(
        f"Generated {len(new_rows)} new signals, broadcast {sent} messages "
        f"(AI={'on' if ai_on else 'off'})"
    )
    return len(new_rows)


async def _apply_ai_ensemble(
    preds: pd.DataFrame, feats: pd.DataFrame
) -> tuple[pd.DataFrame, set[int]]:
    """Blend XGBoost preds with AI predictor for top-N most confident matches.

    Returns (modified_preds, set_of_match_ids_actually_corrected_by_ai).
    """
    import asyncio

    weight = settings.ai_ensemble_weight
    top_n = settings.ai_ensemble_top_n

    feats_by_id = {int(r["match_id"]): r.to_dict() for _, r in feats.iterrows()}
    preds = preds.copy()
    threshold = settings.ai_ensemble_min_prob
    preds["_max_1x2"] = preds[["p_home", "p_draw", "p_away"]].max(axis=1)
    candidates = (
        preds[preds["_max_1x2"] >= threshold]
        .sort_values("_max_1x2", ascending=False)
        .head(top_n)
    )
    preds.drop(columns=["_max_1x2"], inplace=True)

    if candidates.empty:
        return preds, set()

    async with SessionLocal() as session:
        # Skip matches that already have any signal stored — no need to spend
        # another AI call to re-evaluate something we've already published.
        existing_rows = (await session.execute(
            select(SignalRow.match_id)
            .where(SignalRow.match_id.in_([int(m) for m in candidates["match_id"]]))
            .distinct()
        )).scalars().all()
        already_signaled = {int(m) for m in existing_rows}
        if already_signaled:
            logger.info(
                f"AI ensemble: skipping {len(already_signaled)} match(es) that already have signals"
            )
        tasks = []
        match_meta = {}
        for _, row in candidates.iterrows():
            mid = int(row["match_id"])
            if mid in already_signaled:
                continue
            match = await session.get(Match, mid)
            if not match:
                continue
            home = await session.get(Team, match.home_team_id)
            away = await session.get(Team, match.away_team_id)
            if not home or not away:
                continue
            match_meta[mid] = (home.name, away.name, match.competition)
            tasks.append((mid, row))

    sem = asyncio.Semaphore(3)

    async def _one(mid: int, row) -> tuple[int, dict | None]:
        async with sem:
            ml_probs = {
                "p_home": float(row["p_home"]),
                "p_draw": float(row["p_draw"]),
                "p_away": float(row["p_away"]),
                "p_over25": float(row["p_over25"]),
                "p_btts": float(row["p_btts"]),
            }
            for col in (
                "p_home_over05", "p_home_over15", "p_home_over25",
                "p_away_over05", "p_away_over15", "p_away_over25",
            ):
                if col in row and pd.notna(row[col]):
                    ml_probs[col] = float(row[col])
            home, away, comp = match_meta[mid]
            ai = await ai_predict(
                match_id=mid,
                home=home, away=away, competition=comp,
                ml_probs=ml_probs,
                features=feats_by_id.get(mid, {}),
            )
            return mid, ai

    results = await asyncio.gather(*[_one(mid, r) for mid, r in tasks])

    applied: set[int] = set()
    diffs = []
    for mid, ai in results:
        if not ai:
            continue
        mask = preds["match_id"] == mid
        for col in ("p_home", "p_draw", "p_away", "p_over25", "p_btts"):
            ml_v = float(preds.loc[mask, col].iloc[0])
            ai_v = float(ai[col])
            diffs.append(abs(ml_v - ai_v))
            preds.loc[mask, col] = (1 - weight) * ml_v + weight * ai_v
        for col in (
            "p_home_over05", "p_home_over15", "p_home_over25",
            "p_away_over05", "p_away_over15", "p_away_over25",
        ):
            if col not in ai or col not in preds.columns:
                continue
            cur = preds.loc[mask, col].iloc[0]
            if pd.isna(cur):
                continue
            ml_v = float(cur)
            ai_v = float(ai[col])
            diffs.append(abs(ml_v - ai_v))
            preds.loc[mask, col] = (1 - weight) * ml_v + weight * ai_v
        applied.add(mid)

    if applied:
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        logger.info(
            f"AI ensemble applied to {len(applied)}/{len(tasks)} matches "
            f"(weight={weight}, avg |ml-ai| = {avg_diff:.3f})"
        )
    return preds, applied


async def daily_cycle(bot) -> None:
    """Refresh data, train if needed, generate signals, settle results."""
    logger.info("Daily cycle start")
    await refresh_upcoming(days=7)
    await settle_pending()
    await train_models(bot=bot)
    await generate_and_broadcast(bot)
    logger.info("Daily cycle done")
