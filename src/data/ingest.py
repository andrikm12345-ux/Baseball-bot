from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

from loguru import logger
from sqlalchemy import select

from src.data.database import Match, SessionLocal, Team
from src.data.football_api import FootballDataClient


def _parse_dt(s: str) -> datetime:
    # football-data uses e.g. "2024-08-16T19:00:00Z"
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


async def _upsert_team(session, team_data: dict) -> int:
    tid = int(team_data["id"])
    existing = await session.get(Team, tid)
    if existing is None:
        session.add(
            Team(
                id=tid,
                name=team_data.get("name") or f"Team {tid}",
                short_name=team_data.get("shortName"),
            )
        )
    return tid


async def store_matches(matches: Iterable[dict], competition: str) -> int:
    """Upsert matches into DB. Returns count written/updated."""
    count = 0
    async with SessionLocal() as session:
        for m in matches:
            try:
                home = m.get("homeTeam") or {}
                away = m.get("awayTeam") or {}
                if not home.get("id") or not away.get("id"):
                    continue
                home_id = await _upsert_team(session, home)
                away_id = await _upsert_team(session, away)
                score = (m.get("score") or {}).get("fullTime") or {}
                match_id = int(m["id"])
                existing = await session.get(Match, match_id)
                if existing is None:
                    existing = Match(id=match_id)
                    session.add(existing)
                existing.competition = competition
                existing.season = int((m.get("season") or {}).get("startDate", "0-0-0")[:4] or 0)
                existing.utc_date = _parse_dt(m["utcDate"])
                existing.status = m.get("status", "SCHEDULED")
                existing.home_team_id = home_id
                existing.away_team_id = away_id
                existing.home_goals = score.get("home")
                existing.away_goals = score.get("away")
                count += 1
            except Exception as e:
                logger.warning(f"Skip match: {e}")
        await session.commit()
    return count


async def ingest_history(client: FootballDataClient, competitions: List[str], seasons: List[int]) -> int:
    total = 0
    for c in competitions:
        matches = await client.fetch_finished_history(c, seasons)
        n = await store_matches(matches, c)
        total += n
        logger.info(f"Stored {n} matches for {c}")
    return total


async def ingest_upcoming(client: FootballDataClient, competitions: List[str], days_ahead: int = 7) -> int:
    total = 0
    for c in competitions:
        matches = await client.fetch_upcoming(c, days_ahead=days_ahead)
        n = await store_matches(matches, c)
        total += n
    return total
