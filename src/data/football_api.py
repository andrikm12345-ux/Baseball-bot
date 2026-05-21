from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings


BASE_URL = "https://api.football-data.org/v4"


class FootballDataClient:
    """Thin async client for football-data.org v4.

    Free tier: 10 requests/minute, top competitions only. We back off automatically
    when we hit 429 so we don't get banned.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.football_data_api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Semaphore(1)
        self._min_interval = 6.5  # seconds between requests on free tier
        self._last_call = 0.0

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with self._lock:
            # crude rate limit for free tier
            now = asyncio.get_event_loop().time()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            session = await self._session_get()
            url = f"{BASE_URL}{path}"
            async with session.get(url, params=params, timeout=30) as r:
                self._last_call = asyncio.get_event_loop().time()
                if r.status == 429:
                    logger.warning("Rate-limited by football-data.org, backing off")
                    await asyncio.sleep(60)
                    raise aiohttp.ClientResponseError(
                        r.request_info, r.history, status=429, message="rate limited"
                    )
                r.raise_for_status()
                return await r.json()

    async def competition_matches(
        self,
        competition: str,
        season: Optional[int] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if season is not None:
            params["season"] = season
        if status:
            params["status"] = status
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        data = await self._get(f"/competitions/{competition}/matches", params=params)
        return data.get("matches", [])

    async def fetch_finished_history(
        self, competition: str, seasons: List[int]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in seasons:
            try:
                matches = await self.competition_matches(competition, season=s, status="FINISHED")
                logger.info(f"{competition} {s}: {len(matches)} finished matches")
                out.extend(matches)
            except Exception as e:
                logger.warning(f"Failed {competition} season {s}: {e}")
        return out

    async def fetch_upcoming(self, competition: str, days_ahead: int = 7) -> List[Dict[str, Any]]:
        today = datetime.now(timezone.utc).date()
        date_from = today.isoformat()
        from datetime import timedelta
        date_to = (today + timedelta(days=days_ahead)).isoformat()
        try:
            return await self.competition_matches(
                competition, status="SCHEDULED", date_from=date_from, date_to=date_to
            )
        except Exception as e:
            logger.warning(f"Failed upcoming {competition}: {e}")
            return []
