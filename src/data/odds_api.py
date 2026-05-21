"""Client for the-odds-api.com — fetches live bookmaker odds and matches them
to our DB matches by team name + kickoff time."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


BASE_URL = "https://api.the-odds-api.com/v4"


COMPETITION_TO_SPORT = {
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "SA": "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
    "CL": "soccer_uefa_champs_league",
    "PPL": "soccer_portugal_primeira_liga",
    "DED": "soccer_netherlands_eredivisie",
    "ELC": "soccer_efl_champ",
}


class OddsApiClient:
    """Free tier: 500 req/month — call sparingly. Cache results in memory for 1h."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = 3600.0

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    async def _get(self, path: str, params: Dict[str, Any]) -> Any:
        cache_key = f"{path}?{sorted(params.items())}"
        now = asyncio.get_event_loop().time()
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return data
        session = await self._session_get()
        params = {**params, "apiKey": self.api_key}
        async with session.get(f"{BASE_URL}{path}", params=params, timeout=20) as r:
            if r.status == 429:
                logger.warning("Odds API rate limit hit")
                await asyncio.sleep(30)
                r.raise_for_status()
            r.raise_for_status()
            remaining = r.headers.get("x-requests-remaining")
            if remaining:
                logger.info(f"Odds API quota remaining: {remaining}")
            data = await r.json()
            self._cache[cache_key] = (now, data)
            return data

    async def fetch_odds(self, competition: str) -> List[Dict[str, Any]]:
        sport = COMPETITION_TO_SPORT.get(competition)
        if not sport:
            return []
        try:
            return await self._get(
                f"/sports/{sport}/odds",
                {
                    "regions": "eu,uk",
                    "markets": "h2h,totals,btts",
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
            )
        except Exception as e:
            logger.warning(f"Odds API failed for {competition}: {e}")
            return []


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_match(
    home_name: str, away_name: str, kickoff: datetime, events: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    best = None
    best_score = 0.0
    for ev in events:
        try:
            ev_time = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        if abs((ev_time - kickoff).total_seconds()) > 6 * 3600:
            continue
        score = (
            _name_similarity(home_name, ev.get("home_team", ""))
            + _name_similarity(away_name, ev.get("away_team", ""))
        ) / 2
        if score > best_score and score > 0.6:
            best_score = score
            best = ev
    return best


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 == 1 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def extract_odds(event: Dict[str, Any], home_name: str, away_name: str) -> Dict[str, float]:
    """Aggregate odds across bookmakers — use the median to dodge outliers."""
    odds = {
        "odds_home": [], "odds_draw": [], "odds_away": [],
        "odds_over25": [], "odds_under25": [],
        "odds_btts_yes": [], "odds_btts_no": [],
    }
    for bk in event.get("bookmakers", []):
        for market in bk.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])
            if key == "h2h":
                for o in outcomes:
                    n = o.get("name", "")
                    price = float(o.get("price", 0))
                    if price <= 1.0:
                        continue
                    if n.lower() == "draw":
                        odds["odds_draw"].append(price)
                    elif _name_similarity(n, home_name) > 0.6 or _name_similarity(n, event.get("home_team", "")) > 0.8:
                        odds["odds_home"].append(price)
                    elif _name_similarity(n, away_name) > 0.6 or _name_similarity(n, event.get("away_team", "")) > 0.8:
                        odds["odds_away"].append(price)
            elif key == "totals":
                for o in outcomes:
                    if abs(float(o.get("point", 0)) - 2.5) > 0.01:
                        continue
                    price = float(o.get("price", 0))
                    if price <= 1.0:
                        continue
                    if o.get("name", "").lower() == "over":
                        odds["odds_over25"].append(price)
                    elif o.get("name", "").lower() == "under":
                        odds["odds_under25"].append(price)
            elif key == "btts":
                for o in outcomes:
                    price = float(o.get("price", 0))
                    if price <= 1.0:
                        continue
                    n = o.get("name", "").lower()
                    if n == "yes":
                        odds["odds_btts_yes"].append(price)
                    elif n == "no":
                        odds["odds_btts_no"].append(price)
    return {k: _median(v) for k, v in odds.items()}


async def fetch_odds_for_matches(
    client: OddsApiClient,
    upcoming: List[Tuple[int, str, str, str, datetime]],
) -> Dict[int, Dict[str, float]]:
    """upcoming: list of (match_id, competition, home_name, away_name, utc_date).
    Returns {match_id: {odds_home, odds_draw, ...}}."""
    out: Dict[int, Dict[str, float]] = {}
    by_comp: Dict[str, List[Tuple[int, str, str, datetime]]] = {}
    for mid, comp, h, a, dt in upcoming:
        by_comp.setdefault(comp, []).append((mid, h, a, dt))
    for comp, items in by_comp.items():
        events = await client.fetch_odds(comp)
        if not events:
            continue
        for mid, h, a, dt in items:
            ev = _best_match(h, a, dt, events)
            if ev:
                out[mid] = extract_odds(ev, h, a)
    return out
