"""Client for odds-api.io v3.

Architecture:
  1. GET /v3/events?sport=football  → list of upcoming events (id, teams, time, league)
  2. For each event we want odds for: GET /v3/odds?eventId=X&bookmakers=Bet365,Betfair Exchange
  3. Response shape (observed in docs):
       {
         "status": "pending",
         "urls": {"Bet365": "...", "Betfair Exchange": "..."},
         "bookmakers": {
           "Bet365": [ {market}, {market}, ... ],
           "Betfair Exchange": [...]
         }
       }
     Each market has a "name" (e.g. "ML" for moneyline / 1X2) and an "odds" structure
     whose exact shape we discover defensively — the parser tries several layouts and
     logs the raw payload at DEBUG level for the first match to make tuning trivial.

Bookmaker aggregation: we take the MAX odd per outcome across the two books we pay
for. Betfair Exchange tends to be highest (no margin), Bet365 is the practical price
— max gives the best edge while staying realistic.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


BASE_URL = "https://api.odds-api.io/v3"

DEFAULT_BOOKMAKERS = "Bet365,Betfair Exchange"


class OddsApiError(Exception):
    """Raised on non-retryable HTTP errors from odds-api.io (4xx)."""
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"odds-api.io {status}: {body[:200]}")
        self.status = status
        self.body = body


# Currently unused — odds-api.io rejected every slug we tried with 404
# "League not found". Kept for the day we learn the real slugs.
COMPETITION_TO_LEAGUE = {
    "PL": "premier-league",
    "PD": "la-liga",
    "SA": "serie-a",
    "BL1": "bundesliga",
    "FL1": "ligue-1",
    "CL": "champions-league",
    "PPL": "primeira-liga",
    "DED": "eredivisie",
    "ELC": "championship",
}


# Market-name candidates. odds-api.io uses short codes ("ML" for 1X2 in docs);
# the OU/BTTS codes are not shown in the quickstart, so we accept several
# common spellings and log unknown ones so we can extend this list.
ML_NAMES = {"ml", "1x2", "match winner", "moneyline", "h2h"}
OU_NAMES = {"ou", "totals", "over/under", "over_under", "total"}
BTTS_NAMES = {"btts", "both teams to score", "gg/ng", "both_teams_to_score"}


class OddsApiClient:
    """odds-api.io v3 client with in-memory cache and back-off on 429."""

    def __init__(self, api_key: str, cache_ttl_seconds: float = 3600.0) -> None:
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl_seconds
        self._first_odds_logged = False

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type((aiohttp.ClientConnectionError, asyncio.TimeoutError)),
    )
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
                logger.warning("odds-api.io rate-limited, backing off 30s")
                await asyncio.sleep(30)
            if r.status >= 400:
                text = await r.text()
                logger.warning(f"odds-api.io {r.status} on {path}: {text[:200]}")
                # 4xx is a hard error — don't retry, don't burn quota
                raise OddsApiError(r.status, text)
            remaining = r.headers.get("x-requests-remaining") or r.headers.get("X-RateLimit-Remaining")
            if remaining:
                logger.info(f"odds-api.io quota remaining: {remaining}")
            data = await r.json()
            self._cache[cache_key] = (now, data)
            return data

    async def fetch_events(
        self,
        sport: str = "football",
        league: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"sport": sport, "limit": limit}
        if league:
            params["league"] = league
        try:
            data = await self._get("/events", params)
        except Exception as e:
            logger.warning(f"fetch_events failed (sport={sport} league={league}): {e}")
            return []
        return data if isinstance(data, list) else data.get("events", [])

    async def fetch_event_odds(
        self,
        event_id: int | str,
        bookmakers: str = DEFAULT_BOOKMAKERS,
    ) -> Optional[Dict[str, Any]]:
        try:
            data = await self._get(
                "/odds",
                {"eventId": event_id, "bookmakers": bookmakers},
            )
        except Exception as e:
            logger.warning(f"fetch_event_odds {event_id} failed: {e}")
            return None
        if not self._first_odds_logged:
            self._first_odds_logged = True
            try:
                logger.info(f"odds-api.io sample /odds payload: {json.dumps(data)[:1500]}")
            except Exception:
                pass
        return data


# ─────────────────────────── matching ───────────────────────────


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, (int, float)):
        # epoch seconds or ms
        try:
            if value > 1e12:
                return datetime.utcfromtimestamp(value / 1000.0)
            return datetime.utcfromtimestamp(value)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None
    return None


def _event_kickoff(ev: Dict[str, Any]) -> Optional[datetime]:
    for key in ("commenceTime", "commence_time", "startTime", "start_time", "kickoff", "date", "scheduled"):
        if key in ev:
            dt = _parse_dt(ev[key])
            if dt:
                return dt
    return None


def _event_teams(ev: Dict[str, Any]) -> Tuple[str, str]:
    """Try several known shapes for team names."""
    home = ev.get("homeTeam") or ev.get("home_team") or ev.get("home")
    away = ev.get("awayTeam") or ev.get("away_team") or ev.get("away")
    if isinstance(home, dict):
        home = home.get("name") or home.get("title") or ""
    if isinstance(away, dict):
        away = away.get("name") or away.get("title") or ""
    if not home or not away:
        teams = ev.get("teams") or ev.get("participants") or []
        if isinstance(teams, list) and len(teams) >= 2:
            t0 = teams[0].get("name") if isinstance(teams[0], dict) else teams[0]
            t1 = teams[1].get("name") if isinstance(teams[1], dict) else teams[1]
            home = home or t0
            away = away or t1
    return str(home or ""), str(away or "")


def _best_match(
    home_name: str, away_name: str, kickoff: datetime, events: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    best = None
    best_score = 0.0
    for ev in events:
        ev_time = _event_kickoff(ev)
        if ev_time and abs((ev_time - kickoff).total_seconds()) > 6 * 3600:
            continue
        h, a = _event_teams(ev)
        score = (_name_similarity(home_name, h) + _name_similarity(away_name, a)) / 2
        if score > best_score and score > 0.55:
            best_score = score
            best = ev
    return best


# ─────────────────────────── odds parsing ───────────────────────────


def _as_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def _extract_market_outcomes(market: Dict[str, Any]) -> Dict[str, Any]:
    """Returns {outcome_label_lower: raw_value} for whatever shape the market uses.

    Known/guessed shapes:
      A) market["odds"] = {"home": 1.9, "draw": 3.2, "away": 4.0}
      B) market["odds"] = [{"home": ..., "draw": ..., "away": ...}]
      C) market["odds"] = [{"name": "home", "price": 1.9}, ...]
      D) market["outcomes"] = [{"name": "Over 2.5", "price": 1.85, "point": 2.5}, ...]
    """
    out: Dict[str, Any] = {}
    odds_field = market.get("odds")
    if isinstance(odds_field, dict):
        out.update({str(k).lower(): v for k, v in odds_field.items()})
    elif isinstance(odds_field, list) and odds_field:
        first = odds_field[0]
        if isinstance(first, dict) and not {"name", "price"} <= first.keys():
            out.update({str(k).lower(): v for k, v in first.items()})
        else:
            for o in odds_field:
                if isinstance(o, dict) and "name" in o:
                    out[str(o["name"]).lower()] = o.get("price") or o.get("odds") or o.get("value")
    for key in ("outcomes", "selections", "lines"):
        if key in market and isinstance(market[key], list):
            for o in market[key]:
                if isinstance(o, dict) and "name" in o:
                    out[str(o["name"]).lower()] = o.get("price") or o.get("odds") or o.get("value")
    return out


def _market_name(market: Dict[str, Any]) -> str:
    return str(
        market.get("name") or market.get("key") or market.get("market") or ""
    ).lower().strip()


def _market_line(market: Dict[str, Any]) -> Optional[float]:
    for key in ("handicap", "line", "point", "total"):
        if key in market:
            try:
                return float(market[key])
            except Exception:
                pass
    return None


def extract_odds(
    odds_payload: Dict[str, Any],
    home_name: str,
    away_name: str,
    event_home: str = "",
    event_away: str = "",
) -> Dict[str, float]:
    """Walk through bookmakers→markets→outcomes and aggregate max odd per pick."""
    aggregated: Dict[str, List[float]] = {
        "odds_home": [], "odds_draw": [], "odds_away": [],
        "odds_over25": [], "odds_under25": [],
        "odds_btts_yes": [], "odds_btts_no": [],
    }
    books = odds_payload.get("bookmakers") or {}
    if isinstance(books, list):
        books = {str(b.get("name") or b.get("key") or i): b.get("markets", b) for i, b in enumerate(books)}
    if not isinstance(books, dict):
        return {k: 0.0 for k in aggregated}

    home_candidates = {"home", home_name.lower(), event_home.lower(), "1"}
    away_candidates = {"away", away_name.lower(), event_away.lower(), "2"}
    draw_candidates = {"draw", "x", "tie"}
    unknown_markets: set[str] = set()

    for bk_name, markets in books.items():
        if not isinstance(markets, list):
            markets = markets.get("markets", []) if isinstance(markets, dict) else []
        for market in markets:
            if not isinstance(market, dict):
                continue
            name = _market_name(market)
            outcomes = _extract_market_outcomes(market)
            if not outcomes:
                continue

            if name in ML_NAMES:
                for label, val in outcomes.items():
                    f = _as_float(val)
                    if f is None:
                        continue
                    if any(_name_similarity(label, c) > 0.7 for c in home_candidates):
                        aggregated["odds_home"].append(f)
                    elif label in draw_candidates:
                        aggregated["odds_draw"].append(f)
                    elif any(_name_similarity(label, c) > 0.7 for c in away_candidates):
                        aggregated["odds_away"].append(f)
            elif name in OU_NAMES:
                line = _market_line(market)
                if line is not None and abs(line - 2.5) > 0.01:
                    continue
                for label, val in outcomes.items():
                    f = _as_float(val)
                    if f is None:
                        continue
                    if "over" in label or label == "o":
                        aggregated["odds_over25"].append(f)
                    elif "under" in label or label == "u":
                        aggregated["odds_under25"].append(f)
            elif name in BTTS_NAMES:
                for label, val in outcomes.items():
                    f = _as_float(val)
                    if f is None:
                        continue
                    if label in {"yes", "gg", "y"}:
                        aggregated["odds_btts_yes"].append(f)
                    elif label in {"no", "ng", "n"}:
                        aggregated["odds_btts_no"].append(f)
            else:
                unknown_markets.add(name)

    if unknown_markets:
        logger.debug(f"odds-api.io: unhandled market names: {sorted(unknown_markets)[:10]}")

    return {k: (max(v) if v else 0.0) for k, v in aggregated.items()}


# ─────────────────────────── top-level orchestration ───────────────────────────


async def fetch_odds_for_matches(
    client: OddsApiClient,
    upcoming: List[Tuple[int, str, str, str, datetime]],
) -> Dict[int, Dict[str, float]]:
    """upcoming: (match_id, competition, home_name, away_name, utc_date).

    Strategy: one global /events?sport=football call returns all upcoming
    football matches across all leagues. We match by team name + kickoff —
    league filter is skipped because odds-api.io's league slugs are not
    documented and 4xx errors burn the daily quota fast.
    """
    if not upcoming:
        return {}

    events = await client.fetch_events(sport="football", limit=500)
    logger.info(f"odds-api.io: {len(events)} football events available")
    if not events:
        return {}

    out: Dict[int, Dict[str, float]] = {}
    for match_id, _comp, home, away, kickoff in upcoming:
        ev = _best_match(home, away, kickoff, events)
        if not ev:
            continue
        ev_id = ev.get("id") or ev.get("eventId") or ev.get("event_id")
        if not ev_id:
            continue
        payload = await client.fetch_event_odds(ev_id)
        if not payload:
            continue
        ev_home, ev_away = _event_teams(ev)
        odds = extract_odds(payload, home, away, ev_home, ev_away)
        if any(v > 0 for v in odds.values()):
            out[match_id] = odds

    return out
