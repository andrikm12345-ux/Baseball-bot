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


# odds-api.io uses slugs in the form "{country}-{league}" (observed:
# "czechia-cfl"). These are the most likely slugs for our top competitions —
# the client logs the discovered list on first run so we can adjust.
COMPETITION_TO_LEAGUE = {
    "PL": "england-premier-league",
    "PD": "spain-laliga",
    "SA": "italy-serie-a",
    "BL1": "germany-bundesliga",
    "FL1": "france-ligue-1",
    "CL": "uefa-champions-league",
    "PPL": "portugal-primeira-liga",
    "DED": "netherlands-eredivisie",
    "ELC": "england-championship",
}


# Market-name candidates. Observed names from odds-api.io:
#   "ML" (1X2)            "Goals Over/Under" (totals with hdp)
#   "Both Teams To Score" "Totals" (alt totals with hdp 2.75 sometimes)
# We accept several spellings so the parser works across bookmakers.
ML_NAMES = {"ml", "1x2", "match winner", "moneyline", "h2h"}
OU_NAMES = {
    "ou", "totals", "total", "over/under", "over_under",
    "goals over/under", "goals_over_under", "goals over under",
    "match goals", "match total goals",
}
BTTS_NAMES = {
    "btts", "both teams to score", "both_teams_to_score",
    "gg/ng", "both score",
}
HANDICAP_NAMES = {
    "handicap", "asian handicap", "asian_handicap", "ah",
    "goal handicap", "european handicap", "spread", "spreads",
    "handicap result", "goals handicap",
}


# ─────────────────────────── no-vig helpers ───────────────────────────


def novig_two_way(odd_a: Optional[float], odd_b: Optional[float]) -> Optional[Tuple[float, float]]:
    """Remove the bookmaker margin from a 2-outcome market.

    Returns (p_a, p_b) normalised to sum to 1, or None if either price missing.
    """
    if not odd_a or not odd_b or odd_a <= 1 or odd_b <= 1:
        return None
    ia, ib = 1.0 / odd_a, 1.0 / odd_b
    s = ia + ib
    if s <= 0:
        return None
    return ia / s, ib / s


def novig_three_way(
    odd_home: Optional[float], odd_draw: Optional[float], odd_away: Optional[float]
) -> Optional[Tuple[float, float, float]]:
    """Remove margin from the 3-outcome 1X2 market (football has a draw!)."""
    odds = [odd_home, odd_draw, odd_away]
    if any((not o or o <= 1) for o in odds):
        return None
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    if s <= 0:
        return None
    return inv[0] / s, inv[1] / s, inv[2] / s


class OddsApiClient:
    """odds-api.io v3 client with in-memory cache and back-off on 429."""

    def __init__(self, api_key: str, cache_ttl_seconds: float = 3600.0) -> None:
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl_seconds
        self._first_odds_logged = False
        self._first_event_logged = False

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
        except OddsApiError:
            raise
        except Exception as e:
            logger.warning(f"fetch_events failed (sport={sport} league={league}): {e}")
            return []
        events = data if isinstance(data, list) else data.get("events", [])
        if events and not self._first_event_logged:
            self._first_event_logged = True
            try:
                logger.info(
                    f"odds-api.io sample event JSON (keys={list(events[0].keys())}): "
                    f"{json.dumps(events[0])[:1500]}"
                )
            except Exception:
                pass
        return events

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


def _entry_line(entry: Dict[str, Any]) -> Optional[float]:
    for key in ("hdp", "handicap", "line", "point", "total"):
        if key in entry:
            try:
                return round(float(entry[key]) * 4) / 4  # snap to nearest 0.25
            except (TypeError, ValueError):
                pass
    return None


def extract_odds(
    odds_payload: Dict[str, Any],
    home_name: str = "",
    away_name: str = "",
    event_home: str = "",
    event_away: str = "",
) -> Dict[str, Any]:
    """Parse odds-api.io payload into a structured, multi-line shape.

    Returns:
      {
        "ml": {"home", "draw", "away", "p_home", "p_draw", "p_away"} | None,
        "totals": [{"point", "over", "under", "over_novig", "under_novig"}, ...],
        "handicaps": [{"point", "home", "away", "home_novig", "away_novig"}, ...],
      }

    Prices come as strings, lines as numbers in "hdp". We keep ALL total and
    handicap lines (1.5 / 2.5 / 3.5 / -0.5 …), taking the max odd per
    (line, side) across the books we pay for, then compute a no-vig market
    probability for each.
    """
    ml_agg: Dict[str, List[float]] = {"home": [], "draw": [], "away": []}
    totals_agg: Dict[float, Dict[str, List[float]]] = {}
    hcap_agg: Dict[float, Dict[str, List[float]]] = {}

    books = odds_payload.get("bookmakers") or {}
    if isinstance(books, list):
        books = {str(b.get("name") or b.get("key") or i): b.get("markets", b) for i, b in enumerate(books)}
    if not isinstance(books, dict):
        return {"ml": None, "totals": [], "handicaps": []}

    unknown_markets: set[str] = set()

    for _bk_name, markets in books.items():
        if not isinstance(markets, list):
            markets = markets.get("markets", []) if isinstance(markets, dict) else []
        for market in markets:
            if not isinstance(market, dict):
                continue
            name = _market_name(market)
            odds_list = market.get("odds")
            if not isinstance(odds_list, list):
                odds_list = [odds_list] if isinstance(odds_list, dict) else []

            if name in ML_NAMES:
                for entry in odds_list:
                    if not isinstance(entry, dict):
                        continue
                    h, d, a = (_as_float(entry.get("home")), _as_float(entry.get("draw")),
                               _as_float(entry.get("away")))
                    if h: ml_agg["home"].append(h)
                    if d: ml_agg["draw"].append(d)
                    if a: ml_agg["away"].append(a)
            elif name in OU_NAMES:
                for entry in odds_list:
                    if not isinstance(entry, dict):
                        continue
                    pt = _entry_line(entry)
                    over, under = _as_float(entry.get("over")), _as_float(entry.get("under"))
                    if pt is None or (not over and not under):
                        continue
                    slot = totals_agg.setdefault(pt, {"over": [], "under": []})
                    if over: slot["over"].append(over)
                    if under: slot["under"].append(under)
            elif name in HANDICAP_NAMES:
                for entry in odds_list:
                    if not isinstance(entry, dict):
                        continue
                    pt = _entry_line(entry)
                    h, a = _as_float(entry.get("home")), _as_float(entry.get("away"))
                    if pt is None or (not h and not a):
                        continue
                    slot = hcap_agg.setdefault(pt, {"home": [], "away": []})
                    if h: slot["home"].append(h)
                    if a: slot["away"].append(a)
            elif name not in BTTS_NAMES:
                unknown_markets.add(name)

    if unknown_markets:
        logger.debug(f"odds-api.io: unhandled market names: {sorted(unknown_markets)[:10]}")

    ml = None
    if ml_agg["home"] and ml_agg["draw"] and ml_agg["away"]:
        oh, od, oa = max(ml_agg["home"]), max(ml_agg["draw"]), max(ml_agg["away"])
        nv = novig_three_way(oh, od, oa)
        ml = {"home": oh, "draw": od, "away": oa}
        if nv:
            ml.update(p_home=nv[0], p_draw=nv[1], p_away=nv[2])

    totals = []
    for pt, sides in sorted(totals_agg.items()):
        over = max(sides["over"]) if sides["over"] else None
        under = max(sides["under"]) if sides["under"] else None
        nv = novig_two_way(over, under)
        totals.append({
            "point": pt, "over": over, "under": under,
            "over_novig": nv[0] if nv else None,
            "under_novig": nv[1] if nv else None,
        })

    handicaps = []
    for pt, sides in sorted(hcap_agg.items()):
        h = max(sides["home"]) if sides["home"] else None
        a = max(sides["away"]) if sides["away"] else None
        nv = novig_two_way(h, a)
        handicaps.append({
            "point": pt, "home": h, "away": a,
            "home_novig": nv[0] if nv else None,
            "away_novig": nv[1] if nv else None,
        })

    return {"ml": ml, "totals": totals, "handicaps": handicaps}


# ─────────────────────────── top-level orchestration ───────────────────────────


def _is_upcoming(ev: Dict[str, Any], now: datetime) -> bool:
    status = str(ev.get("status", "")).lower()
    if status in {"settled", "finished", "ended", "cancelled", "canceled", "postponed"}:
        return False
    kickoff = _event_kickoff(ev)
    if kickoff is None:
        return True  # keep — better to try than drop
    return kickoff >= now - timedelta(hours=2)  # 2h grace for in-play


async def fetch_odds_for_matches(
    client: OddsApiClient,
    upcoming: List[Tuple[int, str, str, str, datetime]],
) -> Dict[int, Dict[str, Any]]:
    """upcoming: (match_id, competition, home_name, away_name, utc_date).

    Strategy: per-league /events call using slugs from COMPETITION_TO_LEAGUE
    (format observed in their API is "{country}-{league}"). Filter out
    already-settled events locally. If a slug returns 404 we drop it from
    future attempts to save quota.
    """
    if not upcoming:
        return {}

    now = datetime.utcnow()
    events_cache: Dict[str, List[Dict[str, Any]]] = {}
    bad_slugs: set[str] = set()

    for _, comp, _, _, _ in upcoming:
        if comp in events_cache:
            continue
        league = COMPETITION_TO_LEAGUE.get(comp)
        if league and league not in bad_slugs:
            try:
                evs = await client.fetch_events(sport="football", league=league, limit=200)
            except OddsApiError as e:
                if e.status == 404:
                    bad_slugs.add(league)
                    evs = []
                else:
                    evs = []
        else:
            evs = []
        evs = [ev for ev in evs if _is_upcoming(ev, now)]
        events_cache[comp] = evs
        logger.info(f"odds-api.io: {len(evs)} upcoming events for {comp} (league={league})")

    out: Dict[int, Dict[str, float]] = {}
    for match_id, comp, home, away, kickoff in upcoming:
        events = events_cache.get(comp) or []
        if not events:
            continue
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
        if odds.get("ml") or odds.get("totals") or odds.get("handicaps"):
            out[match_id] = odds

    return out
