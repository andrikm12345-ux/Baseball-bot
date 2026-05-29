"""Settle signals against final scores and compute ROI."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from loguru import logger
from sqlalchemy import select

from src.data.database import Match, SessionLocal, Signal


def _did_win(market: str, pick: str, hg: int, ag: int, line: float | None = None) -> Optional[bool]:
    """Return True (win), False (loss) or None (push/void — stake returned)."""
    if market == "1X2":
        if pick == "HOME":
            return hg > ag
        if pick == "DRAW":
            return hg == ag
        if pick == "AWAY":
            return ag > hg
    if market == "TOTAL":
        ln = line if line is not None else 2.5
        total = hg + ag
        if abs(total - ln) < 1e-9:  # whole-number line, exact hit → push
            return None
        return total > ln if pick == "OVER" else total < ln
    if market == "HANDICAP":
        # line is the handicap applied to the HOME team (e.g. -0.5 means home
        # must win by 1+). Margin from the backed side's perspective.
        ln = line if line is not None else 0.0
        adj = (hg + ln) - ag if pick == "HOME" else (ag - ln) - hg
        if abs(adj) < 1e-9:  # exact → push
            return None
        return adj > 0
    return False


@dataclass
class RoiStats:
    n_settled: int
    n_won: int
    staked: float
    returned: float
    profit: float
    roi: float
    hit_rate: float


async def settle_pending() -> int:
    """Mark every unsettled signal whose match is FINISHED."""
    settled = 0
    async with SessionLocal() as session:
        q = await session.execute(
            select(Signal, Match).join(Match, Match.id == Signal.match_id).where(
                Signal.settled.is_(False),
                Match.status == "FINISHED",
            )
        )
        for sig, match in q.all():
            if match.home_goals is None or match.away_goals is None:
                continue
            won = _did_win(sig.market, sig.pick, match.home_goals, match.away_goals, sig.line)
            sig.won = won
            sig.settled = True
            if won is None:
                sig.profit_units = 0.0  # push / void — stake returned
            elif sig.book_odds and sig.book_odds > 1.0:
                sig.profit_units = (sig.stake_units * (sig.book_odds - 1.0)) if won else -sig.stake_units
            else:
                sig.profit_units = sig.stake_units if won else -sig.stake_units
            settled += 1
        await session.commit()
    if settled:
        logger.info(f"Settled {settled} signals")
    return settled


async def roi_stats(
    last_n: int | None = None,
    market: str | None = None,
    since: Optional[datetime] = None,
) -> RoiStats:
    """ROI summary over settled, non-push signals. Optional market filter
    ("1X2" | "TOTAL" | "HANDICAP") and time / count windows.
    """
    async with SessionLocal() as session:
        q = select(Signal).where(Signal.settled.is_(True)).order_by(Signal.created_at.desc())
        if market is not None:
            q = q.where(Signal.market == market)
        if since is not None:
            q = q.where(Signal.created_at >= since)
        if last_n:
            q = q.limit(last_n)
        rows: List[Signal] = list((await session.execute(q)).scalars())
    # Exclude pushes (won is None) from all ratios
    rows = [r for r in rows if r.won is not None]
    n = len(rows)
    if n == 0:
        return RoiStats(0, 0, 0, 0, 0, 0.0, 0.0)
    staked = sum(r.stake_units for r in rows)
    returned = sum(
        (r.stake_units * r.book_odds) if (r.won and r.book_odds > 1.0) else 0.0 for r in rows
    )
    profit = sum(r.profit_units or 0.0 for r in rows)
    won = sum(1 for r in rows if r.won)
    return RoiStats(
        n_settled=n, n_won=won, staked=staked, returned=returned,
        profit=profit,
        roi=(profit / staked * 100.0) if staked > 0 else 0.0,
        hit_rate=(won / n * 100.0) if n > 0 else 0.0,
    )
