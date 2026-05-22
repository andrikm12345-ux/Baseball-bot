"""Turn model probabilities into bet signals.

A signal is emitted only when:
  - model confidence >= MIN_CONFIDENCE
  - book odds available AND in [MIN_ODDS, MAX_ODDS]
  - edge (model_prob * book_odds - 1) >= MIN_EDGE

Stake is sized by quarter-Kelly capped at 2 units, which is the standard
conservative bankroll-management heuristic.

Note: without real bookmaker odds (no API key for an odds provider) we fall
back to a "model-only" signal that publishes the pick with confidence but
does NOT claim a value edge. The bot makes that distinction explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.config import settings


@dataclass
class Signal:
    match_id: int
    market: str       # "1X2" | "OU25" | "BTTS"
    pick: str         # "HOME"/"DRAW"/"AWAY", "OVER"/"UNDER", "YES"/"NO"
    model_prob: float
    fair_odds: float
    book_odds: float  # 0.0 if unknown
    edge: float
    confidence: float
    stake_units: float
    is_value: bool


def _kelly(p: float, odds: float, fraction: float = 0.25, cap: float = 2.0) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    k = (b * p - q) / b
    if k <= 0:
        return 0.0
    return min(k * fraction * 10, cap)  # 10 = base bankroll units


def _best_1x2(row: pd.Series) -> tuple[str, float]:
    probs = {"HOME": row["p_home"], "DRAW": row["p_draw"], "AWAY": row["p_away"]}
    pick = max(probs, key=probs.get)
    return pick, probs[pick]


def _book_odds(row: pd.Series, market: str, pick: str) -> Optional[float]:
    col = {
        ("1X2", "HOME"): "odds_home",
        ("1X2", "DRAW"): "odds_draw",
        ("1X2", "AWAY"): "odds_away",
        ("OU25", "OVER"): "odds_over25",
        ("OU25", "UNDER"): "odds_under25",
        ("BTTS", "YES"): "odds_btts_yes",
        ("BTTS", "NO"): "odds_btts_no",
    }.get((market, pick))
    if col and col in row and pd.notna(row[col]) and row[col] > 1.0:
        return float(row[col])
    return None


_ITB_MARKETS = [
    ("HOME_OVER05", "p_home_over05"),
    ("HOME_OVER15", "p_home_over15"),
    ("HOME_OVER25", "p_home_over25"),
    ("AWAY_OVER05", "p_away_over05"),
    ("AWAY_OVER15", "p_away_over15"),
    ("AWAY_OVER25", "p_away_over25"),
]


def generate(predictions_with_odds: pd.DataFrame) -> List[Signal]:
    """Predictions df expects: match_id, p_home, p_draw, p_away, p_over25, p_btts,
    plus optional odds_* columns from a bookmaker feed and optional p_*_over* ITB probs."""
    out: List[Signal] = []
    for _, row in predictions_with_odds.iterrows():
        # 1X2
        pick, prob = _best_1x2(row)
        out.extend(_make_signal(row, "1X2", pick, prob))
        # OU 2.5
        ou_pick = "OVER" if row["p_over25"] >= 0.5 else "UNDER"
        ou_prob = row["p_over25"] if ou_pick == "OVER" else 1 - row["p_over25"]
        out.extend(_make_signal(row, "OU25", ou_pick, ou_prob))
        # BTTS
        bt_pick = "YES" if row["p_btts"] >= 0.5 else "NO"
        bt_prob = row["p_btts"] if bt_pick == "YES" else 1 - row["p_btts"]
        out.extend(_make_signal(row, "BTTS", bt_pick, bt_prob))
        # ITB signals intentionally disabled — without bookmaker odds the model
        # flags ИТБ 0.5 on almost every match (P >= 0.7 routinely). Re-enable
        # once team-totals odds feed is wired up so value/edge can gate them.
    return out


def _make_signal(row: pd.Series, market: str, pick: str, prob: float) -> List[Signal]:
    if prob < settings.min_confidence:
        return []
    fair = 1.0 / max(prob, 1e-6)
    book = _book_odds(row, market, pick)
    if book is not None:
        if book < settings.min_odds or book > settings.max_odds:
            return []
        edge = prob * book - 1.0
        if edge < settings.min_edge:
            return []
        stake = _kelly(prob, book)
        return [Signal(
            match_id=int(row["match_id"]),
            market=market, pick=pick,
            model_prob=float(prob), fair_odds=float(fair),
            book_odds=float(book), edge=float(edge),
            confidence=float(prob), stake_units=float(round(stake, 2)),
            is_value=True,
        )]
    # No bookmaker odds — still publish the pick as a "model signal" if confident
    if prob >= max(settings.min_confidence, 0.60):
        return [Signal(
            match_id=int(row["match_id"]),
            market=market, pick=pick,
            model_prob=float(prob), fair_odds=float(fair),
            book_odds=0.0, edge=0.0,
            confidence=float(prob), stake_units=1.0,
            is_value=False,
        )]
    return []
