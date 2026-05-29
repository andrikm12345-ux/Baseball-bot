"""Signal dataclass + stake sizing (quarter-Kelly).

Signal construction itself lives in the pipeline (Claude picks the bet and the
no-vig edge is computed there); this module only holds the shared shape and the
Kelly helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    match_id: int
    market: str       # "1X2" | "TOTAL" | "HANDICAP"
    pick: str         # "HOME"/"DRAW"/"AWAY", "OVER"/"UNDER"
    model_prob: float
    fair_odds: float
    book_odds: float
    edge: float       # confidence − market no-vig probability
    confidence: float
    stake_units: float
    is_value: bool
    line: Optional[float] = None


def _kelly(p: float, odds: float, fraction: float = 0.25, cap: float = 2.0) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    k = (b * p - q) / b
    if k <= 0:
        return 0.0
    return min(k * fraction * 10, cap)  # 10 = base bankroll units
