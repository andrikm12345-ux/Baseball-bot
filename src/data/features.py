"""Feature engineering for football match prediction.

Builds, for every historical match, a feature vector available BEFORE kickoff
(no leakage): rolling form, attack/defense rates, Elo rating, head-to-head.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_form_pts",
    "away_form_pts",
    "home_gf_avg",
    "home_ga_avg",
    "away_gf_avg",
    "away_ga_avg",
    "home_gf_home_avg",
    "home_ga_home_avg",
    "away_gf_away_avg",
    "away_ga_away_avg",
    "h2h_home_winrate",
    "h2h_avg_goals",
    "rest_diff",
]


@dataclass
class TeamState:
    elo: float = 1500.0
    last_results: Deque[Tuple[int, int, str]] = field(default_factory=lambda: deque(maxlen=10))  # (gf, ga, venue)
    last_home: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=10))
    last_away: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=10))
    last_match_date: pd.Timestamp | None = None


def _form_pts(results: Deque[Tuple[int, int, str]]) -> float:
    if not results:
        return 1.0
    pts = 0
    for gf, ga, _ in results:
        if gf > ga:
            pts += 3
        elif gf == ga:
            pts += 1
    return pts / len(results)


def _avg(results: Deque, idx: int) -> float:
    if not results:
        return 1.2
    return float(np.mean([r[idx] for r in results]))


def _elo_expected(home_elo: float, away_elo: float, home_adv: float = 65.0) -> float:
    return 1.0 / (1.0 + 10 ** (-(home_elo + home_adv - away_elo) / 400.0))


def _elo_update(home_elo: float, away_elo: float, home_g: int, away_g: int, k: float = 20.0) -> Tuple[float, float]:
    expected_home = _elo_expected(home_elo, away_elo)
    if home_g > away_g:
        score = 1.0
    elif home_g == away_g:
        score = 0.5
    else:
        score = 0.0
    margin = abs(home_g - away_g)
    k_eff = k * (1 + np.log1p(margin))
    delta = k_eff * (score - expected_home)
    return home_elo + delta, away_elo - delta


def build_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build features chronologically — state at row N reflects only rows < N.

    Expects columns: id, utc_date, home_team_id, away_team_id, home_goals, away_goals.
    Returns rows for matches with known scores (training set) plus features.
    """
    df = matches_df.sort_values("utc_date").reset_index(drop=True).copy()

    team_state: Dict[int, TeamState] = defaultdict(TeamState)
    h2h: Dict[Tuple[int, int], Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=6))

    feats: List[Dict[str, float]] = []
    targets: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        h = team_state[home_id]
        a = team_state[away_id]

        rest_h = 7.0
        rest_a = 7.0
        if h.last_match_date is not None:
            rest_h = (row["utc_date"] - h.last_match_date).days
        if a.last_match_date is not None:
            rest_a = (row["utc_date"] - a.last_match_date).days

        key = tuple(sorted([home_id, away_id]))
        h2h_list = h2h[key]
        if h2h_list:
            home_wins = sum(1 for gh, ga in h2h_list if gh > ga)
            h2h_home_wr = home_wins / len(h2h_list)
            h2h_avg_g = float(np.mean([gh + ga for gh, ga in h2h_list]))
        else:
            h2h_home_wr = 0.5
            h2h_avg_g = 2.6

        feat = {
            "elo_diff": h.elo - a.elo,
            "home_elo": h.elo,
            "away_elo": a.elo,
            "home_form_pts": _form_pts(h.last_results),
            "away_form_pts": _form_pts(a.last_results),
            "home_gf_avg": _avg(h.last_results, 0),
            "home_ga_avg": _avg(h.last_results, 1),
            "away_gf_avg": _avg(a.last_results, 0),
            "away_ga_avg": _avg(a.last_results, 1),
            "home_gf_home_avg": _avg(h.last_home, 0),
            "home_ga_home_avg": _avg(h.last_home, 1),
            "away_gf_away_avg": _avg(a.last_away, 0),
            "away_ga_away_avg": _avg(a.last_away, 1),
            "h2h_home_winrate": h2h_home_wr,
            "h2h_avg_goals": h2h_avg_g,
            "rest_diff": rest_h - rest_a,
        }

        if pd.notna(row.get("home_goals")) and pd.notna(row.get("away_goals")):
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            outcome = 0 if hg > ag else (1 if hg == ag else 2)
            total_goals = hg + ag
            btts = int(hg > 0 and ag > 0)
            targets.append(
                {
                    "match_id": int(row["id"]),
                    "outcome": outcome,
                    "over25": int(total_goals > 2),
                    "btts": btts,
                    "total_goals": total_goals,
                }
            )
            feats.append({"match_id": int(row["id"]), **feat})
            # update state
            h.last_results.append((hg, ag, "H"))
            a.last_results.append((ag, hg, "A"))
            h.last_home.append((hg, ag))
            a.last_away.append((ag, hg))
            h.last_match_date = row["utc_date"]
            a.last_match_date = row["utc_date"]
            new_h_elo, new_a_elo = _elo_update(h.elo, a.elo, hg, ag)
            h.elo = new_h_elo
            a.elo = new_a_elo
            h2h[key].append((hg, ag) if home_id == key[0] else (ag, hg))

    if not feats:
        return pd.DataFrame(columns=["match_id", *FEATURE_COLUMNS, "outcome", "over25", "btts"])
    f_df = pd.DataFrame(feats)
    t_df = pd.DataFrame(targets)
    return f_df.merge(t_df, on="match_id")


def build_inference_features(
    upcoming_df: pd.DataFrame, history_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute features for upcoming matches by replaying history first."""
    all_df = pd.concat([history_df, upcoming_df], ignore_index=True).sort_values("utc_date").reset_index(drop=True)

    team_state: Dict[int, TeamState] = defaultdict(TeamState)
    h2h: Dict[Tuple[int, int], Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=6))

    rows: List[Dict[str, float]] = []
    upcoming_ids = set(upcoming_df["id"].astype(int).tolist())

    for _, row in all_df.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        h = team_state[home_id]
        a = team_state[away_id]

        rest_h = 7.0 if h.last_match_date is None else (row["utc_date"] - h.last_match_date).days
        rest_a = 7.0 if a.last_match_date is None else (row["utc_date"] - a.last_match_date).days

        key = tuple(sorted([home_id, away_id]))
        h2h_list = h2h[key]
        if h2h_list:
            home_wins = sum(1 for gh, ga in h2h_list if gh > ga)
            h2h_home_wr = home_wins / len(h2h_list)
            h2h_avg_g = float(np.mean([gh + ga for gh, ga in h2h_list]))
        else:
            h2h_home_wr = 0.5
            h2h_avg_g = 2.6

        feat = {
            "match_id": int(row["id"]),
            "elo_diff": h.elo - a.elo,
            "home_elo": h.elo,
            "away_elo": a.elo,
            "home_form_pts": _form_pts(h.last_results),
            "away_form_pts": _form_pts(a.last_results),
            "home_gf_avg": _avg(h.last_results, 0),
            "home_ga_avg": _avg(h.last_results, 1),
            "away_gf_avg": _avg(a.last_results, 0),
            "away_ga_avg": _avg(a.last_results, 1),
            "home_gf_home_avg": _avg(h.last_home, 0),
            "home_ga_home_avg": _avg(h.last_home, 1),
            "away_gf_away_avg": _avg(a.last_away, 0),
            "away_ga_away_avg": _avg(a.last_away, 1),
            "h2h_home_winrate": h2h_home_wr,
            "h2h_avg_goals": h2h_avg_g,
            "rest_diff": rest_h - rest_a,
        }
        rows.append(feat)

        if pd.notna(row.get("home_goals")) and pd.notna(row.get("away_goals")):
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            h.last_results.append((hg, ag, "H"))
            a.last_results.append((ag, hg, "A"))
            h.last_home.append((hg, ag))
            a.last_away.append((ag, hg))
            h.last_match_date = row["utc_date"]
            a.last_match_date = row["utc_date"]
            new_h_elo, new_a_elo = _elo_update(h.elo, a.elo, hg, ag)
            h.elo = new_h_elo
            a.elo = new_a_elo
            h2h[key].append((hg, ag) if home_id == key[0] else (ag, hg))

    feats_df = pd.DataFrame(rows)
    return feats_df[feats_df["match_id"].isin(upcoming_ids)].reset_index(drop=True)
