from datetime import datetime, timedelta

import pandas as pd

from src.data.features import FEATURE_COLUMNS, build_features


def _toy_matches(n: int = 60) -> pd.DataFrame:
    rows = []
    start = datetime(2023, 1, 1)
    for i in range(n):
        home = (i % 4) + 1
        away = ((i + 1) % 4) + 1
        if home == away:
            away = (away % 4) + 1
        rows.append({
            "id": i + 1,
            "utc_date": start + timedelta(days=i),
            "home_team_id": home,
            "away_team_id": away,
            "home_goals": (i % 4),
            "away_goals": ((i + 1) % 3),
        })
    return pd.DataFrame(rows)


def test_build_features_shape():
    df = _toy_matches(40)
    feats = build_features(df)
    assert not feats.empty
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
    for col in ["outcome", "over25", "btts"]:
        assert col in feats.columns
    assert feats["outcome"].between(0, 2).all()


def test_features_no_leakage_first_row():
    df = _toy_matches(5)
    feats = build_features(df).sort_values("match_id").reset_index(drop=True)
    first = feats.iloc[0]
    assert first["home_elo"] == 1500.0
    assert first["away_elo"] == 1500.0
    assert first["elo_diff"] == 0.0
