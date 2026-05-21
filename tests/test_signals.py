import pandas as pd

from src.config import settings
from src.signals.generator import generate


def test_value_signal_with_odds():
    settings.min_edge = 0.05
    settings.min_confidence = 0.55
    settings.min_odds = 1.5
    settings.max_odds = 4.5
    df = pd.DataFrame([{
        "match_id": 1,
        "p_home": 0.65, "p_draw": 0.20, "p_away": 0.15,
        "p_over25": 0.62, "p_btts": 0.55,
        "odds_home": 1.90, "odds_draw": 3.6, "odds_away": 6.0,
        "odds_over25": 1.85, "odds_under25": 1.95,
        "odds_btts_yes": 1.80, "odds_btts_no": 2.00,
    }])
    sigs = generate(df)
    home_sig = next(s for s in sigs if s.market == "1X2" and s.pick == "HOME")
    assert home_sig.is_value
    assert home_sig.edge > 0.05
    assert home_sig.stake_units > 0


def test_no_signal_when_low_confidence():
    df = pd.DataFrame([{
        "match_id": 1,
        "p_home": 0.40, "p_draw": 0.30, "p_away": 0.30,
        "p_over25": 0.50, "p_btts": 0.50,
    }])
    sigs = generate(df)
    assert sigs == []


def test_model_only_signal_without_odds():
    settings.min_confidence = 0.55
    df = pd.DataFrame([{
        "match_id": 1,
        "p_home": 0.72, "p_draw": 0.18, "p_away": 0.10,
        "p_over25": 0.50, "p_btts": 0.50,
    }])
    sigs = generate(df)
    assert any(s.market == "1X2" and s.pick == "HOME" and not s.is_value for s in sigs)
