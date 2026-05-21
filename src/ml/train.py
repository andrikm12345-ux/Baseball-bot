"""Train three calibrated XGBoost models: 1X2, Over/Under 2.5, BTTS."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import MODELS_DIR
from src.data.features import FEATURE_COLUMNS


def _make_estimator(multiclass: bool) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="multi:softprob" if multiclass else "binary:logistic",
        eval_metric="mlogloss" if multiclass else "logloss",
        n_jobs=-1,
        tree_method="hist",
    )


def _train_one(X: pd.DataFrame, y: np.ndarray, multiclass: bool, name: str) -> CalibratedClassifierCV:
    base = _make_estimator(multiclass)
    # isotonic calibration with TimeSeries split — better for probabilistic betting decisions
    cal = CalibratedClassifierCV(base, method="isotonic", cv=TimeSeriesSplit(n_splits=4))
    cal.fit(X, y)
    proba = cal.predict_proba(X)
    if multiclass:
        ll = log_loss(y, proba, labels=[0, 1, 2])
        logger.info(f"[{name}] in-sample multiclass logloss={ll:.4f}")
    else:
        bs = brier_score_loss(y, proba[:, 1])
        logger.info(f"[{name}] in-sample Brier={bs:.4f}")
    return cal


def train_all(features_df: pd.DataFrame) -> Dict[str, Path]:
    if features_df.empty or len(features_df) < 200:
        raise RuntimeError(
            f"Not enough training data ({len(features_df)} rows). "
            "Run the history ingest first."
        )
    features_df = features_df.dropna(subset=["outcome", "over25", "btts"]).reset_index(drop=True)
    X = features_df[FEATURE_COLUMNS].astype(float)

    paths: Dict[str, Path] = {}

    logger.info(f"Training 1X2 on {len(X)} rows")
    m_1x2 = _train_one(X, features_df["outcome"].astype(int).values, multiclass=True, name="1X2")
    p = MODELS_DIR / "model_1x2.joblib"
    joblib.dump({"model": m_1x2, "features": FEATURE_COLUMNS}, p)
    paths["1X2"] = p

    logger.info(f"Training OU2.5 on {len(X)} rows")
    m_ou = _train_one(X, features_df["over25"].astype(int).values, multiclass=False, name="OU2.5")
    p = MODELS_DIR / "model_ou25.joblib"
    joblib.dump({"model": m_ou, "features": FEATURE_COLUMNS}, p)
    paths["OU25"] = p

    logger.info(f"Training BTTS on {len(X)} rows")
    m_btts = _train_one(X, features_df["btts"].astype(int).values, multiclass=False, name="BTTS")
    p = MODELS_DIR / "model_btts.joblib"
    joblib.dump({"model": m_btts, "features": FEATURE_COLUMNS}, p)
    paths["BTTS"] = p

    return paths


def evaluate_walk_forward(features_df: pd.DataFrame) -> Dict[str, float]:
    """Honest out-of-sample evaluation via expanding-window CV."""
    features_df = features_df.dropna(subset=["outcome", "over25", "btts"]).reset_index(drop=True)
    if len(features_df) < 400:
        logger.warning("Not enough rows for walk-forward eval")
        return {}
    X = features_df[FEATURE_COLUMNS].astype(float).values
    y_1x2 = features_df["outcome"].astype(int).values
    y_ou = features_df["over25"].astype(int).values
    y_btts = features_df["btts"].astype(int).values
    tscv = TimeSeriesSplit(n_splits=5)
    metrics: Dict[str, list] = {"1x2_logloss": [], "ou_brier": [], "btts_brier": []}
    for tr, te in tscv.split(X):
        m = _make_estimator(True).fit(X[tr], y_1x2[tr])
        metrics["1x2_logloss"].append(log_loss(y_1x2[te], m.predict_proba(X[te]), labels=[0, 1, 2]))
        m = _make_estimator(False).fit(X[tr], y_ou[tr])
        metrics["ou_brier"].append(brier_score_loss(y_ou[te], m.predict_proba(X[te])[:, 1]))
        m = _make_estimator(False).fit(X[tr], y_btts[tr])
        metrics["btts_brier"].append(brier_score_loss(y_btts[te], m.predict_proba(X[te])[:, 1]))
    return {k: float(np.mean(v)) for k, v in metrics.items()}
