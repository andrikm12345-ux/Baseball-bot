from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.config import MODELS_DIR
from src.data.features import FEATURE_COLUMNS


class Predictor:
    def __init__(self) -> None:
        self.m_1x2 = self._load(MODELS_DIR / "model_1x2.joblib")
        self.m_ou = self._load(MODELS_DIR / "model_ou25.joblib")
        self.m_btts = self._load(MODELS_DIR / "model_btts.joblib")

    @staticmethod
    def _load(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Cannot load {path}: {e}")
            return None

    @property
    def ready(self) -> bool:
        return all([self.m_1x2, self.m_ou, self.m_btts])

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if not self.ready:
            raise RuntimeError("Models not trained yet. Run training first.")
        X = features_df[FEATURE_COLUMNS].astype(float).values
        p_1x2 = self.m_1x2["model"].predict_proba(X)
        p_ou = self.m_ou["model"].predict_proba(X)[:, 1]
        p_btts = self.m_btts["model"].predict_proba(X)[:, 1]
        out = features_df[["match_id"]].copy()
        out["p_home"] = p_1x2[:, 0]
        out["p_draw"] = p_1x2[:, 1]
        out["p_away"] = p_1x2[:, 2]
        out["p_over25"] = p_ou
        out["p_btts"] = p_btts
        return out
