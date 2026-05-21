from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def _split_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip().lstrip("-").isdigit()]


def _split_upper(raw: str, default: List[str]) -> List[str]:
    if not raw or not raw.strip():
        return default
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


class Settings(BaseSettings):
    """Plain scalar fields only. Lists are read from os.environ via properties
    so we don't depend on pydantic-settings' JSON-decoding behaviour for env
    vars (which has changed across versions and breaks on plain CSV input)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    telegram_bot_token: str = ""
    football_data_api_key: str = ""
    api_football_key: str = ""
    odds_api_key: str = ""
    anthropic_api_key: str = ""

    database_url: str = "sqlite+aiosqlite:///./bot.db"

    min_edge: float = 0.05
    min_confidence: float = 0.55
    min_odds: float = 1.50
    max_odds: float = 4.50

    tz: str = "Europe/Moscow"

    @property
    def admin_ids(self) -> List[int]:
        return _split_ints(os.getenv("ADMIN_IDS", ""))

    @property
    def competitions(self) -> List[str]:
        return _split_upper(
            os.getenv("COMPETITIONS", ""),
            default=["PL", "PD", "SA", "BL1", "FL1", "CL"],
        )


settings = Settings()
