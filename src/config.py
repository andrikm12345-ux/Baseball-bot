from __future__ import annotations

from pathlib import Path
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    telegram_bot_token: str = ""
    admin_ids: Annotated[List[int], NoDecode] = Field(default_factory=list)

    football_data_api_key: str = ""
    api_football_key: str = ""
    competitions: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["PL", "PD", "SA", "BL1", "FL1", "CL"]
    )

    database_url: str = "sqlite+aiosqlite:///./bot.db"

    min_edge: float = 0.05
    min_confidence: float = 0.55
    min_odds: float = 1.50
    max_odds: float = 4.50

    tz: str = "Europe/Moscow"

    @field_validator("admin_ids", mode="before")
    @classmethod
    def _split_admins(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v

    @field_validator("competitions", mode="before")
    @classmethod
    def _split_comps(cls, v):
        if v is None or v == "":
            return ["PL", "PD", "SA", "BL1", "FL1", "CL"]
        if isinstance(v, str):
            return [x.strip().upper() for x in v.split(",") if x.strip()]
        return v


settings = Settings()
