from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from loguru import logger

from src.config import settings


def _normalize_url(url: str) -> str:
    # Railway provides postgres://... — SQLAlchemy + asyncpg wants postgresql+asyncpg://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


DATABASE_URL = _normalize_url(settings.database_url)

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    short_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition: Mapped[str] = mapped_column(String(16), index=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    utc_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    home_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    market: Mapped[str] = mapped_column(String(32))  # 1X2, OU25, BTTS
    pick: Mapped[str] = mapped_column(String(16))    # HOME/DRAW/AWAY, OVER/UNDER, YES/NO
    model_prob: Mapped[float] = mapped_column(Float)
    fair_odds: Mapped[float] = mapped_column(Float)
    book_odds: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    stake_units: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    settled: Mapped[bool] = mapped_column(Boolean, default=False)
    won: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    profit_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    commentary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    match = relationship("Match")

    __table_args__ = (
        UniqueConstraint("match_id", "market", "pick", name="uq_signal_match_market_pick"),
    )


class Subscriber(Base):
    __tablename__ = "subscribers"
    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    subscribed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        try:
            await conn.execute(text("ALTER TABLE signals ADD COLUMN IF NOT EXISTS commentary TEXT"))
        except Exception as e:
            logger.warning(f"commentary column migration skipped: {e}")
