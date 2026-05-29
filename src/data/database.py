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
    market: Mapped[str] = mapped_column(String(32))  # 1X2, TOTAL, HANDICAP
    pick: Mapped[str] = mapped_column(String(16))    # HOME/DRAW/AWAY, OVER/UNDER
    line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # total/handicap line
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
    is_ai_ensemble: Mapped[bool] = mapped_column(Boolean, default=False)

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
    notifications_enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class Setting(Base):
    __tablename__ = "settings_kv"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class PendingUser(Base):
    __tablename__ = "pending_users"
    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    start_count: Mapped[int] = mapped_column(Integer, default=1)


class AiPrediction(Base):
    __tablename__ = "ai_predictions"
    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
    payload: Mapped[str] = mapped_column(String)  # JSON-encoded ai_predict result


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        try:
            await conn.execute(text("ALTER TABLE signals ADD COLUMN IF NOT EXISTS commentary TEXT"))
        except Exception as e:
            logger.warning(f"commentary column migration skipped: {e}")
        try:
            await conn.execute(text("ALTER TABLE signals ADD COLUMN IF NOT EXISTS line FLOAT"))
        except Exception as e:
            logger.warning(f"line column migration skipped: {e}")
        try:
            await conn.execute(text(
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS is_ai_ensemble BOOLEAN DEFAULT FALSE"
            ))
        except Exception as e:
            logger.warning(f"is_ai_ensemble column migration skipped: {e}")
        try:
            await conn.execute(text(
                "ALTER TABLE subscribers "
                "ADD COLUMN IF NOT EXISTS notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE"
            ))
        except Exception as e:
            logger.warning(f"notifications_enabled column migration skipped: {e}")
        try:
            # One-time amnesty for users kicked by the old buggy unsubscribe flow:
            # restore access, keep notifications off so we don't spam them unprompted.
            result = await conn.execute(text(
                "UPDATE subscribers SET active = TRUE, notifications_enabled = FALSE "
                "WHERE active = FALSE"
            ))
            if result.rowcount:
                logger.warning(f"Amnesty: restored access for {result.rowcount} users (notifications off)")
        except Exception as e:
            logger.warning(f"amnesty UPDATE skipped: {e}")
        try:
            # Drop legacy commentary from non-AI signals: generate_commentary
            # used to attach text to every signal, leaving 🧠-looking commentary
            # on plain MODEL/VALUE rows. Idempotent on later boots.
            result = await conn.execute(text(
                "UPDATE signals SET commentary = NULL "
                "WHERE commentary IS NOT NULL AND is_ai_ensemble = FALSE"
            ))
            if result.rowcount:
                logger.warning(
                    f"Stripped stale commentary from {result.rowcount} non-AI signals"
                )
        except Exception as e:
            logger.warning(f"commentary cleanup skipped: {e}")
        try:
            # Collapse duplicate unsettled signals per match: keep one strongest
            # pick (VALUE before MODEL, then highest edge, then highest
            # confidence). Settled rows stay so historical ROI is unchanged.
            # Idempotent on subsequent boots.
            result = await conn.execute(text("""
                DELETE FROM signals WHERE id IN (
                  SELECT id FROM (
                    SELECT id,
                      ROW_NUMBER() OVER (
                        PARTITION BY match_id
                        ORDER BY
                          CASE WHEN book_odds > 1.0 THEN 0 ELSE 1 END,
                          edge DESC NULLS LAST,
                          confidence DESC NULLS LAST,
                          id ASC
                      ) AS rn
                    FROM signals
                    WHERE settled = FALSE
                  ) ranked
                  WHERE rn > 1
                )
            """))
            if result.rowcount:
                logger.warning(
                    f"Collapsed {result.rowcount} duplicate unsettled signals to one per match"
                )
        except Exception as e:
            logger.warning(f"unsettled-dedup skipped: {e}")
