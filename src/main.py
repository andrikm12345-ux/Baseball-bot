from __future__ import annotations

import asyncio
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from src.bot.access import AccessControlMiddleware
from src.bot.handlers import broadcast_digest, register
from src.config import settings
from src.data.database import init_db
from src.pipeline import (
    bootstrap_history,
    daily_cycle,
    generate_and_broadcast,
    refresh_upcoming,
    train_models,
)
from src.signals.tracker import settle_pending


async def _on_startup(bot: Bot) -> None:
    await init_db()
    logger.info("DB initialised")
    await _purge_disabled_market_signals()
    # On the very first boot, bootstrap history in the background so the
    # bot is responsive immediately.
    asyncio.create_task(_first_boot_warmup(bot))


async def _purge_disabled_market_signals() -> None:
    from sqlalchemy import delete
    from src.data.database import SessionLocal, Signal
    from src.pipeline import DISABLED_MARKETS

    async with SessionLocal() as session:
        result = await session.execute(
            delete(Signal).where(Signal.market.in_(DISABLED_MARKETS))
        )
        await session.commit()
        if result.rowcount:
            logger.warning(f"Purged {result.rowcount} stale ITB signals from DB")


async def _first_boot_warmup(bot: Bot) -> None:
    from sqlalchemy import select, func
    from src.data.database import Match, SessionLocal
    from src.ml.predict import Predictor
    try:
        async with SessionLocal() as s:
            n = (await s.execute(select(func.count(Match.id)))).scalar_one()
        cold_start = n < 100
        predictor_stale = not Predictor().ready
        if cold_start:
            logger.info("Cold start: bootstrapping historical data")
            await bootstrap_history()
        if cold_start or predictor_stale:
            logger.info(
                f"Training models (cold_start={cold_start}, stale={predictor_stale})"
            )
            await train_models(bot=bot)
            await refresh_upcoming(days=7)
            await generate_and_broadcast(bot)
    except Exception as e:
        logger.exception(f"Warm-up failed: {e}")


async def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set — exiting")
        sys.exit(1)

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    dp = Dispatcher()
    access_mw = AccessControlMiddleware()
    dp.message.middleware(access_mw)
    dp.callback_query.middleware(access_mw)
    register(dp)

    scheduler = AsyncIOScheduler(timezone=settings.tz)
    # Full pipeline once a day at 04:00 local time
    scheduler.add_job(daily_cycle, CronTrigger(hour=4, minute=0), args=[bot], id="daily")
    # Generate/publish signals every hour for matches kicking off in the next 4
    # hours, with fresh odds — limits stale-line bait.
    scheduler.add_job(
        generate_and_broadcast, IntervalTrigger(hours=1), args=[bot], id="signals_loop"
    )
    # Settle results every 30 min
    scheduler.add_job(settle_pending, IntervalTrigger(minutes=30), id="settle")
    # Pull upcoming matches every 6 hours
    scheduler.add_job(refresh_upcoming, IntervalTrigger(hours=6), id="refresh_upcoming")
    # Morning digest at 09:00 local time
    scheduler.add_job(broadcast_digest, CronTrigger(hour=9, minute=0), args=[bot], id="digest")
    scheduler.start()

    await _on_startup(bot)

    logger.info("Bot polling started")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(
            bot,
            allowed_updates=dp.resolve_used_update_types(),
            polling_timeout=10,
        )
    finally:
        scheduler.shutdown(wait=False)
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
