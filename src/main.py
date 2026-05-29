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
    daily_stats_broadcast,
    generate_and_broadcast,
    refresh_upcoming,
)
from src.signals.tracker import settle_pending


async def _setup_commands(bot: Bot) -> None:
    """Register the slash-command menu shown in Telegram's "/" picker."""
    from aiogram.types import (
        BotCommand, BotCommandScopeChat, BotCommandScopeDefault,
    )
    user_cmds = [
        BotCommand(command="signals", description="🎯 Текущие ставки"),
        BotCommand(command="today", description="📅 Матчи сегодня"),
        BotCommand(command="stats", description="📈 ROI и статистика"),
        BotCommand(command="chart", description="📊 График прибыли"),
        BotCommand(command="history", description="📜 История ставок"),
        BotCommand(command="help", description="ℹ️ Как работает бот"),
        BotCommand(command="menu", description="📋 Меню"),
        BotCommand(command="start", description="🚀 Старт"),
    ]
    try:
        await bot.set_my_commands(user_cmds, scope=BotCommandScopeDefault())
        admin_cmds = user_cmds + [
            BotCommand(command="diag", description="🩺 Диагностика сигналов"),
            BotCommand(command="breakdown", description="🔬 ROI по рынкам/лигам"),
            BotCommand(command="allowed", description="👥 Список подписчиков"),
            BotCommand(command="allow", description="✅ Открыть доступ"),
            BotCommand(command="deny", description="🚫 Закрыть доступ"),
        ]
        for admin_id in settings.admin_ids:
            try:
                await bot.set_my_commands(admin_cmds, scope=BotCommandScopeChat(chat_id=admin_id))
            except Exception as e:
                logger.warning(f"set_my_commands for admin {admin_id} failed: {e}")
    except Exception as e:
        logger.warning(f"set_my_commands failed: {e}")


async def _on_startup(bot: Bot) -> None:
    await init_db()
    logger.info("DB initialised")
    await _setup_commands(bot)
    await _purge_disabled_market_signals()
    # On the very first boot, bootstrap history in the background so the
    # bot is responsive immediately.
    asyncio.create_task(_first_boot_warmup(bot))
    # And on every boot — kick a signals_loop in the background so users
    # don't wait up to an hour for the next scheduled tick (after rare
    # purges or container restarts the DB can otherwise look empty).
    asyncio.create_task(_post_boot_generate(bot))


async def _post_boot_generate(bot: Bot) -> None:
    await asyncio.sleep(20)  # let polling settle first
    try:
        await refresh_upcoming(days=2)
        await generate_and_broadcast(bot)
    except Exception as e:
        logger.warning(f"post-boot generate failed: {e}")


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
    try:
        async with SessionLocal() as s:
            n = (await s.execute(select(func.count(Match.id)))).scalar_one()
        if n < 100:
            logger.info("Cold start: bootstrapping historical data")
            await bootstrap_history()
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
    # Morning digest at 09:10 — right after the stats digest at 09:00 so the
    # two messages arrive in the natural order (yesterday's results → today's
    # picks) without colliding on the same scheduler tick.
    scheduler.add_job(broadcast_digest, CronTrigger(hour=9, minute=10), args=[bot], id="digest")
    # End-of-day stats digest at 09:00 MSK. Late kicks (Brazil, late Italian
    # matches) finish around 04-06 MSK; pushing the digest to 09:00 lets
    # settle_pending pick them up before we summarise.
    scheduler.add_job(
        daily_stats_broadcast,
        CronTrigger(hour=9, minute=0),
        args=[bot],
        id="daily_stats",
    )
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
