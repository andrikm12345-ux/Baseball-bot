from __future__ import annotations

from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from loguru import logger
from sqlalchemy import and_, select

from src.bot.formatters import HELP, WELCOME, format_roi, format_signal
from src.config import settings
from src.data.database import Match, SessionLocal, Signal, Subscriber, Team
from src.signals.tracker import roi_stats


router = Router()


@router.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    await msg.answer(WELCOME, parse_mode="HTML")


@router.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(HELP, parse_mode="HTML")


@router.message(Command("subscribe"))
async def cmd_subscribe(msg: Message) -> None:
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, msg.chat.id)
        if sub is None:
            session.add(Subscriber(
                chat_id=msg.chat.id,
                username=msg.from_user.username if msg.from_user else None,
                active=True,
            ))
        else:
            sub.active = True
        await session.commit()
    await msg.answer("✅ Подписка активна. Буду слать сигналы по мере их появления.")


@router.message(Command("unsubscribe"))
async def cmd_unsubscribe(msg: Message) -> None:
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, msg.chat.id)
        if sub:
            sub.active = False
            await session.commit()
    await msg.answer("👋 Отписан. Сигналы больше не приходят.")


@router.message(Command("signals"))
async def cmd_signals(msg: Message) -> None:
    now = datetime.utcnow()
    horizon = now + timedelta(days=3)
    async with SessionLocal() as session:
        q = await session.execute(
            select(Signal, Match, Team, Team.__table__.alias("away_t"))
            .join(Match, Match.id == Signal.match_id)
            .where(Match.utc_date.between(now, horizon))
            .order_by(Signal.confidence.desc())
            .limit(10)
        )
        # simpler: pull signals + match, then load teams
        result = await session.execute(
            select(Signal, Match)
            .join(Match, Match.id == Signal.match_id)
            .where(and_(Match.utc_date >= now, Match.utc_date <= horizon))
            .order_by(Signal.edge.desc(), Signal.confidence.desc())
            .limit(10)
        )
        pairs = result.all()
        if not pairs:
            await msg.answer("Пока нет сигналов на ближайшие дни. Загляни позже.")
            return
        for sig, match in pairs:
            home = await session.get(Team, match.home_team_id)
            away = await session.get(Team, match.away_team_id)
            await msg.answer(format_signal(sig, match, home, away), parse_mode="HTML")


@router.message(Command("today"))
async def cmd_today(msg: Message) -> None:
    today = datetime.utcnow().date()
    start = datetime.combine(today, datetime.min.time())
    end = start + timedelta(days=1)
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(Match).where(and_(Match.utc_date >= start, Match.utc_date < end))
            .order_by(Match.utc_date)
        )).scalars().all()
        if not rows:
            await msg.answer("Сегодня матчей в отслеживаемых турнирах не нашёл.")
            return
        lines = ["<b>Матчи сегодня:</b>"]
        for m in rows[:20]:
            h = await session.get(Team, m.home_team_id)
            a = await session.get(Team, m.away_team_id)
            t = m.utc_date.strftime("%H:%M")
            lines.append(f"• {t} <i>{m.competition}</i> — {h.name} vs {a.name}")
    await msg.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("stats"))
async def cmd_stats(msg: Message) -> None:
    all_stats = await roi_stats(only_value=False)
    val_stats = await roi_stats(only_value=True)
    last30 = await roi_stats(last_n=30, only_value=False)
    text = (
        format_roi(all_stats, "Общий ROI") + "\n\n"
        + format_roi(val_stats, "Только VALUE-сигналы") + "\n\n"
        + format_roi(last30, "Последние 30 ставок")
    )
    await msg.answer(text, parse_mode="HTML")


@router.message(Command("admin"))
async def cmd_admin(msg: Message) -> None:
    if not msg.from_user or msg.from_user.id not in settings.admin_ids:
        return
    async with SessionLocal() as session:
        n_subs = len((await session.execute(
            select(Subscriber).where(Subscriber.active.is_(True))
        )).scalars().all())
        n_sig = len((await session.execute(select(Signal))).scalars().all())
        n_match = len((await session.execute(select(Match))).scalars().all())
    await msg.answer(
        f"<b>Admin</b>\nПодписчики: {n_subs}\nСигналы в БД: {n_sig}\nМатчи в БД: {n_match}",
        parse_mode="HTML",
    )


def register(dp: Dispatcher) -> None:
    dp.include_router(router)


async def broadcast_signal(bot: Bot, text: str) -> int:
    sent = 0
    async with SessionLocal() as session:
        subs = (await session.execute(
            select(Subscriber).where(Subscriber.active.is_(True))
        )).scalars().all()
    for s in subs:
        try:
            await bot.send_message(s.chat_id, text, parse_mode="HTML")
            sent += 1
        except Exception as e:
            logger.warning(f"Send to {s.chat_id} failed: {e}")
    return sent
