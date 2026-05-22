from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, Message, ReplyKeyboardRemove
from loguru import logger
from sqlalchemy import and_, select

from src.bot.access import is_allowed
from src.bot.formatters import HELP, WELCOME, format_roi, format_signal, format_signal_short, format_stats_table
from src.bot.keyboards import admin_menu, filters_menu, main_menu
from src.config import settings
from src.data.database import Match, SessionLocal, Signal, Subscriber, Team
from src.data.settings_store import get_bool, set_bool
from src.signals.tracker import roi_stats


class AdminFSM(StatesGroup):
    waiting_add = State()
    waiting_remove = State()


router = Router()


# ─────────────────────────── COMMANDS ───────────────────────────


@router.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    is_admin = msg.from_user and msg.from_user.id in settings.admin_ids
    ai_on = await get_bool("ai_ensemble_enabled", False)
    sub_active = await _is_subscribed(msg.chat.id)
    if is_admin:
        await msg.answer(
            "👋 Привет, админ! Управляй ботом через панель ниже.",
            parse_mode="HTML",
            reply_markup=admin_menu(ai_on),
        )
        await msg.answer(WELCOME, parse_mode="HTML", reply_markup=main_menu(sub_active, ai_on))
        return
    if await is_allowed(msg.chat.id):
        await msg.answer(WELCOME, parse_mode="HTML", reply_markup=main_menu(sub_active, ai_on))
        return
    locked = (
        "🔒 Доступ ограничен.\n\n"
        f"Твой ID: <code>{msg.chat.id}</code>\n\n"
        "Перешли этот номер админу — он откроет тебе доступ."
    )
    await msg.answer(locked, parse_mode="HTML")


@router.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(HELP, parse_mode="HTML")


@router.message(Command("menu"))
async def cmd_menu(msg: Message) -> None:
    sub_active = await _is_subscribed(msg.chat.id)
    ai_on = await get_bool("ai_ensemble_enabled", False)
    await msg.answer("Меню:", reply_markup=main_menu(sub_active, ai_on))


@router.message(Command("subscribe"))
async def cmd_subscribe(msg: Message) -> None:
    await _subscribe(msg.chat.id, msg.from_user.username if msg.from_user else None)
    await msg.answer("✅ Подписка активна.")


@router.message(Command("unsubscribe"))
async def cmd_unsubscribe(msg: Message) -> None:
    await _unsubscribe(msg.chat.id)
    await msg.answer("👋 Отписан.")


@router.message(Command("signals"))
async def cmd_signals(msg: Message) -> None:
    await _send_signals(msg, league=None, market=None, only_value=False)


@router.message(Command("today"))
async def cmd_today(msg: Message) -> None:
    await _send_today(msg)


@router.message(Command("stats"))
async def cmd_stats(msg: Message) -> None:
    await _send_stats(msg)


@router.message(Command("chart"))
async def cmd_chart(msg: Message) -> None:
    await _send_chart(msg)


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


@router.message(Command("allow"))
async def cmd_allow(msg: Message) -> None:
    if not msg.from_user or msg.from_user.id not in settings.admin_ids:
        return
    parts = (msg.text or "").split(maxsplit=2)
    if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
        await msg.answer("Использование: <code>/allow &lt;chat_id&gt; [username]</code>", parse_mode="HTML")
        return
    chat_id = int(parts[1])
    username = parts[2].lstrip("@") if len(parts) > 2 else None
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, chat_id)
        if sub is None:
            session.add(Subscriber(chat_id=chat_id, username=username, active=True))
        else:
            sub.active = True
            if username:
                sub.username = username
        await session.commit()
    notified = True
    try:
        await msg.bot.send_message(chat_id, "✅ Доступ открыт! Напиши /start.")
    except TelegramForbiddenError:
        notified = False
    except Exception as e:
        logger.warning(f"/allow notify failed for {chat_id}: {e}")
        notified = False
    if notified:
        await msg.answer(f"✅ <code>{chat_id}</code> добавлен и уведомлён.", parse_mode="HTML")
    else:
        await msg.answer(
            f"✅ <code>{chat_id}</code> добавлен. Уведомить не получилось — "
            "пусть сам напишет /start боту.",
            parse_mode="HTML",
        )


@router.message(Command("deny"))
async def cmd_deny(msg: Message) -> None:
    if not msg.from_user or msg.from_user.id not in settings.admin_ids:
        return
    parts = (msg.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
        await msg.answer("Использование: <code>/deny &lt;chat_id&gt;</code>", parse_mode="HTML")
        return
    chat_id = int(parts[1])
    await _unsubscribe(chat_id)
    await msg.answer(f"🚫 <code>{chat_id}</code> отозван.", parse_mode="HTML")


@router.message(Command("allowed"))
async def cmd_allowed(msg: Message) -> None:
    if not msg.from_user or msg.from_user.id not in settings.admin_ids:
        return
    await _send_subscribers(msg)


async def _send_subscribers(msg: Message) -> None:
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(Subscriber).where(Subscriber.active.is_(True))
            .order_by(Subscriber.subscribed_at.desc())
            .limit(50)
        )).scalars().all()
    if not rows:
        await msg.answer("Активных подписчиков нет.")
        return
    lines = [f"<b>Активные подписчики ({len(rows)}):</b>"]
    for s in rows:
        uname = f"@{s.username}" if s.username else "—"
        date = s.subscribed_at.strftime("%Y-%m-%d")
        lines.append(f"<code>{s.chat_id}</code>  {uname}  {date}")
    await msg.answer("\n".join(lines), parse_mode="HTML")


# ─────────────────────────── ADMIN PANEL BUTTONS ───────────────────────────


def _is_admin(msg: Message) -> bool:
    return bool(msg.from_user and msg.from_user.id in settings.admin_ids)


@router.message(F.text == "👥 Подписчики")
async def btn_subscribers(msg: Message) -> None:
    if not _is_admin(msg):
        return
    await _send_subscribers(msg)


@router.message(F.text == "➕ Добавить")
async def btn_add(msg: Message, state: FSMContext) -> None:
    if not _is_admin(msg):
        return
    await state.set_state(AdminFSM.waiting_add)
    await msg.answer(
        "Введи <b>chat_id</b> пользователя, которому хочешь открыть доступ.\n\n"
        "Пример: <code>123456789</code>\n\n"
        "Или /cancel для отмены.",
        parse_mode="HTML",
    )


@router.message(F.text == "🚫 Удалить")
async def btn_remove(msg: Message, state: FSMContext) -> None:
    if not _is_admin(msg):
        return
    await state.set_state(AdminFSM.waiting_remove)
    await msg.answer(
        "Введи <b>chat_id</b> пользователя, которому хочешь закрыть доступ.\n\n"
        "Пример: <code>123456789</code>\n\n"
        "Или /cancel для отмены.",
        parse_mode="HTML",
    )


@router.message(F.text == "📊 Статистика")
async def btn_stats(msg: Message) -> None:
    if not _is_admin(msg):
        return
    await _send_stats(msg)


@router.message(F.text == "🎯 Сигналы")
async def btn_signals(msg: Message) -> None:
    if not _is_admin(msg):
        return
    await _send_signals(msg, league=None, market=None, only_value=False)


@router.message(F.text == "📅 Сегодня")
async def btn_today(msg: Message) -> None:
    if not _is_admin(msg):
        return
    await _send_today(msg)


@router.message(F.text.regexp(r"^🧠 AI"))
async def btn_ai_toggle(msg: Message) -> None:
    if not _is_admin(msg):
        return
    cur = await get_bool("ai_ensemble_enabled", False)
    new = not cur
    await set_bool("ai_ensemble_enabled", new)
    status = "🟢 ВКЛ" if new else "🔴 ВЫКЛ"
    detail = (
        "Claude участвует в анализе топ-кандидатов с веб-поиском. "
        "Эффект увидишь в следующем цикле генерации."
        if new
        else "Только XGBoost. AI отключён."
    )
    await msg.answer(
        f"AI-ансамбль: <b>{status}</b>\n\n{detail}",
        parse_mode="HTML",
        reply_markup=admin_menu(new),
    )


@router.message(Command("cancel"))
async def cmd_cancel(msg: Message, state: FSMContext) -> None:
    await state.clear()
    if _is_admin(msg):
        ai_on = await get_bool("ai_ensemble_enabled", False)
        await msg.answer("Отменено.", reply_markup=admin_menu(ai_on))
    else:
        await msg.answer("Отменено.", reply_markup=ReplyKeyboardRemove())


@router.message(AdminFSM.waiting_add)
async def fsm_add_user(msg: Message, state: FSMContext) -> None:
    text = (msg.text or "").strip()
    if not text.lstrip("-").isdigit():
        await msg.answer("Нужно ввести числовой chat_id. Попробуй ещё раз или /cancel.")
        return
    chat_id = int(text)
    await state.clear()
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, chat_id)
        if sub is None:
            session.add(Subscriber(chat_id=chat_id, active=True))
        else:
            sub.active = True
        await session.commit()
    notified = True
    try:
        await msg.bot.send_message(chat_id, "✅ Доступ открыт! Напиши /start.")
    except TelegramForbiddenError:
        notified = False
    except Exception as e:
        logger.warning(f"notify failed for {chat_id}: {e}")
        notified = False
    if notified:
        await msg.answer(f"✅ <code>{chat_id}</code> добавлен и уведомлён.", parse_mode="HTML", reply_markup=admin_menu(await get_bool("ai_ensemble_enabled", False)))
    else:
        await msg.answer(
            f"✅ <code>{chat_id}</code> добавлен. Уведомить не получилось — "
            "пусть сам напишет /start боту.",
            parse_mode="HTML",
            reply_markup=admin_menu(await get_bool("ai_ensemble_enabled", False)),
        )


@router.message(AdminFSM.waiting_remove)
async def fsm_remove_user(msg: Message, state: FSMContext) -> None:
    text = (msg.text or "").strip()
    if not text.lstrip("-").isdigit():
        await msg.answer("Нужно ввести числовой chat_id. Попробуй ещё раз или /cancel.")
        return
    chat_id = int(text)
    await state.clear()
    await _unsubscribe(chat_id)
    await msg.answer(f"🚫 <code>{chat_id}</code> удалён.", parse_mode="HTML", reply_markup=admin_menu(await get_bool("ai_ensemble_enabled", False)))


# ─────────────────────────── CALLBACKS ───────────────────────────


@router.callback_query(F.data.startswith("menu:"))
async def cb_menu(q: CallbackQuery) -> None:
    action = q.data.split(":", 1)[1]
    await q.answer()
    if action == "signals":
        await _send_signals(q.message, league=None, market=None, only_value=False)
    elif action == "today":
        await _send_today(q.message)
    elif action == "stats":
        await _send_stats(q.message)
    elif action == "chart":
        await _send_chart(q.message)
    elif action == "subscribe":
        await _subscribe(q.message.chat.id, q.from_user.username)
        ai_on = await get_bool("ai_ensemble_enabled", False)
        await q.message.answer("✅ Подписка активна.", reply_markup=main_menu(True, ai_on))
    elif action == "unsubscribe":
        await _unsubscribe(q.message.chat.id)
        ai_on = await get_bool("ai_ensemble_enabled", False)
        await q.message.answer("👋 Отписан.", reply_markup=main_menu(False, ai_on))
    elif action == "ai_info":
        ai_on = await get_bool("ai_ensemble_enabled", False)
        status = "🟢 ВКЛ" if ai_on else "🔴 ВЫКЛ"
        await q.message.answer(
            f"🧠 AI-ансамбль управляется админом.\nТекущий статус: <b>{status}</b>",
            parse_mode="HTML",
        )
    elif action == "filters":
        await q.message.answer("Выбери фильтр:", reply_markup=filters_menu())
    elif action == "back":
        sub_active = await _is_subscribed(q.message.chat.id)
        ai_on = await get_bool("ai_ensemble_enabled", False)
        await q.message.answer("Меню:", reply_markup=main_menu(sub_active, ai_on))


@router.callback_query(F.data.startswith("filter:"))
async def cb_filter(q: CallbackQuery) -> None:
    _, kind, value = q.data.split(":", 2)
    await q.answer()
    if kind == "league":
        await _send_signals(q.message, league=value, market=None, only_value=False)
    elif kind == "market":
        await _send_signals(q.message, league=None, market=value, only_value=False)
    elif kind == "type":
        only_value = value == "VALUE"
        await _send_signals(q.message, league=None, market=None, only_value=only_value)


# ─────────────────────────── HELPERS ───────────────────────────


async def _is_subscribed(chat_id: int) -> bool:
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, chat_id)
    return bool(sub and sub.active)


async def _subscribe(chat_id: int, username: Optional[str]) -> None:
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, chat_id)
        if sub is None:
            session.add(Subscriber(chat_id=chat_id, username=username, active=True))
        else:
            sub.active = True
        await session.commit()


async def _unsubscribe(chat_id: int) -> None:
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, chat_id)
        if sub:
            sub.active = False
            await session.commit()


async def _send_signals(
    msg: Message,
    league: Optional[str],
    market: Optional[str],
    only_value: bool,
) -> None:
    now = datetime.utcnow()
    horizon = now + timedelta(days=3)
    async with SessionLocal() as session:
        stmt = (
            select(Signal, Match)
            .join(Match, Match.id == Signal.match_id)
            .where(and_(Match.utc_date >= now, Match.utc_date <= horizon))
        )
        if league:
            stmt = stmt.where(Match.competition == league)
        if market:
            stmt = stmt.where(Signal.market == market)
        if only_value:
            stmt = stmt.where(Signal.book_odds > 1.0)
        stmt = stmt.order_by(Signal.edge.desc(), Signal.confidence.desc()).limit(10)
        pairs = (await session.execute(stmt)).all()
        if not pairs:
            await msg.answer("Под этот фильтр сигналов нет. Попробуй другой.")
            return
        for sig, match in pairs:
            home = await session.get(Team, match.home_team_id)
            away = await session.get(Team, match.away_team_id)
            await msg.answer(
                format_signal(sig, match, home, away, sig.commentary),
                parse_mode="HTML",
            )


async def _send_today(msg: Message) -> None:
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
            sigs = (await session.execute(
                select(Signal).where(Signal.match_id == m.id)
            )).scalars().all()
            lines.append(f"• {t} <i>{m.competition}</i> — {h.name} vs {a.name}")
            lines.append(f"   {format_signal_short(sigs)}")
    await msg.answer("\n".join(lines), parse_mode="HTML")


async def _send_stats(msg: Message) -> None:
    model_s = await roi_stats(only_value=False)
    value_s = await roi_stats(only_value=True)
    ai_s = await roi_stats(only_value=None, ai_only=True)
    total_s = await roi_stats(only_value=None)
    text = format_stats_table(model_s, value_s, ai_s, total_s)
    await msg.answer(text, parse_mode="HTML")


async def _send_chart(msg: Message) -> None:
    png = await _build_roi_chart()
    if png is None:
        await msg.answer("Пока нет рассчитанных ставок — нечего рисовать.")
        return
    await msg.answer_photo(
        BufferedInputFile(png, filename="roi.png"),
        caption="📊 Кумулятивная прибыль (ед.)",
    )


async def _build_roi_chart() -> bytes | None:
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(Signal).where(Signal.settled.is_(True)).order_by(Signal.created_at)
        )).scalars().all()
    if not rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — /chart unavailable")
        return None
    xs = list(range(1, len(rows) + 1))
    cum = []
    running = 0.0
    for r in rows:
        running += r.profit_units or 0.0
        cum.append(running)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(xs, cum, linewidth=2, color="#2E86AB")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.fill_between(xs, cum, 0, where=[c >= 0 for c in cum], alpha=0.2, color="#2E86AB")
    ax.fill_between(xs, cum, 0, where=[c < 0 for c in cum], alpha=0.2, color="#E63946")
    ax.set_xlabel("Номер ставки")
    ax.set_ylabel("Прибыль, ед.")
    ax.set_title(f"Кумулятивная прибыль: {cum[-1]:+.2f} ед. за {len(rows)} ставок")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ─────────────────────────── BROADCAST ───────────────────────────


def register(dp: Dispatcher) -> None:
    dp.include_router(router)


async def broadcast_signal(bot: Bot, text: str) -> int:
    sent = 0
    async with SessionLocal() as session:
        subs = (await session.execute(
            select(Subscriber).where(Subscriber.active.is_(True))
        )).scalars().all()
    if not subs:
        logger.warning("broadcast_signal: no active subscribers — nobody to send to")
        return 0
    for s in subs:
        try:
            await bot.send_message(s.chat_id, text, parse_mode="HTML")
            sent += 1
        except TelegramForbiddenError:
            logger.info(f"Subscriber {s.chat_id} auto-deactivated: TelegramForbiddenError")
            await _unsubscribe(s.chat_id)
        except Exception as e:
            logger.warning(f"Send to {s.chat_id} failed: {e}")
    return sent


async def broadcast_digest(bot: Bot) -> int:
    """Morning digest: list of today's signals (top 5 by edge)."""
    now = datetime.utcnow()
    end_of_day = now.replace(hour=23, minute=59, second=59)
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(Signal, Match)
            .join(Match, Match.id == Signal.match_id)
            .where(and_(Match.utc_date >= now, Match.utc_date <= end_of_day))
            .order_by(Signal.edge.desc(), Signal.confidence.desc())
            .limit(5)
        )).all()
        if not rows:
            return 0
        lines = [f"☀️ <b>Утренний дайджест — {now.strftime('%d.%m.%Y')}</b>", ""]
        for sig, m in rows:
            h = await session.get(Team, m.home_team_id)
            a = await session.get(Team, m.away_team_id)
            badge = "🎯" if sig.book_odds > 1.0 else "🤖"
            kickoff = m.utc_date.strftime("%H:%M")
            lines.append(
                f"{badge} {kickoff} <i>{m.competition}</i> — {h.name} vs {a.name}\n"
                f"   <b>{sig.market}</b> · <b>{sig.pick}</b> · {sig.model_prob*100:.0f}%"
                + (f" · edge {sig.edge*100:.0f}%" if sig.book_odds > 1.0 else "")
            )
        text = "\n".join(lines)
    sent = 0
    async with SessionLocal() as session:
        subs = (await session.execute(
            select(Subscriber).where(Subscriber.active.is_(True))
        )).scalars().all()
    for s in subs:
        try:
            await bot.send_message(s.chat_id, text, parse_mode="HTML")
            sent += 1
        except TelegramForbiddenError:
            logger.info(f"Subscriber {s.chat_id} auto-deactivated: TelegramForbiddenError")
            await _unsubscribe(s.chat_id)
        except Exception as e:
            logger.warning(f"Digest to {s.chat_id} failed: {e}")
    return sent
