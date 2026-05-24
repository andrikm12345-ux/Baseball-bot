from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardRemove,
)
from loguru import logger
from sqlalchemy import and_, select

from src.bot.access import is_allowed
from src.bot.formatters import HELP, WELCOME, fmt_msk, format_roi, format_signal, format_signal_short, format_stats_table
from src.bot.keyboards import admin_menu, filters_menu, main_menu
from src.config import settings
from src.data.database import Match, PendingUser, SessionLocal, Signal, Subscriber, Team
from src.data.settings_store import get_bool, set_bool
from src.signals.tracker import roi_stats, settle_pending


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
    await _track_pending(msg)


async def _track_pending(msg: Message) -> None:
    u = msg.from_user
    try:
        async with SessionLocal() as session:
            p = await session.get(PendingUser, msg.chat.id)
            if p is None:
                session.add(PendingUser(
                    chat_id=msg.chat.id,
                    username=(u.username if u else None),
                    first_name=(u.first_name if u else None),
                    last_name=(u.last_name if u else None),
                    start_count=1,
                ))
            else:
                p.start_count += 1
                if u:
                    p.username = u.username or p.username
                    p.first_name = u.first_name or p.first_name
                    p.last_name = u.last_name or p.last_name
            await session.commit()
    except Exception as e:
        logger.warning(f"_track_pending failed for {msg.chat.id}: {e}")


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


@router.message(Command("history"))
async def cmd_history(msg: Message) -> None:
    await _send_history(msg)


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


@router.message(Command("diag"))
async def cmd_diag(msg: Message) -> None:
    """Admin: why are signals not settling. Also force-runs settle_pending."""
    if not msg.from_user or msg.from_user.id not in settings.admin_ids:
        return

    from sqlalchemy import func
    async with SessionLocal() as session:
        n_total = (await session.execute(
            select(func.count()).select_from(Signal)
        )).scalar_one()
        n_settled = (await session.execute(
            select(func.count()).select_from(Signal).where(Signal.settled.is_(True))
        )).scalar_one()
        n_won = (await session.execute(
            select(func.count()).select_from(Signal).where(Signal.won.is_(True))
        )).scalar_one()
        n_ai = (await session.execute(
            select(func.count()).select_from(Signal).where(Signal.is_ai_ensemble.is_(True))
        )).scalar_one()
        n_value = (await session.execute(
            select(func.count()).select_from(Signal).where(Signal.book_odds > 1.0)
        )).scalar_one()

        # Unsettled breakdown
        unsettled_rows = (await session.execute(
            select(Signal, Match)
            .join(Match, Match.id == Signal.match_id, isouter=True)
            .where(Signal.settled.is_(False))
        )).all()

        no_match = sum(1 for s, m in unsettled_rows if m is None)
        not_finished = sum(
            1 for s, m in unsettled_rows
            if m is not None and m.status != "FINISHED"
        )
        no_goals = sum(
            1 for s, m in unsettled_rows
            if m is not None and m.status == "FINISHED"
            and (m.home_goals is None or m.away_goals is None)
        )
        ready = sum(
            1 for s, m in unsettled_rows
            if m is not None and m.status == "FINISHED"
            and m.home_goals is not None and m.away_goals is not None
        )

        # Sample of unsettled rows for context
        sample_lines = []
        for s, m in unsettled_rows[:5]:
            if m is None:
                reason = "матч не в БД"
                meta = f"match_id={s.match_id}"
            elif m.status != "FINISHED":
                reason = f"status={m.status}"
                meta = f"kickoff={m.utc_date.strftime('%d.%m %H:%M')}"
            elif m.home_goals is None or m.away_goals is None:
                reason = "FINISHED без счёта"
                meta = f"kickoff={m.utc_date.strftime('%d.%m %H:%M')}"
            else:
                reason = "готов к settle"
                meta = f"{m.home_goals}:{m.away_goals}"
            sample_lines.append(f"  • sig#{s.id} {s.market}/{s.pick} — {reason} ({meta})")

    # If there is anything ready — fire settle right now
    forced_settled = 0
    if ready:
        forced_settled = await settle_pending()

    text = [
        "🩺 <b>ДИАГНОСТИКА</b>",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"Всего сигналов: <b>{n_total}</b>",
        f"  • settled: <b>{n_settled}</b> (won: {n_won})",
        f"  • unsettled: <b>{n_total - n_settled}</b>",
        f"  • VALUE (с кэфом): <b>{n_value}</b>",
        f"  • с участием AI: <b>{n_ai}</b>",
        "",
        "<b>Почему не settled:</b>",
        f"  • матча нет в БД: <b>{no_match}</b>",
        f"  • матч не FINISHED: <b>{not_finished}</b>",
        f"  • FINISHED, но счёт NULL: <b>{no_goals}</b>",
        f"  • готов к settle прямо сейчас: <b>{ready}</b>",
    ]
    if sample_lines:
        text += ["", "<b>Примеры (5 первых):</b>", *sample_lines]
    if ready:
        text += ["", f"⚙ Принудительный settle отработал: <b>{forced_settled}</b> закрыто."]
    await msg.answer("\n".join(text), parse_mode="HTML")


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
        pend = await session.get(PendingUser, chat_id)
        if pend:
            await session.delete(pend)
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


def _humanize_delta(d: timedelta) -> str:
    s = int(d.total_seconds())
    if s < 60:
        return "только что"
    if s < 3600:
        return f"{s // 60} мин назад"
    if s < 86400:
        return f"{s // 3600} ч назад"
    return f"{s // 86400} дн назад"


def _format_user_label(p: PendingUser) -> str:
    if p.username:
        return f"<b>@{p.username}</b>"
    full = " ".join(filter(None, [p.first_name, p.last_name])).strip()
    if full:
        return f"<b>{full}</b>"
    return "<b>(без имени)</b>"


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


@router.message(F.text == "📥 Лиды")
async def btn_leads(msg: Message) -> None:
    if not _is_admin(msg):
        return
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(PendingUser).order_by(PendingUser.last_seen_at.desc()).limit(30)
        )).scalars().all()
    if not rows:
        await msg.answer("📭 Лидов нет — никто не нажимал /start без одобрения.")
        return
    lines = [f"<b>📥 Лиды ({len(rows)}):</b>", ""]
    for p in rows:
        label = _format_user_label(p)
        ago = _humanize_delta(datetime.utcnow() - p.last_seen_at)
        lines.append(
            f"{label}\n"
            f"  <code>{p.chat_id}</code>  ·  стартов: {p.start_count}  ·  {ago}"
        )
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=f"✅ Одобрить {p.chat_id}",
            callback_data=f"approve:{p.chat_id}",
        )] for p in rows[:10]
    ])
    await msg.answer("\n".join(lines), parse_mode="HTML", reply_markup=kb)


@router.callback_query(F.data.startswith("approve:"))
async def cb_approve(q: CallbackQuery) -> None:
    if not q.from_user or q.from_user.id not in settings.admin_ids:
        await q.answer("Только админ", show_alert=True)
        return
    try:
        chat_id = int(q.data.split(":", 1)[1])
    except (ValueError, IndexError):
        await q.answer("Битый callback", show_alert=True)
        return
    async with SessionLocal() as session:
        pend = await session.get(PendingUser, chat_id)
        username = pend.username if pend else None
        sub = await session.get(Subscriber, chat_id)
        if sub is None:
            session.add(Subscriber(chat_id=chat_id, username=username, active=True))
        else:
            sub.active = True
            if username:
                sub.username = username
        if pend:
            await session.delete(pend)
        await session.commit()
    try:
        await q.bot.send_message(chat_id, "✅ Доступ открыт! Напиши /start.")
    except TelegramForbiddenError:
        pass
    except Exception as e:
        logger.warning(f"cb_approve notify failed for {chat_id}: {e}")
    await q.answer(f"✅ {chat_id} одобрен")
    try:
        await q.message.edit_text(
            (q.message.html_text or q.message.text or "") + f"\n\n<i>✅ {chat_id} одобрен.</i>",
            parse_mode="HTML",
        )
    except Exception:
        pass


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
        pend = await session.get(PendingUser, chat_id)
        if pend:
            await session.delete(pend)
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
    elif action == "history":
        await _send_history(q.message)
    elif action == "subscribe":
        await _subscribe(q.message.chat.id, q.from_user.username)
        ai_on = await get_bool("ai_ensemble_enabled", False)
        await q.message.answer("🔔 Уведомления включены.", reply_markup=main_menu(True, ai_on))
    elif action == "unsubscribe":
        await _unsubscribe(q.message.chat.id)
        ai_on = await get_bool("ai_ensemble_enabled", False)
        await q.message.answer(
            "🔕 Уведомления отключены. Доступ к боту сохранён — заходи в любое время.",
            reply_markup=main_menu(False, ai_on),
        )
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
    """True if user has notifications enabled (drives the menu button state)."""
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, chat_id)
    return bool(sub and sub.notifications_enabled)


async def _subscribe(chat_id: int, username: Optional[str]) -> None:
    """Re-enable notifications. Does NOT grant access — only admin can do that."""
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, chat_id)
        if sub is None:
            session.add(Subscriber(
                chat_id=chat_id, username=username,
                active=True, notifications_enabled=True,
            ))
        else:
            sub.notifications_enabled = True
        await session.commit()


async def _unsubscribe(chat_id: int) -> None:
    """Pause notifications. Access stays intact."""
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, chat_id)
        if sub:
            sub.notifications_enabled = False
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
        else:
            from src.pipeline import DISABLED_MARKETS
            stmt = stmt.where(Signal.market.notin_(DISABLED_MARKETS))
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
    from src.bot.formatters import msk_now, MSK_OFFSET
    today_msk = msk_now().date()
    start_msk = datetime.combine(today_msk, datetime.min.time())
    start = start_msk - MSK_OFFSET  # back to UTC for DB comparison
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
            t = fmt_msk(m.utc_date, "%H:%M")
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


async def _send_history(msg: Message) -> None:
    from src.bot.formatters import format_history
    async with SessionLocal() as session:
        pairs = (await session.execute(
            select(Signal, Match)
            .join(Match, Match.id == Signal.match_id)
            .where(Signal.settled.is_(True))
            .order_by(Match.utc_date.desc(), Signal.id.desc())
            .limit(50)
        )).all()
        rows = []
        for sig, match in pairs:
            home = await session.get(Team, match.home_team_id)
            away = await session.get(Team, match.away_team_id)
            if home and away:
                rows.append((sig, match, home, away))
    await msg.answer(format_history(rows), parse_mode="HTML")


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


async def broadcast_signal(bot: Bot, text: str, respect_notifications: bool = True) -> int:
    """Send `text` to every allowed subscriber.

    respect_notifications=True (default) — only those with notifications_enabled=True.
    respect_notifications=False — everyone with access, used for stats digests that
    should reach even users who muted live-signal notifications.
    """
    sent = 0
    async with SessionLocal() as session:
        filters = [Subscriber.active.is_(True)]
        if respect_notifications:
            filters.append(Subscriber.notifications_enabled.is_(True))
        subs = (await session.execute(
            select(Subscriber).where(*filters)
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
    from src.bot.formatters import msk_now, MSK_OFFSET
    now = datetime.utcnow()
    today_msk = msk_now().date()
    end_msk = datetime.combine(today_msk, datetime.min.time()) + timedelta(days=1)
    end_of_day = end_msk - MSK_OFFSET  # back to UTC for DB
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
        lines = [f"☀️ <b>Утренний дайджест — {today_msk.strftime('%d.%m.%Y')}</b>", ""]
        for sig, m in rows:
            h = await session.get(Team, m.home_team_id)
            a = await session.get(Team, m.away_team_id)
            badge = "🎯" if sig.book_odds > 1.0 else "🤖"
            kickoff = fmt_msk(m.utc_date, "%H:%M")
            lines.append(
                f"{badge} {kickoff} <i>{m.competition}</i> — {h.name} vs {a.name}\n"
                f"   <b>{sig.market}</b> · <b>{sig.pick}</b> · {sig.model_prob*100:.0f}%"
                + (f" · edge {sig.edge*100:.0f}%" if sig.book_odds > 1.0 else "")
            )
        text = "\n".join(lines)
    sent = 0
    async with SessionLocal() as session:
        subs = (await session.execute(
            select(Subscriber).where(
                Subscriber.active.is_(True),
                Subscriber.notifications_enabled.is_(True),
            )
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
