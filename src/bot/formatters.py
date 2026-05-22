from __future__ import annotations

from datetime import datetime
from typing import Iterable

from src.data.database import Match, Signal, Team
from src.signals.tracker import RoiStats


_MARKET_LABEL = {"1X2": "Исход", "OU25": "Тотал 2.5", "BTTS": "Обе забьют"}
_PICK_LABEL = {
    "HOME": "П1", "DRAW": "X", "AWAY": "П2",
    "OVER": "Б", "UNDER": "М",
    "YES": "Да", "NO": "Нет",
}


def format_signal(
    sig: Signal, match: Match, home: Team, away: Team, ai_comment: str | None = None
) -> str:
    kickoff = match.utc_date.strftime("%d.%m %H:%M UTC")
    badge = "🎯 VALUE" if (sig.book_odds and sig.book_odds > 1.0) else "🤖 MODEL"
    lines = [
        f"<b>{badge}</b>  <i>{match.competition}</i>",
        f"⚽ <b>{home.name}</b> — <b>{away.name}</b>",
        f"🕒 {kickoff}",
        f"📊 Рынок: <b>{_MARKET_LABEL.get(sig.market, sig.market)}</b>",
        f"✅ Ставка: <b>{_PICK_LABEL.get(sig.pick, sig.pick)}</b>",
        f"🔢 Вероятность модели: <b>{sig.model_prob*100:.1f}%</b>",
        f"⚖️ Справедливый кф: <b>{sig.fair_odds:.2f}</b>",
    ]
    if sig.book_odds and sig.book_odds > 1.0:
        lines += [
            f"💰 Кф букмекера: <b>{sig.book_odds:.2f}</b>",
            f"📈 Edge: <b>{sig.edge*100:.1f}%</b>",
            f"💵 Стейк: <b>{sig.stake_units:.2f}</b> ед.",
        ]
    if ai_comment:
        lines += ["", f"🧠 <i>{ai_comment}</i>"]
    return "\n".join(lines)


def format_signal_short(sigs: list[Signal]) -> str:
    if not sigs:
        return "⚪ нет сигнала"
    value_sigs = [s for s in sigs if s.book_odds and s.book_odds > 1.0]
    if value_sigs:
        best = max(value_sigs, key=lambda s: s.edge)
        market = _MARKET_LABEL.get(best.market, best.market)
        pick = _PICK_LABEL.get(best.pick, best.pick)
        return f"🎯 {market} {pick} · edge {best.edge*100:.0f}% · кф {best.book_odds:.2f}"
    best = max(sigs, key=lambda s: s.confidence)
    market = _MARKET_LABEL.get(best.market, best.market)
    pick = _PICK_LABEL.get(best.pick, best.pick)
    return f"🤖 {market} {pick} · уверенность {best.confidence*100:.0f}%"


def format_stats_table(
    model_s: RoiStats,
    value_s: RoiStats,
    ai_s: RoiStats,
    total_s: RoiStats,
) -> str:
    def _row(label: str, s: RoiStats) -> str:
        if s.n_settled == 0:
            return f"{label}  —  пока нет ставок"
        return (
            f"{label}\n"
            f"  Ставок: <b>{s.n_settled}</b> · Зашло: <b>{s.n_won}</b> "
            f"({s.hit_rate:.1f}%)\n"
            f"  ROI: <b>{s.roi:+.2f}%</b> · Прибыль: <b>{s.profit:+.2f}</b> ед."
        )

    return (
        "<b>📈 СТАТИСТИКА</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"{_row('🤖 MODEL (без линий)', model_s)}\n\n"
        f"{_row('🎯 VALUE (edge ≥ 5%)', value_s)}\n\n"
        f"{_row('🧠 AI-ансамбль', ai_s)}\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"{_row('ИТОГО', total_s)}"
    )


def format_roi(stats: RoiStats, title: str = "ROI") -> str:
    if stats.n_settled == 0:
        return f"<b>{title}</b>\nЕщё нет рассчитанных ставок."
    return (
        f"<b>📈 {title}</b>\n"
        f"Ставок: <b>{stats.n_settled}</b>\n"
        f"Зашло: <b>{stats.n_won}</b> ({stats.hit_rate:.1f}%)\n"
        f"Поставлено: <b>{stats.staked:.2f}</b> ед.\n"
        f"Прибыль: <b>{stats.profit:+.2f}</b> ед.\n"
        f"ROI: <b>{stats.roi:+.2f}%</b>"
    )


WELCOME = (
    "◆ <b>FOOTBALL INTELLIGENCE — PRIVATE ACCESS</b>\n\n"
    "<i>Закрытый аналитический контур. Доступ по приглашению.</i>\n\n"
    "━━━━━━━━━━━━━━━━━━━━━\n\n"
    "<b>АРХИТЕКТУРА СИГНАЛА</b>\n\n"
    "<b>I. XGBoost-модель</b>\n"
    "19-мерный признаковый вектор: Elo, форма, очные встречи, временные интервалы. "
    "Изотоническая калибровка вероятностей. Переобучение каждые 24 часа.\n\n"
    "<b>II. Edge-фильтр</b>\n"
    "Сопоставление модельной вероятности с консенсусом рынка. "
    "В работу принимаются только расхождения ≥ 5%.\n\n"
    "<b>III. Claude AI</b>\n"
    "Контекстная верификация каждого сигнала: факторы, не учтённые статистикой. "
    "Подтверждение либо отмена.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━\n\n"
    "<b>ОПЕРИРОВАНИЕ</b>\n\n"
    "/signals — активные позиции\n"
    "/today — кикоффы дня\n"
    "/stats — ROI · hit-rate · drawdown\n"
    "/chart — кривая доходности\n"
    "/menu — управление\n"
    "/subscribe — подключить уведомления\n\n"
    "━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Размер позиции — <b>1/4 Kelly</b>. Потолок — <b>2 единицы</b>.\n"
    "Расчёт результата автоматический, по факту матча.\n\n"
    "<i>Контур предоставляет данные. Решения принимает оператор.</i>"
)


HELP = (
    "◆ <b>ПРОТОКОЛ РАБОТЫ</b>\n\n"
    "<b>I. Данные.</b> Подтягиваются результаты топ-лиг каждые 6 часов.\n"
    "<b>II. Перерасчёт.</b> Обновляются Elo-рейтинги, форма, очные встречи.\n"
    "<b>III. Прогноз.</b> XGBoost генерирует вероятности по трём рынкам: "
    "1X2 · Тотал 2.5 · Обе забьют.\n"
    "<b>IV. Сопоставление.</b> Сверка с консенсусом букмекеров. "
    "Расхождение ≥ 5% → сигнал.\n"
    "<b>V. Верификация.</b> Claude AI оценивает контекст и подтверждает "
    "либо отменяет ставку.\n"
    "<b>VI. Расчёт.</b> После матча результат автоматически попадает в ROI.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━\n\n"
    "<b>КЛАССИФИКАЦИЯ</b>\n\n"
    "<b>VALUE</b> — есть кф букмекера, посчитан edge и размер позиции\n"
    "<b>MODEL</b> — только модельная вероятность (линии недоступны)\n\n"
    "━━━━━━━━━━━━━━━━━━━━━\n\n"
    "<b>УПРАВЛЕНИЕ КАПИТАЛОМ</b>\n\n"
    "1/4 Kelly · потолок 2 единицы · фиксированный банк."
)
