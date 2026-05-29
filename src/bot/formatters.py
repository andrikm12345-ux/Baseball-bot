from __future__ import annotations

from datetime import datetime, timedelta

from src.data.database import Match, Signal, Team
from src.signals.tracker import RoiStats


MSK_OFFSET = timedelta(hours=3)


def to_msk(dt: datetime) -> datetime:
    """Convert a naive UTC datetime (as stored in DB) to naive Moscow time."""
    return dt + MSK_OFFSET


def fmt_msk(dt: datetime, pattern: str) -> str:
    return to_msk(dt).strftime(pattern)


def msk_now() -> datetime:
    return datetime.utcnow() + MSK_OFFSET


_MARKET_LABEL = {"1X2": "Исход", "TOTAL": "Тотал", "HANDICAP": "Фора"}
_PICK_LABEL = {
    "HOME": "П1", "DRAW": "X", "AWAY": "П2",
    "OVER": "Больше", "UNDER": "Меньше",
}


def _market_pick_text(sig: Signal) -> str:
    """Human-readable 'market + pick + line', e.g. 'Тотал Больше 2.5'."""
    market = _MARKET_LABEL.get(sig.market, sig.market)
    pick = _PICK_LABEL.get(sig.pick, sig.pick)
    if sig.market == "TOTAL" and sig.line is not None:
        return f"{market} {pick} {sig.line:g}"
    if sig.market == "HANDICAP" and sig.line is not None:
        return f"{market} {pick} ({sig.line:+g})"
    return f"{market} {pick}"


def format_signal(
    sig: Signal, match: Match, home: Team, away: Team, ai_comment: str | None = None
) -> str:
    kickoff = fmt_msk(match.utc_date, "%d.%m %H:%M МСК")
    lines = [
        f"<b>🧠 CLAUDE</b>  <i>{match.competition}</i>",
        f"⚽ <b>{home.name}</b> — <b>{away.name}</b>",
        f"🕒 {kickoff}",
        f"✅ Ставка: <b>{_market_pick_text(sig)}</b>",
        f"🔢 Уверенность Claude: <b>{sig.model_prob*100:.1f}%</b>",
    ]
    if sig.book_odds and sig.book_odds > 1.0:
        lines += [
            f"💰 Кф букмекера: <b>{sig.book_odds:.2f}</b>",
            f"📈 Расхождение с рынком: <b>{sig.edge*100:+.1f}%</b>",
            f"💵 Стейк: <b>{sig.stake_units:.2f}</b> ед.",
        ]
    if ai_comment:
        lines += ["", f"🧠 <i>{ai_comment}</i>"]
    return "\n".join(lines)


def format_daily_digest(
    *,
    yesterday_total: RoiStats,
    yesterday_by_market: dict,
    total: RoiStats,
    date_label: str,
) -> str:
    """End-of-day broadcast: yesterday's result split by market + running totals."""

    def _block(label: str, s: RoiStats) -> str:
        if s.n_settled == 0:
            return f"{label} — нет ставок"
        sign = "📈" if s.profit >= 0 else "📉"
        return (
            f"{label}: <b>{s.n_settled}</b> ставок · "
            f"зашло <b>{s.n_won}</b> ({s.hit_rate:.0f}%) · "
            f"{sign} ROI <b>{s.roi:+.2f}%</b> · "
            f"<b>{s.profit:+.2f} ед.</b>"
        )

    lines = [
        f"📊 <b>СВОДКА ЗА {date_label}</b>",
        "━━━━━━━━━━━━━━━━━━━━━",
    ]
    if yesterday_total.n_settled == 0:
        lines.append("Вчера закрытых ставок не было.")
    else:
        for m in ("1X2", "TOTAL", "HANDICAP"):
            s = yesterday_by_market.get(m)
            if s and s.n_settled:
                lines.append(_block(_MARKET_TITLE.get(m, m), s))
        lines += ["─", _block("<b>ИТОГО</b>", yesterday_total)]
    lines += [
        "",
        "<b>За всё время:</b>",
        f"Ставок: <b>{total.n_settled}</b> · "
        f"Зашло: <b>{total.n_won}</b> ({total.hit_rate:.0f}%)",
        f"ROI: <b>{total.roi:+.2f}%</b> · "
        f"Прибыль: <b>{total.profit:+.2f} ед.</b>",
        "",
        "<i>Подробности — кнопка «📜 История ставок» в меню.</i>",
    ]
    return "\n".join(lines)


def format_history(rows: list[tuple[Signal, Match, Team, Team]], limit: int = 20) -> str:
    """Per-signal settled history: outcome, market, pick, odds, score, profit."""
    if not rows:
        return "📜 <b>ИСТОРИЯ СТАВОК</b>\n\nПока нет закрытых ставок. Заходи позже."

    n = len(rows)
    won = sum(1 for s, *_ in rows if s.won)
    profit = sum(s.profit_units or 0.0 for s, *_ in rows)
    header = [
        "📜 <b>ИСТОРИЯ СТАВОК</b>",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"Закрыто: <b>{n}</b> · Зашло: <b>{won}</b> ({won / n * 100:.0f}%) · "
        f"Прибыль: <b>{profit:+.2f} ед.</b>",
        "",
    ]
    body: list[str] = []
    for sig, match, home, away in rows[:limit]:
        ok = "✅" if sig.won else "❌"
        date = fmt_msk(match.utc_date, "%d.%m")
        market = _MARKET_LABEL.get(sig.market, sig.market)
        pick = _PICK_LABEL.get(sig.pick, sig.pick)
        score = (
            f"{match.home_goals}:{match.away_goals}"
            if match.home_goals is not None and match.away_goals is not None
            else "—"
        )
        odds_str = f"кф {sig.book_odds:.2f}" if sig.book_odds and sig.book_odds > 1.0 else "MODEL"
        ai_mark = " 🧠" if getattr(sig, "is_ai_ensemble", False) else ""
        pnl = f"{sig.profit_units:+.2f}" if sig.profit_units is not None else "?"
        body.append(
            f"{ok} <b>{date}</b> {home.name} — {away.name}{ai_mark}\n"
            f"   {market}: <b>{pick}</b> · {odds_str} · счёт {score} · <b>{pnl} ед.</b>"
        )
    if n > limit:
        body.append(f"\n<i>… и ещё {n - limit} ставок раньше.</i>")
    return "\n".join(header + body)


def format_signal_short(sigs: list[Signal]) -> str:
    if not sigs:
        return "⚪ нет сигнала"
    value_sigs = [s for s in sigs if s.book_odds and s.book_odds > 1.0]
    if value_sigs:
        best = max(value_sigs, key=lambda s: s.edge)
        market = _MARKET_LABEL.get(best.market, best.market)
        pick = _PICK_LABEL.get(best.pick, best.pick)
        ai_tag = " · 🧠" if getattr(best, "is_ai_ensemble", False) else ""
        return f"🎯 {market} {pick} · edge {best.edge*100:.0f}% · кф {best.book_odds:.2f}{ai_tag}"
    best = max(sigs, key=lambda s: s.confidence)
    market = _MARKET_LABEL.get(best.market, best.market)
    pick = _PICK_LABEL.get(best.pick, best.pick)
    ai_tag = " · 🧠" if getattr(best, "is_ai_ensemble", False) else ""
    return f"🤖 {market} {pick} · уверенность {best.confidence*100:.0f}%{ai_tag}"


_MARKET_TITLE = {"1X2": "Исход 1X2", "TOTAL": "Тотал", "HANDICAP": "Фора"}


def format_stats_table(total_s: RoiStats, by_market: dict) -> str:
    def _row(label: str, s: RoiStats) -> str:
        if s.n_settled == 0:
            return f"{label}  —  пока нет ставок"
        return (
            f"{label}\n"
            f"  Ставок: <b>{s.n_settled}</b> · Зашло: <b>{s.n_won}</b> "
            f"({s.hit_rate:.1f}%)\n"
            f"  ROI: <b>{s.roi:+.2f}%</b> · Прибыль: <b>{s.profit:+.2f}</b> ед."
        )

    market_blocks = "\n\n".join(
        _row(_MARKET_TITLE.get(m, m), by_market[m]) for m in ("1X2", "TOTAL", "HANDICAP")
    )
    return (
        "<b>📈 СТАТИСТИКА</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"{_row('ИТОГО', total_s)}\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "<b>По рынкам:</b>\n\n"
        f"{market_blocks}"
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
    "Привет. Бот разбирает футбольные матчи и публикует ставки, у которых "
    "математически есть преимущество над букмекером.\n\n"
    "<b>Как это устроено:</b>\n\n"
    "• <b>Модель</b> — XGBoost на 19 признаках (Elo, форма, очные, отдых "
    "между матчами). Калибрована, переобучается раз в сутки. Даёт "
    "вероятность исхода 1X2, тотала 2.5 и обе забьют.\n\n"
    "• <b>Валуй (value)</b> — сравнение нашей вероятности с кэфом "
    "букмекера. Сигнал публикуется только если перевес ≥ 5%. Размер "
    "ставки — четверть Kelly, потолок 2 единицы.\n\n"
    "• <b>AI-ансамбль</b> — Claude независимо оценивает тот же матч с "
    "учётом свежих новостей: травмы, составы, мотивация. Его прогноз "
    "смешивается с моделью, помечается значком 🧠.\n\n"
    "/signals — текущие ставки\n"
    "/today — матчи дня\n"
    "/stats — ROI · hit-rate · drawdown\n"
    "/chart — кривая доходности\n"
    "/menu — настройки\n\n"
    "<i>Бот считает за тебя. Ставить или нет — решаешь сам.</i>"
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
