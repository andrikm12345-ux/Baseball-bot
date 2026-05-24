from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

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


_MARKET_LABEL = {
    "1X2": "Исход",
    "OU25": "Тотал 2.5",
    "BTTS": "Обе забьют",
    "HOME_OVER05": "ИТБ1 0.5",
    "HOME_OVER15": "ИТБ1 1.5",
    "HOME_OVER25": "ИТБ1 2.5",
    "AWAY_OVER05": "ИТБ2 0.5",
    "AWAY_OVER15": "ИТБ2 1.5",
    "AWAY_OVER25": "ИТБ2 2.5",
}
_PICK_LABEL = {
    "HOME": "П1", "DRAW": "X", "AWAY": "П2",
    "OVER": "Б", "UNDER": "М",
    "YES": "Да", "NO": "Нет",
}


def format_signal(
    sig: Signal, match: Match, home: Team, away: Team, ai_comment: str | None = None
) -> str:
    kickoff = fmt_msk(match.utc_date, "%d.%m %H:%M МСК")
    badge = "🎯 VALUE" if (sig.book_odds and sig.book_odds > 1.0) else "🤖 MODEL"
    if getattr(sig, "is_ai_ensemble", False):
        badge += " · 🧠 AI"
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


def format_daily_digest(
    yesterday: RoiStats, total: RoiStats, date_label: str
) -> str:
    """End-of-day broadcast: yesterday's result + running totals."""
    lines = [
        f"📊 <b>СВОДКА ЗА {date_label}</b>",
        "━━━━━━━━━━━━━━━━━━━━━",
    ]
    if yesterday.n_settled == 0:
        lines.append("Вчера закрытых ставок не было.")
    else:
        sign = "📈" if yesterday.profit >= 0 else "📉"
        lines += [
            f"Ставок: <b>{yesterday.n_settled}</b> · "
            f"Зашло: <b>{yesterday.n_won}</b> ({yesterday.hit_rate:.0f}%)",
            f"{sign} ROI: <b>{yesterday.roi:+.2f}%</b> · "
            f"Прибыль: <b>{yesterday.profit:+.2f} ед.</b>",
        ]
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


def format_training_report(m: dict) -> str:
    def _arrow(d: float) -> str:
        if abs(d) < 0.001:
            return "≈"
        return "↓" if d < 0 else "↑"

    diff = m.get("diff_vs_prev", {})
    walk = m.get("walk_forward", {})
    lines = [
        "🎓 <b>МОДЕЛЬ ПЕРЕОБУЧЕНА</b>",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"Матчей в выборке: <b>{m['n_train']}</b>",
        "",
        "<b>Качество (in-sample):</b>",
        f"• 1X2 logloss: <b>{m['1x2_logloss']:.4f}</b> "
        f"{_arrow(diff.get('1x2_logloss', 0))} {abs(diff.get('1x2_logloss', 0)):.4f}",
        f"• OU2.5 Brier: <b>{m['ou_brier']:.4f}</b> "
        f"{_arrow(diff.get('ou_brier', 0))} {abs(diff.get('ou_brier', 0)):.4f}",
        f"• BTTS Brier: <b>{m['btts_brier']:.4f}</b> "
        f"{_arrow(diff.get('btts_brier', 0))} {abs(diff.get('btts_brier', 0)):.4f}",
    ]
    if walk:
        lines += [
            "",
            "<b>Честная проверка (walk-forward CV):</b>",
            f"• 1X2: <b>{walk.get('1x2_logloss', 0):.4f}</b>",
            f"• OU2.5: <b>{walk.get('ou_brier', 0):.4f}</b>",
            f"• BTTS: <b>{walk.get('btts_brier', 0):.4f}</b>",
        ]
    if m.get("top_features"):
        lines += ["", "<b>Главные фичи (важность):</b>"]
        for f in m["top_features"]:
            lines.append(f"  • {f}")
    lines.append("\n<i>Низкие logloss/Brier = лучше. Стрелки — изменение от прошлого обучения.</i>")
    return "\n".join(lines)


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
