from __future__ import annotations

from datetime import datetime
from typing import Iterable

from src.data.database import Match, Signal, Team
from src.signals.tracker import RoiStats


_MARKET_LABEL = {"1X2": "Исход", "OU25": "Тотал 2.5", "BTTS": "Обе забьют"}
_PICK_LABEL = {
    "HOME": "П1", "DRAW": "X", "AWAY": "П2",
    "OVER": "Б 2.5", "UNDER": "М 2.5",
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
    "👋 <b>Привет!</b>\n\n"
    "Я бот-аналитик футбольных матчей. Гоняю ML-модель (XGBoost) на исторических "
    "матчах топ-лиг, считаю edge против котировок букмекеров и публикую "
    "value-сигналы.\n\n"
    "❗ Это <b>аналитика</b>, а не гарантия. Реальный ROI считается по факту "
    "сыгранных ставок и публикуется честно — без приукрашивания.\n\n"
    "<b>Команды:</b>\n"
    "/signals — сигналы на ближайшие матчи\n"
    "/today — матчи на сегодня\n"
    "/stats — статистика и ROI\n"
    "/chart — график кумулятивной прибыли\n"
    "/menu — открыть меню с кнопками\n"
    "/subscribe — получать сигналы автоматически\n"
    "/unsubscribe — отписаться\n"
    "/help — помощь"
)


HELP = (
    "<b>Как это работает:</b>\n"
    "1. Каждую ночь модель пересчитывает рейтинги (Elo), форму и фичи.\n"
    "2. На ближайшие 7 дней генерируются прогнозы по трём рынкам: исход, тотал 2.5, обе забьют.\n"
    "3. Если у вас подключен фид котировок — публикуются только value-сигналы (edge ≥ 5%).\n"
    "4. После матчей результаты автоматически проставляются и ROI обновляется.\n\n"
    "<b>Метки:</b>\n"
    "🎯 VALUE — модель видит расхождение с букмекером\n"
    "🤖 MODEL — высокая уверенность модели (без котировок)\n\n"
    "Размер стейка считается по 1/4 Kelly и ограничен 2 ед."
)
