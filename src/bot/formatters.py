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
    "⚡ <b>FOOTBALL ANALYTICS — ЗАКРЫТЫЙ КОНТУР</b>\n\n"
    "Здесь не пишут «прогнозы». Здесь публикуются числа.\n\n"
    "Каждый матч топ-лиг проходит через три фильтра:\n\n"
    "▸ <b>XGBoost</b> — модель на 19 фичах, обученная на тысячах матчей: "
    "Elo-рейтинги, форма, очные встречи, дни отдыха.\n"
    "▸ <b>Edge-фильтр</b> — сравнение модельной вероятности с линией букмекера. "
    "В работу идут только расхождения ≥ 5%.\n"
    "▸ <b>Claude AI</b> — комментирует каждую ставку, подсвечивает контекст, "
    "проверяет логику сигнала.\n\n"
    "ROI считается по факту сыгранных матчей.\n\n"
    "<b>Команды:</b>\n"
    "/signals — активные value-сигналы\n"
    "/today — матчи дня\n"
    "/stats — ROI и hit-rate\n"
    "/chart — кривая прибыли\n"
    "/menu — кнопки\n"
    "/subscribe — авто-доставка сигналов\n"
    "/unsubscribe — отписка\n"
    "/help — как это работает\n\n"
    "❗ Это <b>инструмент анализа</b>. Решение ставить — твоё."
)


HELP = (
    "<b>КАК ЭТО РАБОТАЕТ</b>\n\n"
    "<b>1. Сбор данных.</b> Каждую ночь подтягиваются результаты топ-лиг.\n"
    "<b>2. Перерасчёт.</b> Обновляются Elo, форма, статистика очных встреч.\n"
    "<b>3. Прогноз.</b> XGBoost генерит вероятности для 1X2, Тотала 2.5 и BTTS.\n"
    "<b>4. Edge-сравнение.</b> Сверяется с котировками букмекеров. "
    "Если расхождение ≥ 5% — это сигнал.\n"
    "<b>5. AI-проверка.</b> Claude формулирует контекст: почему модель видит "
    "ценность в этой ставке.\n"
    "<b>6. Результат.</b> После матча всё автоматически рассчитывается и попадает в ROI.\n\n"
    "<b>Метки:</b>\n"
    "🎯 VALUE — есть кф букмекера, посчитан edge и стейк по Kelly\n"
    "🤖 MODEL — только модельная вероятность (если линии не подгрузились)\n\n"
    "<b>Стейк:</b> 1/4 Kelly, потолок 2 единицы. Никаких «всё или ничего»."
)
