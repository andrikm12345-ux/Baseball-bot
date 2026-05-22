from __future__ import annotations

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)


def main_menu(subscribed: bool = False, ai_enabled: bool = False) -> InlineKeyboardMarkup:
    sub_text = "🔔 Подписка: АКТИВНА" if subscribed else "🔕 Подписаться"
    sub_data = "menu:unsubscribe" if subscribed else "menu:subscribe"
    ai_dot = "🟢" if ai_enabled else "🔴"
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🎯 Сигналы", callback_data="menu:signals"),
            InlineKeyboardButton(text="📅 Сегодня", callback_data="menu:today"),
        ],
        [
            InlineKeyboardButton(text="📈 ROI", callback_data="menu:stats"),
            InlineKeyboardButton(text="📊 График", callback_data="menu:chart"),
        ],
        [
            InlineKeyboardButton(text=sub_text, callback_data=sub_data),
            InlineKeyboardButton(text=f"🧠 AI: {ai_dot}", callback_data="menu:ai_info"),
        ],
        [InlineKeyboardButton(text="🔧 Фильтры", callback_data="menu:filters")],
    ])


def admin_menu(ai_enabled: bool = False) -> ReplyKeyboardMarkup:
    ai_label = "🧠 AI: 🟢 ВКЛ" if ai_enabled else "🧠 AI: 🔴 ВЫКЛ"
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="👥 Подписчики"), KeyboardButton(text="➕ Добавить")],
            [KeyboardButton(text="🚫 Удалить"),    KeyboardButton(text="📊 Статистика")],
            [KeyboardButton(text="🎯 Сигналы"),    KeyboardButton(text="📅 Сегодня")],
            [KeyboardButton(text="📥 Лиды"),       KeyboardButton(text=ai_label)],
        ],
        resize_keyboard=True,
        persistent=True,
    )


_LEAGUE_FLAG = {
    "PL": "🏴", "PD": "🇪🇸", "SA": "🇮🇹", "BL1": "🇩🇪", "FL1": "🇫🇷",
    "CL": "🏆", "DED": "🇳🇱", "PPL": "🇵🇹", "ELC": "🏴", "BSA": "🇧🇷",
    "CLI": "🏆", "EC": "🏆", "WC": "🏆",
}


def filters_menu() -> InlineKeyboardMarkup:
    from src.config import settings

    league_btns = [
        InlineKeyboardButton(
            text=f"{_LEAGUE_FLAG.get(c, '🌍')} {c}",
            callback_data=f"filter:league:{c}",
        )
        for c in settings.competitions
    ]
    league_rows = [league_btns[i:i + 3] for i in range(0, len(league_btns), 3)]

    market_rows = [
        [
            InlineKeyboardButton(text="Исход 1X2", callback_data="filter:market:1X2"),
            InlineKeyboardButton(text="Тотал 2.5", callback_data="filter:market:OU25"),
            InlineKeyboardButton(text="Обе забьют", callback_data="filter:market:BTTS"),
        ],
    ]
    type_row = [
        InlineKeyboardButton(text="🎯 Только VALUE", callback_data="filter:type:VALUE"),
        InlineKeyboardButton(text="🤖 Все", callback_data="filter:type:ALL"),
    ]
    back_row = [InlineKeyboardButton(text="« Назад", callback_data="menu:back")]
    return InlineKeyboardMarkup(
        inline_keyboard=[*league_rows, *market_rows, type_row, back_row]
    )
