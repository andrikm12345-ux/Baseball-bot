from __future__ import annotations

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)


def main_menu() -> InlineKeyboardMarkup:
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
            InlineKeyboardButton(text="🔔 Подписаться", callback_data="menu:subscribe"),
            InlineKeyboardButton(text="🔕 Отписаться", callback_data="menu:unsubscribe"),
        ],
        [InlineKeyboardButton(text="🔧 Фильтры", callback_data="menu:filters")],
    ])


def admin_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="👥 Подписчики"), KeyboardButton(text="➕ Добавить")],
            [KeyboardButton(text="🚫 Удалить"),    KeyboardButton(text="📊 Статистика")],
            [KeyboardButton(text="🎯 Сигналы"),    KeyboardButton(text="📅 Сегодня")],
        ],
        resize_keyboard=True,
        persistent=True,
    )


def filters_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🏴 PL", callback_data="filter:league:PL"),
            InlineKeyboardButton(text="🇪🇸 PD", callback_data="filter:league:PD"),
            InlineKeyboardButton(text="🇮🇹 SA", callback_data="filter:league:SA"),
        ],
        [
            InlineKeyboardButton(text="🇩🇪 BL1", callback_data="filter:league:BL1"),
            InlineKeyboardButton(text="🇫🇷 FL1", callback_data="filter:league:FL1"),
            InlineKeyboardButton(text="🏆 CL", callback_data="filter:league:CL"),
        ],
        [
            InlineKeyboardButton(text="Исход 1X2", callback_data="filter:market:1X2"),
            InlineKeyboardButton(text="Тотал 2.5", callback_data="filter:market:OU25"),
            InlineKeyboardButton(text="Обе забьют", callback_data="filter:market:BTTS"),
        ],
        [
            InlineKeyboardButton(text="🎯 Только VALUE", callback_data="filter:type:VALUE"),
            InlineKeyboardButton(text="🤖 Все", callback_data="filter:type:ALL"),
        ],
        [InlineKeyboardButton(text="« Назад", callback_data="menu:back")],
    ])
