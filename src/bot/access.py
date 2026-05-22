from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject, User
from loguru import logger

from src.config import settings
from src.data.database import SessionLocal, Subscriber


EXEMPT_COMMANDS = ("/start",)


async def is_allowed(chat_id: int) -> bool:
    if chat_id in settings.admin_ids:
        return True
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, chat_id)
    return bool(sub and sub.active)


def _extract_text(event: TelegramObject) -> Optional[str]:
    if isinstance(event, Message):
        return (event.text or event.caption or "").strip() or None
    if isinstance(event, CallbackQuery):
        return event.data
    return None


class AccessControlMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user: Optional[User] = data.get("event_from_user")
        if user is None:
            return None

        if user.id in settings.admin_ids:
            return await handler(event, data)

        async with SessionLocal() as session:
            sub = await session.get(Subscriber, user.id)
        if sub and sub.active:
            return await handler(event, data)

        if isinstance(event, Message):
            text = (event.text or event.caption or "").strip()
            if text:
                first = text.split(maxsplit=1)[0].split("@", 1)[0]
                if first in EXEMPT_COMMANDS:
                    return await handler(event, data)

        logger.debug(
            f"access denied: user_id={user.id} "
            f"event={type(event).__name__} text={_extract_text(event)!r}"
        )
        return None
