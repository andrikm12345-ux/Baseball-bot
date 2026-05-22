from __future__ import annotations

from src.data.database import SessionLocal, Setting


async def get_setting(key: str, default: str = "") -> str:
    async with SessionLocal() as s:
        row = await s.get(Setting, key)
        return row.value if row else default


async def set_setting(key: str, value: str) -> None:
    async with SessionLocal() as s:
        row = await s.get(Setting, key)
        if row:
            row.value = value
        else:
            s.add(Setting(key=key, value=value))
        await s.commit()


async def get_bool(key: str, default: bool = False) -> bool:
    v = await get_setting(key, "")
    if not v:
        return default
    return v.lower() == "true"


async def set_bool(key: str, value: bool) -> None:
    await set_setting(key, "true" if value else "false")
