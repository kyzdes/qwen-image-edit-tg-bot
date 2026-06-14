"""Конфигурация бота.

Все значения читаются ИЗ ОКРУЖЕНИЯ только внутри load_config().
На уровне модуля никаких обращений к env — модуль импортируется без побочных эффектов.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Дефолтный id Space (публичный). Можно переопределить через env SPACE_ID.
DEFAULT_SPACE_ID = "productowner/Qwen-Image-Edit-2509-LoRAs-Fast-v2"
# Дефолтный таймаут запроса к Space, секунды.
DEFAULT_REQUEST_TIMEOUT = 300


@dataclass
class Config:
    """Контейнер с настройками бота."""

    bot_token: str
    allowed_user_ids: set[int] = field(default_factory=set)
    space_id: str = DEFAULT_SPACE_ID
    hf_token: str | None = None
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT


def _parse_user_ids(raw: str | None) -> set[int]:
    """Разобрать ALLOWED_USER_IDS: целые числа через запятую и/или пробелы.

    Пустая/отсутствующая строка => пустой set. Нечисловые токены игнорируются.
    """
    if not raw:
        return set()
    # Заменяем запятые на пробелы и режем по любому пробелу.
    ids: set[int] = set()
    for token in raw.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        try:
            ids.add(int(token))
        except ValueError:
            # Молча пропускаем мусор, чтобы кривой env не ронял бот.
            continue
    return ids


def _parse_timeout(raw: str | None) -> int:
    """Разобрать REQUEST_TIMEOUT; при отсутствии/мусоре — дефолт."""
    if not raw:
        return DEFAULT_REQUEST_TIMEOUT
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT


def load_config() -> Config:
    """Прочитать конфигурацию из переменных окружения.

    BOT_TOKEN          — обязателен, иначе RuntimeError.
    ALLOWED_USER_IDS   — int'ы через запятую/пробел; пусто => пустой set (пускать всех).
    SPACE_ID           — по умолчанию productowner/Qwen-Image-Edit-2509-LoRAs-Fast-v2.
    HF_TOKEN           — опционально (None если не задан/пусто).
    REQUEST_TIMEOUT    — секунды, по умолчанию 300.
    """
    bot_token = os.environ.get("BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError(
            "BOT_TOKEN не задан. Укажите токен бота в переменной окружения BOT_TOKEN."
        )

    allowed_user_ids = _parse_user_ids(os.environ.get("ALLOWED_USER_IDS"))

    space_id = os.environ.get("SPACE_ID", "").strip() or DEFAULT_SPACE_ID

    hf_token_raw = os.environ.get("HF_TOKEN", "").strip()
    hf_token = hf_token_raw or None

    request_timeout = _parse_timeout(os.environ.get("REQUEST_TIMEOUT"))

    return Config(
        bot_token=bot_token,
        allowed_user_ids=allowed_user_ids,
        space_id=space_id,
        hf_token=hf_token,
        request_timeout=request_timeout,
    )
