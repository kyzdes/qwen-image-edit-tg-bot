"""Инлайн-клавиатуры Telegram-бота "Qwen Image Edit".

Здесь собраны все экранные клавиатуры. Каждый callback_data строго
соответствует общей схеме (см. SHARED CONTRACT):

    f:<key>            — выбрать фичу
    d:<key>:<idx>      — выбрать направление внутри фичи
    gen                — сгенерировать сейчас
    set                — открыть настройки (и "холостое" пере-рисование)
    s:steps:- / s:steps:+   — шаги -/+
    s:cfg:-   / s:cfg:+     — guidance (CFG) -/+
    s:rand             — переключить рандомизацию seed
    s:seed             — попросить пользователя ввести seed
    again              — повторить (с новым случайным seed)
    newphoto           — сбросить, попросить новое фото
    menu               — показать меню фич
    limits             — показать лимиты ZeroGPU

Никаких побочных эффектов на импорте — только определения функций.
"""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from features import FEATURES, FEATURES_BY_KEY
from state import Settings


def main_menu() -> InlineKeyboardMarkup:
    """Главное меню: 11 кнопок-фич (по 2 в ряд) + ряд "Настройки/Сгенерировать"."""
    rows: list[list[InlineKeyboardButton]] = []

    # Фичи по две в ряд, в порядке объявления FEATURES.
    for i in range(0, len(FEATURES), 2):
        pair = FEATURES[i:i + 2]
        rows.append([
            InlineKeyboardButton(feature.label, callback_data=f"f:{feature.key}")
            for feature in pair
        ])

    # Нижний ряд действий.
    rows.append([
        InlineKeyboardButton("⚙️ Настройки", callback_data="set"),
        InlineKeyboardButton("✨ Сгенерировать", callback_data="gen"),
    ])
    # Просмотр лимитов ZeroGPU.
    rows.append([
        InlineKeyboardButton("📊 GPU-лимиты", callback_data="limits"),
    ])

    return InlineKeyboardMarkup(rows)


def directions_keyboard(feature_key: str) -> InlineKeyboardMarkup:
    """Подменю направлений для фичи: по кнопке на каждое направление + "Назад".

    callback_data направления: "d:<key>:<idx>", где idx — индекс в списке
    feature.directions. Если у фичи нет направлений — отдаём только "Назад".
    """
    feature = FEATURES_BY_KEY[feature_key]
    directions = feature.directions or []

    rows: list[list[InlineKeyboardButton]] = []

    # Направления по две кнопки в ряд (1-2 в ряд).
    for i in range(0, len(directions), 2):
        pair = directions[i:i + 2]
        rows.append([
            InlineKeyboardButton(label, callback_data=f"d:{feature_key}:{i + offset}")
            for offset, (label, _prompt) in enumerate(pair)
        ])

    # Ряд "Назад" в главное меню.
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu")])

    return InlineKeyboardMarkup(rows)


def settings_keyboard(settings: Settings) -> InlineKeyboardMarkup:
    """Клавиатура настроек: шаги, CFG, рандомизация seed, ручной seed, "Назад".

    Кнопки с текущими значениями несут callback "set" — тап по ним просто
    пере-рисовывает экран настроек (callback "noop" в схеме не разрешён).
    """
    # CFG показываем с одним знаком после запятой.
    cfg_display = f"{settings.guidance:.1f}"
    rand_display = "🎲 ON" if settings.randomize_seed else "🎲 OFF"

    rows: list[list[InlineKeyboardButton]] = [
        # Шаги: ➖ значение ➕
        [
            InlineKeyboardButton("➖", callback_data="s:steps:-"),
            InlineKeyboardButton(f"Шаги: {settings.steps}", callback_data="set"),
            InlineKeyboardButton("➕", callback_data="s:steps:+"),
        ],
        # CFG (guidance): ➖ значение ➕
        [
            InlineKeyboardButton("➖", callback_data="s:cfg:-"),
            InlineKeyboardButton(f"CFG: {cfg_display}", callback_data="set"),
            InlineKeyboardButton("➕", callback_data="s:cfg:+"),
        ],
        # Переключатель рандомизации seed.
        [InlineKeyboardButton(f"Случайный seed: {rand_display}", callback_data="s:rand")],
        # Текущий seed + кнопка задать вручную.
        [
            InlineKeyboardButton(f"seed: {settings.seed}", callback_data="set"),
            InlineKeyboardButton("Задать seed ✏️", callback_data="s:seed"),
        ],
        # Назад в главное меню.
        [InlineKeyboardButton("⬅️ Назад", callback_data="menu")],
    ]

    return InlineKeyboardMarkup(rows)


def result_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура под результатом: Повторить/Настройки и Новое фото/Меню фич."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 Повторить", callback_data="again"),
            InlineKeyboardButton("🎚 Настройки", callback_data="set"),
        ],
        [
            InlineKeyboardButton("🆕 Новое фото", callback_data="newphoto"),
            InlineKeyboardButton("📋 Меню фич", callback_data="menu"),
        ],
    ])
