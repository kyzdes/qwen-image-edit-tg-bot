"""features.py — каталог фич бота Qwen Image Edit.

Определяет неизменяемый dataclass Feature и ровно 11 фич в строго заданном
порядке (см. SHARED CONTRACT). Строки `lora` и `default_prompt` должны точно
совпадать со строками, которые понимает Space API, поэтому менять их нельзя.

Модуль не имеет побочных эффектов: только определения данных.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    """Одна фича-редактор.

    Поля:
      key            — внутренний идентификатор (используется в callback-data "f:<key>").
      lora           — ТОЧНАЯ строка LoRA-адаптера для Space API.
      label          — текст кнопки в меню (ru, с эмодзи).
      default_prompt — промпт по умолчанию (ТОЧНАЯ английская строка для API).
      description    — короткое описание (ru).
      directions     — опциональное подменю: список пар (текст_кнопки, промпт);
                       None, если у фичи нет направлений.
      is_upscale     — True только для апскейл-фич: их результат бот отправляет
                       несжатым документом, а не фото.
    """

    key: str
    lora: str
    label: str
    default_prompt: str
    description: str
    directions: list[tuple[str, str]] | None
    is_upscale: bool = False


# Ровно 11 фич в порядке из контракта. Не переставлять и не править строки.
FEATURES: list[Feature] = [
    Feature(
        key="anime",
        lora="Photo-to-Anime",
        label="🎨 Аниме",
        default_prompt="Transform into anime.",
        description="Фото → аниме-стиль",
        directions=None,
    ),
    Feature(
        key="angles",
        lora="Multiple-Angles",
        label="🔄 Сменить ракурс",
        default_prompt="Rotate the camera 45 degrees to the left.",
        description="Поворот/смена ракурса камеры",
        directions=[
            ("⬅️ Поворот влево 45°", "Rotate the camera 45 degrees to the left."),
            ("➡️ Поворот вправо 45°", "Rotate the camera 45 degrees to the right."),
            ("⬆️ Вид сверху", "Switch the camera to a top-down right corner view."),
            ("⬇️ Вид снизу", "Switch the camera to a bottom-up view."),
        ],
    ),
    Feature(
        key="light_restore",
        lora="Light-Restoration",
        label="💡 Убрать тени",
        default_prompt="Remove shadows and relight the image using soft lighting.",
        description="Убрать тени, мягкий свет",
        directions=None,
    ),
    Feature(
        key="relight",
        lora="Relight",
        label="🌅 Пересвет",
        default_prompt="Use a subtle golden-hour filter with smooth light diffusion.",
        description="Изменить освещение (golden hour и т.п.)",
        directions=None,
    ),
    Feature(
        key="multi_light",
        lora="Multi-Angle-Lighting",
        label="🔦 Направленный свет",
        default_prompt="Light source from the Right Rear.",
        description="Источник света по направлению",
        directions=[
            ("↗️ Справа сзади", "Light source from the Right Rear."),
            ("↖️ Слева сзади", "Light source from the Left Rear."),
            ("➡️ Справа", "Light source from the Right."),
            ("⬅️ Слева", "Light source from the Left."),
            ("⬆️ Сверху", "Light source from the Top."),
            ("⬇️ Снизу", "Light source from the Below."),
        ],
    ),
    Feature(
        key="skin",
        lora="Edit-Skin",
        label="🧑 Детали кожи",
        default_prompt="Make the subject's skin details more prominent and natural.",
        description="Улучшить детали кожи",
        directions=None,
    ),
    Feature(
        key="next_scene",
        lora="Next-Scene",
        label="🎬 Следующая сцена",
        default_prompt=(
            "The camera moves slightly forward as sunlight breaks through the "
            "clouds, casting a soft glow around the character's silhouette in the "
            "mist."
        ),
        description="Кинематографичная следующая сцена",
        directions=None,
    ),
    Feature(
        key="flat_log",
        lora="Flat-Log",
        label="🎞 Flat/Log цвет",
        default_prompt=(
            "flatcolor Desaturate the image and lower the contrast to create a "
            "flat, ungraded look similar to a camera log profile."
        ),
        description="Плоский log-профиль под цветокор",
        directions=None,
    ),
    Feature(
        key="upscale",
        lora="Upscale-Image",
        label="⬆️ Апскейл",
        default_prompt="Upscale the image.",
        description="Увеличить/улучшить",
        directions=None,
        is_upscale=True,
    ),
    Feature(
        key="upscale2k",
        lora="Upscale2K",
        label="🖼 Апскейл 2K",
        default_prompt="Upscale this picture to 4K resolution.",
        description="Апскейл до 2K/4K",
        directions=None,
        is_upscale=True,
    ),
    Feature(
        key="dotted",
        lora="Dotted-Illustration",
        label="⚫ Точечная иллюстрация",
        default_prompt="dotted illustration.",
        description="Стиль точечной иллюстрации",
        directions=None,
    ),
]


# Быстрый доступ по ключу: {key: Feature}. Порядок соответствует FEATURES.
FEATURES_BY_KEY: dict[str, Feature] = {feature.key: feature for feature in FEATURES}
