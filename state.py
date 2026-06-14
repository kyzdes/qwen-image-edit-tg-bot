"""Состояние пользовательских сессий бота.

Здесь хранятся:
- Settings    — параметры генерации (steps / guidance / seed / randomize) с клампингом;
- константы лимитов (STEPS_*, GUIDANCE_*);
- Session     — текущая сессия пользователя (фото, выбранная фича, промпт, настройки);
- SessionStore — простое in-memory хранилище сессий по user_id;
- store        — модульный синглтон SessionStore.

Модуль без побочных эффектов: ничего не читает из окружения и не создаёт
внешних подключений при импорте.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# === Лимиты параметров генерации (см. SHARED CONTRACT) ===
STEPS_MIN: int = 1
STEPS_MAX: int = 50
STEPS_STEP: int = 1

GUIDANCE_MIN: float = 1.0
GUIDANCE_MAX: float = 10.0
GUIDANCE_STEP: float = 0.5

# Дефолты генерации.
DEFAULT_STEPS: int = 4
DEFAULT_GUIDANCE: float = 1.0
DEFAULT_SEED: int = 0
DEFAULT_RANDOMIZE: bool = True

# Границы seed для api space.
SEED_MIN: int = 0
SEED_MAX: int = 2147483647


def _clamp_steps(value: int) -> int:
    """Зажать число шагов в диапазон [STEPS_MIN, STEPS_MAX] (целое)."""
    return max(STEPS_MIN, min(STEPS_MAX, int(value)))


def _clamp_guidance(value: float) -> float:
    """Зажать guidance в [GUIDANCE_MIN, GUIDANCE_MAX] и округлить до 1 знака."""
    clamped = max(GUIDANCE_MIN, min(GUIDANCE_MAX, float(value)))
    return round(clamped, 1)


def _clamp_seed(value: int) -> int:
    """Зажать seed в допустимый диапазон space [SEED_MIN, SEED_MAX]."""
    return max(SEED_MIN, min(SEED_MAX, int(value)))


@dataclass
class Settings:
    """Параметры генерации для одной сессии.

    Значения по умолчанию соответствуют дефолтам контракта.
    Используйте методы set_* / inc_* / dec_* — они применяют клампинг,
    так что значения всегда остаются в допустимых границах.
    """

    steps: int = DEFAULT_STEPS
    guidance: float = DEFAULT_GUIDANCE
    seed: int = DEFAULT_SEED
    randomize_seed: bool = DEFAULT_RANDOMIZE

    def __post_init__(self) -> None:
        # Нормализуем значения, переданные в конструктор.
        self.steps = _clamp_steps(self.steps)
        self.guidance = _clamp_guidance(self.guidance)
        self.seed = _clamp_seed(self.seed)
        self.randomize_seed = bool(self.randomize_seed)

    # --- steps ---
    def set_steps(self, value: int) -> int:
        """Установить число шагов (с клампингом). Вернуть новое значение."""
        self.steps = _clamp_steps(value)
        return self.steps

    def inc_steps(self) -> int:
        """Увеличить шаги на STEPS_STEP (с клампингом)."""
        return self.set_steps(self.steps + STEPS_STEP)

    def dec_steps(self) -> int:
        """Уменьшить шаги на STEPS_STEP (с клампингом)."""
        return self.set_steps(self.steps - STEPS_STEP)

    # --- guidance ---
    def set_guidance(self, value: float) -> float:
        """Установить guidance (с клампингом и округлением)."""
        self.guidance = _clamp_guidance(value)
        return self.guidance

    def inc_guidance(self) -> float:
        """Увеличить guidance на GUIDANCE_STEP (с клампингом)."""
        return self.set_guidance(self.guidance + GUIDANCE_STEP)

    def dec_guidance(self) -> float:
        """Уменьшить guidance на GUIDANCE_STEP (с клампингом)."""
        return self.set_guidance(self.guidance - GUIDANCE_STEP)

    # --- seed ---
    def set_seed(self, value: int) -> int:
        """Задать seed вручную; это автоматически отключает рандомизацию."""
        self.seed = _clamp_seed(value)
        self.randomize_seed = False
        return self.seed

    def toggle_randomize(self) -> bool:
        """Переключить флаг рандомизации seed. Вернуть новое значение."""
        self.randomize_seed = not self.randomize_seed
        return self.randomize_seed


@dataclass
class Session:
    """Состояние диалога с одним пользователем.

    awaiting управляет тем, как трактовать следующее текстовое сообщение:
        None     — обычный режим;
        "prompt" — ждём кастомный промпт от пользователя;
        "seed"   — ждём ввод числового seed.
    """

    image_b64: str | None = None
    image_name: str | None = None
    feature_key: str | None = None
    prompt: str | None = None
    settings: Settings = field(default_factory=Settings)
    awaiting: str | None = None  # {None, "prompt", "seed"}


class SessionStore:
    """In-memory хранилище сессий, ключ — telegram user_id."""

    def __init__(self) -> None:
        self._sessions: dict[int, Session] = {}

    def get(self, user_id: int) -> Session:
        """Вернуть сессию пользователя, создав пустую при первом обращении."""
        session = self._sessions.get(user_id)
        if session is None:
            session = Session()
            self._sessions[user_id] = session
        return session

    def reset(self, user_id: int) -> None:
        """Полностью сбросить сессию пользователя (новое фото / старт заново)."""
        self._sessions[user_id] = Session()


# Модульный синглтон — общий стор сессий для всего приложения.
store = SessionStore()
