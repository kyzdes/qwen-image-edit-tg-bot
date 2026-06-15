"""Учёт расхода ZeroGPU-квоты ботом — своими силами.

HF не отдаёт остаток квоты по публичному API (проверено: в ответах ZeroGPU
нет полей квоты, только generic API-rate-limit). Поэтому считаем сами:
число генераций за текущее скользящее 24-часовое окно + грубая оценка
GPU-времени. А когда HF присылает ошибку исчерпания квоты — кэшируем из неё
точные цифры (остаток/время сброса), это единственный авторитетный источник.

Состояние общее на весь бот (квота ZeroGPU — один пул на аккаунт).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

WINDOW_SECONDS = 24 * 3600
XLARGE_MULTIPLIER = 2  # этот Space = xlarge → расход квоты ×2


@dataclass
class QuotaReport:
    """Точные цифры из текста ошибки квоты HF (если распарсились)."""

    remaining_seconds: float | None
    retry_seconds: float | None
    at: float  # time.time() в момент получения


class UsageTracker:
    """Скользящее 24-ч окно расхода + кэш последней ошибки квоты."""

    def __init__(self) -> None:
        self.window_start: float | None = None
        self.generations: int = 0
        self.est_gpu_seconds: float = 0.0
        self.last_quota: QuotaReport | None = None

    def _roll(self) -> None:
        now = time.time()
        if self.window_start is None or now - self.window_start >= WINDOW_SECONDS:
            self.window_start = now
            self.generations = 0
            self.est_gpu_seconds = 0.0

    def record_generation(self, wall_seconds: float) -> None:
        """Зафиксировать успешную генерацию (wall_seconds — длительность вызова)."""
        self._roll()
        self.generations += 1
        # Грубо: верхняя оценка GPU-расхода = wall-time × 2 (xlarge).
        self.est_gpu_seconds += max(0.0, wall_seconds) * XLARGE_MULTIPLIER

    def record_quota_error(
        self, remaining_seconds: float | None, retry_seconds: float | None
    ) -> None:
        # Ошибка квоты — тоже активность: если окно ещё не открыто, открываем,
        # чтобы оценка сброса (window_start + 24ч) работала даже без успешных
        # генераций в этой сессии.
        if self.window_start is None:
            self.window_start = time.time()
        self.last_quota = QuotaReport(remaining_seconds, retry_seconds, time.time())

    def reset_in_seconds(self) -> float | None:
        if self.window_start is None:
            return None
        return max(0.0, WINDOW_SECONDS - (time.time() - self.window_start))


# Глобальный синглтон на весь процесс бота.
tracker = UsageTracker()
