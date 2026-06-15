"""Учёт расхода ZeroGPU-квоты ботом — своими силами, с персистом на диск.

HF не отдаёт остаток квоты по публичному API (проверено: в ответах ZeroGPU
нет полей квоты). Поэтому считаем сами: число генераций за текущее скользящее
24-часовое окно + грубая оценка GPU-времени. А когда HF присылает ошибку
исчерпания квоты — кэшируем из неё точные цифры (остаток + «Try again in …»),
это единственный авторитетный источник.

Состояние общее на весь бот (квота ZeroGPU — один пул на аккаунт) и
сохраняется в JSON-файл (STATE_FILE, по умолчанию на volume /data), чтобы
пережить рестарты/редеплои. Если файл недоступен — работаем in-memory.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 24 * 3600
XLARGE_MULTIPLIER = 2  # этот Space = xlarge → расход квоты ×2

# Путь к файлу состояния. На Dokploy сюда монтируется persistent volume.
STATE_FILE = os.environ.get("STATE_FILE", "/data/usage_state.json")


@dataclass
class QuotaReport:
    """Точные цифры из текста ошибки квоты HF (если распарсились)."""

    remaining_seconds: float | None
    retry_seconds: float | None
    at: float  # time.time() в момент получения


class UsageTracker:
    """Скользящее 24-ч окно расхода + кэш последней ошибки квоты (с персистом)."""

    def __init__(self) -> None:
        self.window_start: float | None = None
        self.generations: int = 0
        self.est_gpu_seconds: float = 0.0
        self.last_quota: QuotaReport | None = None
        self._load()

    # --- персист ------------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            self.window_start = d.get("window_start")
            self.generations = int(d.get("generations", 0))
            self.est_gpu_seconds = float(d.get("est_gpu_seconds", 0.0))
            lq = d.get("last_quota")
            if lq:
                self.last_quota = QuotaReport(
                    lq.get("remaining_seconds"),
                    lq.get("retry_seconds"),
                    float(lq.get("at", 0.0)),
                )
            logger.info("Состояние расхода загружено из %s", STATE_FILE)
        except FileNotFoundError:
            logger.info("Файл состояния %s не найден — старт с нуля", STATE_FILE)
        except Exception as exc:  # noqa: BLE001 — персист не должен ронять бота
            logger.warning("Не удалось загрузить состояние (%s): %s", STATE_FILE, exc)

    def _save(self) -> None:
        data = {
            "window_start": self.window_start,
            "generations": self.generations,
            "est_gpu_seconds": self.est_gpu_seconds,
            "last_quota": asdict(self.last_quota) if self.last_quota else None,
        }
        try:
            os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, STATE_FILE)  # атомарная запись
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось сохранить состояние (%s): %s", STATE_FILE, exc)

    # --- логика -------------------------------------------------------------

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
        self._save()

    def record_quota_error(
        self, remaining_seconds: float | None, retry_seconds: float | None
    ) -> None:
        # Ошибка квоты — тоже активность: если окно ещё не открыто, открываем,
        # чтобы оценка сброса (window_start + 24ч) работала даже без успешных
        # генераций в этой сессии.
        if self.window_start is None:
            self.window_start = time.time()
        self.last_quota = QuotaReport(remaining_seconds, retry_seconds, time.time())
        self._save()

    def reset_in_seconds(self) -> float | None:
        if self.window_start is None:
            return None
        return max(0.0, WINDOW_SECONDS - (time.time() - self.window_start))


# Глобальный синглтон на весь процесс бота (грузит состояние при импорте).
tracker = UsageTracker()
