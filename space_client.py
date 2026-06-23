"""Клиент к Hugging Face Space (Qwen-Image-Edit-2509-LoRAs-Fast-v2).

Модуль изолирован: импортирует только gradio_client + стандартную библиотеку,
никаких зависимостей от telegram. Это позволяет тестировать его отдельно.

Публичный API:
    - class SpaceError(Exception)          — базовая ошибка работы со Space
    - class QuotaExceeded(SpaceError)      — исчерпана квота ZeroGPU
    - class SpaceClient                    — обёртка над gradio_client.Client
        .generate(...) -> tuple[bytes, int]
"""

from __future__ import annotations

import logging
import re
import time

from gradio_client import Client

logger = logging.getLogger(__name__)

# Имя инференс-эндпоинта в Space.
API_NAME = "/infer"

# Подстроки, по которым распознаём ошибку исчерпания квоты ZeroGPU.
# Сравнение регистронезависимое.
_QUOTA_MARKERS = ("quota", "exceeded", "gpu quota", "gpu task")


class SpaceError(Exception):
    """Базовая ошибка взаимодействия со Space."""


class QuotaExceeded(SpaceError):
    """Исчерпана дневная квота ZeroGPU на стороне Space.

    remaining_seconds / retry_seconds — точные цифры из текста ошибки HF
    («... X left. Retry in Y»), если их удалось распарсить, иначе None.
    Это единственный момент, когда HF сообщает реальный остаток квоты.
    """

    def __init__(
        self,
        message: str,
        remaining_seconds: float | None = None,
        retry_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.remaining_seconds = remaining_seconds
        self.retry_seconds = retry_seconds


def _looks_like_quota_error(message: str) -> bool:
    """True, если текст ошибки похож на ошибку квоты ZeroGPU."""
    low = message.lower()
    return any(marker in low for marker in _QUOTA_MARKERS)


# ZeroGPU не смог ВЫДЕЛИТЬ слот (а не исчерпал квоту). На xlarge нужно 2 слота
# сразу — под нагрузкой планировщик отдаёт «No GPU was available after 60s» или
# роняет задачу («GPU task aborted»). Это ТРАНЗИЕНТНО: повтор через паузу обычно
# ловит освободившийся слот. Проверяем ДО квоты, чтобы «task aborted» не выдавался
# за «квота исчерпана» (его ловил маркер "gpu task").
_GPU_UNAVAIL_MARKERS = ("no gpu", "gpu task aborted", "gpu was not available")


def _looks_like_gpu_unavailable(message: str) -> bool:
    """True, если ZeroGPU не выделил/уронил GPU-слот (транзиентно, ретраибельно)."""
    low = message.lower()
    return any(marker in low for marker in _GPU_UNAVAIL_MARKERS)


# Парсинг точных чисел из текста ошибки квоты ZeroGPU.
_RE_LEFT = re.compile(r"(\d+(?:\.\d+)?)\s*s(?:econds)?\s*left", re.IGNORECASE)
_RE_VS = re.compile(r"vs\.?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
# HF пишет «Try again in 2:39:03» (или «retry in …») — ловим оба варианта.
_RE_RETRY_HMS = re.compile(r"(?:try again|retry) in\s*(\d{1,2}:\d{2}(?::\d{2})?)", re.IGNORECASE)
_RE_RETRY_S = re.compile(r"(?:try again|retry) in\s*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _parse_quota_numbers(message: str) -> tuple[float | None, float | None]:
    """Достать (remaining_seconds, retry_seconds) из текста ошибки квоты."""
    remaining: float | None = None
    m = _RE_LEFT.search(message) or _RE_VS.search(message)
    if m:
        try:
            remaining = float(m.group(1))
        except ValueError:
            remaining = None

    retry: float | None = None
    mh = _RE_RETRY_HMS.search(message)
    if mh:
        secs = 0
        for part in mh.group(1).split(":"):
            secs = secs * 60 + int(part)
        retry = float(secs)
    else:
        ms = _RE_RETRY_S.search(message)
        if ms:
            try:
                retry = float(ms.group(1))
            except ValueError:
                retry = None
    return remaining, retry


class SpaceClient:
    """Синхронная обёртка над gradio_client.Client.

    Клиент создаётся лениво при первом вызове и кэшируется. При ошибке
    соединения клиент пересоздаётся один раз и вызов повторяется.

    Все методы блокирующие/синхронные — вызывающая сторона (бот) должна
    запускать их через asyncio.to_thread, чтобы не блокировать event loop.
    """

    def __init__(self, space_id: str, hf_token: str | None = None, timeout: int = 300) -> None:
        self.space_id = space_id
        self.hf_token = hf_token
        self.timeout = timeout
        self._client: Client | None = None

    # --- управление клиентом ------------------------------------------------

    def _build_client(self) -> Client:
        """Создать новый gradio_client.Client.

        Если задан hf_token — передаём его, чтобы запросы шли от имени нашего
        HF-аккаунта и расходовали его PRO-квоту ZeroGPU (40 мин/день), а не
        анонимный тариф (~2 мин/день). На старых версиях gradio_client, где
        kwarg hf_token/headers отсутствует, аккуратно откатываемся на аноним.
        """
        if self.hf_token:
            # У gradio_client менялось имя параметра авторизации: сейчас это
            # token= (>=1.x), раньше hf_token=, на совсем старых — только
            # заголовок. Пробуем по очереди; любой успех = ходим от имени
            # владельца токена → расход по его PRO-квоте ZeroGPU (40 мин/день),
            # а не по анонимному тарифу (~2 мин/день).
            for kwargs in (
                {"token": self.hf_token},
                {"hf_token": self.hf_token},
                {"headers": {"Authorization": f"Bearer {self.hf_token}"}},
            ):
                try:
                    logger.info(
                        "Создаю gradio_client.Client (auth via %s) для Space %s",
                        next(iter(kwargs)),
                        self.space_id,
                    )
                    return Client(self.space_id, **kwargs)
                except TypeError:
                    continue
            logger.warning("gradio_client не принял авторизацию — иду анонимно")
        logger.info("Создаю gradio_client.Client (anon) для Space %s", self.space_id)
        return Client(self.space_id)

    def _get_client(self) -> Client:
        """Вернуть закэшированный клиент, создав его при необходимости."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # --- основной вызов -----------------------------------------------------

    def generate(
        self,
        image_b64: str,
        prompt: str,
        lora: str,
        seed: int,
        randomize: bool,
        guidance: float,
        steps: int,
    ) -> tuple[bytes, int]:
        """Сгенерировать изображение.

        Аргументы соответствуют сигнатуре api_name="/infer":
            predict(image_b64, prompt, lora_adapter, seed, randomize_seed,
                    guidance_scale, steps)

        :param image_b64: base64 data-URI строка ("data:image/jpeg;base64,...").
        :param prompt: текстовый промпт.
        :param lora: точное имя LoRA-адаптера (поле Feature.lora).
        :param seed: int 0..2147483647.
        :param randomize: пересоздавать ли seed на стороне Space.
        :param guidance: guidance_scale, float 1.0..10.0.
        :param steps: число шагов, int 1..50.
        :returns: (png_bytes, used_seed) — байты PNG и фактически
                  использованный seed.
        :raises QuotaExceeded: если ошибка похожа на исчерпание квоты ZeroGPU.
        :raises SpaceError: при любой другой ошибке вызова Space.
        """
        result = self._predict_with_retry(
            image_b64, prompt, lora, seed, randomize, guidance, steps
        )

        # Space возвращает (image_filepath, used_seed).
        try:
            image_filepath, used_seed = result
        except (TypeError, ValueError) as exc:
            raise SpaceError(f"Неожиданный ответ Space: {result!r}") from exc

        # Читаем PNG с диска в байты.
        try:
            with open(image_filepath, "rb") as fh:
                png_bytes = fh.read()
        except OSError as exc:
            raise SpaceError(f"Не удалось прочитать результат: {exc}") from exc

        # used_seed может прийти как float/str — приводим к int безопасно.
        try:
            used_seed_int = int(used_seed)
        except (TypeError, ValueError):
            used_seed_int = seed

        return png_bytes, used_seed_int

    def _predict_with_retry(
        self,
        image_b64: str,
        prompt: str,
        lora: str,
        seed: int,
        randomize: bool,
        guidance: float,
        steps: int,
    ):
        """Вызвать predict с ретраями и классификацией ошибки gradio:

          - GPU-слот не выделен / задача уронена → транзиент: пауза + повтор (до 2 раз);
          - квота → QuotaExceeded (ретрай не помогает);
          - сбой соединения → пересоздать клиент и повторить один раз;
          - прочее → SpaceError.
        """
        GPU_RETRIES = 2     # доп. попытки при «No GPU available» / aborted-слоте
        GPU_BACKOFF = 8     # пауза перед повторной постановкой в очередь, сек
        gpu_attempt = 0
        while True:
            try:
                client = self._get_client()
                return self._call_predict(
                    client, image_b64, prompt, lora, seed, randomize, guidance, steps
                )
            except Exception as exc:  # noqa: BLE001 — нужно поймать любые ошибки gradio
                message = str(exc) or exc.__class__.__name__

                # ZeroGPU не выделил слот / уронил задачу — транзиентно, повторяем.
                # Проверяем ПЕРВЫМ, чтобы «GPU task aborted» не выдавался за квоту.
                if _looks_like_gpu_unavailable(message):
                    if gpu_attempt < GPU_RETRIES:
                        gpu_attempt += 1
                        logger.warning(
                            "ZeroGPU слот не выделен (%s) — повтор %d/%d через %dс",
                            message, gpu_attempt, GPU_RETRIES, GPU_BACKOFF,
                        )
                        time.sleep(GPU_BACKOFF)
                        continue
                    logger.error(
                        "ZeroGPU не выделил слот после %d попыток: %s", GPU_RETRIES + 1, message
                    )
                    raise SpaceError(
                        "ZeroGPU перегружен — не удалось получить GPU-слот после %d попыток. %s"
                        % (GPU_RETRIES + 1, message)
                    ) from exc

                # Ошибки квоты не лечатся ретраем — сразу пробрасываем.
                if _looks_like_quota_error(message):
                    logger.warning("Квота ZeroGPU исчерпана: %s", message)
                    rem, retry = _parse_quota_numbers(message)
                    raise QuotaExceeded(message, rem, retry) from exc

                # Похоже на проблему соединения — пересоздаём клиент и пробуем ещё раз.
                if self._is_connection_error(exc):
                    logger.warning(
                        "Похоже на сбой соединения (%s), пересоздаю клиент и повторяю", message
                    )
                    self._client = None
                    try:
                        client = self._get_client()
                        return self._call_predict(
                            client, image_b64, prompt, lora, seed, randomize, guidance, steps
                        )
                    except Exception as exc2:  # noqa: BLE001
                        message2 = str(exc2) or exc2.__class__.__name__
                        if _looks_like_quota_error(message2):
                            logger.warning("Квота ZeroGPU исчерпана: %s", message2)
                            rem2, retry2 = _parse_quota_numbers(message2)
                            raise QuotaExceeded(message2, rem2, retry2) from exc2
                        logger.error("Повторный вызов Space не удался: %s", message2)
                        raise SpaceError(message2) from exc2

                # Любая прочая ошибка.
                logger.error("Ошибка вызова Space: %s", message)
                raise SpaceError(message) from exc

    @staticmethod
    def _call_predict(
        client: Client,
        image_b64: str,
        prompt: str,
        lora: str,
        seed: int,
        randomize: bool,
        guidance: float,
        steps: int,
    ):
        """Единственное место, где реально дёргается client.predict(...)."""
        return client.predict(
            image_b64,
            prompt,
            lora,
            seed,
            randomize,
            guidance,
            steps,
            api_name=API_NAME,
        )

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Грубая эвристика: похожа ли ошибка на сетевую/соединения.

        Ретраить имеет смысл именно такие — когда Space «уснул» или
        соединение оборвалось. Опираемся и на тип, и на текст.
        """
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return True
        low = str(exc).lower()
        markers = (
            "connection",
            "connect",
            "timed out",
            "timeout",
            "refused",
            "reset",
            "broken pipe",
            "remote end closed",
            "max retries",
            "temporarily unavailable",
            "502",
            "503",
            "504",
        )
        return any(marker in low for marker in markers)
