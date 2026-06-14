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
    """Исчерпана дневная квота ZeroGPU на стороне Space."""


def _looks_like_quota_error(message: str) -> bool:
    """True, если текст ошибки похож на ошибку квоты ZeroGPU."""
    low = message.lower()
    return any(marker in low for marker in _QUOTA_MARKERS)


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
            try:
                logger.info("Создаю gradio_client.Client (auth) для Space %s", self.space_id)
                return Client(self.space_id, hf_token=self.hf_token)
            except TypeError:
                try:
                    return Client(
                        self.space_id,
                        headers={"Authorization": f"Bearer {self.hf_token}"},
                    )
                except TypeError:
                    logger.warning(
                        "gradio_client не принимает hf_token/headers — иду анонимно"
                    )
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
        """Вызвать predict; при ошибке соединения пересоздать клиент и повторить.

        Классифицирует исходную ошибку gradio в SpaceError/QuotaExceeded.
        """
        try:
            client = self._get_client()
            return self._call_predict(
                client, image_b64, prompt, lora, seed, randomize, guidance, steps
            )
        except Exception as exc:  # noqa: BLE001 — нужно поймать любые ошибки gradio
            message = str(exc) or exc.__class__.__name__

            # Ошибки квоты не лечатся ретраем — сразу пробрасываем.
            if _looks_like_quota_error(message):
                logger.warning("Квота ZeroGPU исчерпана: %s", message)
                raise QuotaExceeded(message) from exc

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
                        raise QuotaExceeded(message2) from exc2
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
