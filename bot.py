"""bot.py — точка входа Telegram-бота "Qwen Image Edit".

Собирает PTB v21 Application, регистрирует все хендлеры, маршрутизирует
callback-кнопки и запускает генерацию через backend (HF Space) в отдельном
потоке (asyncio.to_thread), чтобы не блокировать event loop.

Запуск:  python bot.py   (или CMD из Dockerfile)
Конфиг читается из окружения ВНУТРИ main() (см. config.load_config()).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from io import BytesIO

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import Config, load_config
from features import FEATURES, FEATURES_BY_KEY
from keyboards import (
    directions_keyboard,
    main_menu,
    result_keyboard,
    settings_keyboard,
)
from space_client import QuotaExceeded, SpaceClient, SpaceError
from state import store

logger = logging.getLogger(__name__)

# Ключи, под которыми мы кладём объекты в application.bot_data.
BD_CONFIG = "config"
BD_SPACE = "space"

# Один раз залогировать предупреждение про пустой allowlist.
_warned_open_allowlist = False


# --------------------------------------------------------------------------- #
# Доступ / allowlist
# --------------------------------------------------------------------------- #
def is_allowed(user_id: int, config: Config) -> bool:
    """True, если пользователю разрешён доступ.

    Пустой allowlist => бот открыт для всех (с однократным предупреждением).
    """
    global _warned_open_allowlist
    if not config.allowed_user_ids:
        if not _warned_open_allowlist:
            logger.warning(
                "ALLOWED_USER_IDS пуст — бот открыт для ВСЕХ пользователей. "
                "Заполни ALLOWED_USER_IDS, чтобы ограничить доступ."
            )
            _warned_open_allowlist = True
        return True
    return user_id in config.allowed_user_ids


def _get_config(context: ContextTypes.DEFAULT_TYPE) -> Config:
    return context.application.bot_data[BD_CONFIG]


def _get_space(context: ContextTypes.DEFAULT_TYPE) -> SpaceClient:
    return context.application.bot_data[BD_SPACE]


async def _guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Проверка доступа. Если не разрешён — отвечает и возвращает False."""
    user = update.effective_user
    config = _get_config(context)
    if user is None or not is_allowed(user.id, config):
        if update.callback_query is not None:
            await update.callback_query.answer("🔒 Бот приватный", show_alert=True)
        elif update.effective_message is not None:
            await update.effective_message.reply_text("🔒 Бот приватный")
        return False
    return True


# --------------------------------------------------------------------------- #
# Команды /start, /help
# --------------------------------------------------------------------------- #
WELCOME_TEXT = (
    "👋 Привет! Это бот для редактирования изображений на базе "
    "Qwen-Image-Edit (LoRA).\n\n"
    "Как пользоваться:\n"
    "1️⃣ Пришли фото (или картинку файлом).\n"
    "2️⃣ Выбери фичу из меню.\n"
    "3️⃣ При желании напиши свой prompt текстом — иначе используется "
    "стандартный.\n"
    "4️⃣ Подкрути настройки (⚙️ шаги / cfg / seed) — по желанию.\n"
    "5️⃣ Жми ✨ Сгенерировать.\n\n"
    "Доступные фичи:\n"
    "🎨 Аниме · 🔄 Сменить ракурс · 💡 Убрать тени · 🌅 Пересвет · "
    "🔦 Направленный свет · 🧑 Детали кожи · 🎬 Следующая сцена · "
    "🎞 Flat/Log цвет · ⬆️ Апскейл · 🖼 Апскейл 2K · ⚫ Точечная иллюстрация.\n\n"
    "Команды: /start, /help — это сообщение · /limits — лимиты GPU."
)

# Текст про лимиты ZeroGPU. Бот авторизуется в Space как PRO-аккаунт, поэтому
# и показываем PRO-тариф. Живой остаток в секундах HF наружу по API не отдаёт —
# поэтому даём факты по тарифу + ссылку на страницу с живым индикатором.
LIMITS_TEXT = (
    "📊 Лимиты ZeroGPU\n\n"
    "Лимит считается во времени GPU в день на аккаунт (общий пул на все "
    "ZeroGPU-Space), а не в числе запросов. Бот ходит в Space как PRO-аккаунт:\n\n"
    "• Тариф PRO → 40 мин GPU в день, наивысший приоритет в очереди.\n"
    "• Этот Space = xlarge (полная RTX Pro 6000 Blackwell, 96 ГБ) → расход ×2, "
    "то есть ~20 мин реальной генерации в день.\n"
    "• Сброс: через 24 ч после первого использования (скользящее окно).\n"
    "• Сверх лимита (PRO): доплата $1 за 10 мин GPU из кредитов.\n\n"
    "Живой остаток квоты — на странице Space (индикатор вверху) и в биллинге:\n"
    "https://huggingface.co/spaces/productowner/Qwen-Image-Edit-2509-LoRAs-Fast-v2\n"
    "https://huggingface.co/settings/billing"
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    await update.effective_message.reply_text(WELCOME_TEXT)


async def cmd_limits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/limits — показать лимиты ZeroGPU."""
    if not await _guard(update, context):
        return
    await update.effective_message.reply_text(
        LIMITS_TEXT, disable_web_page_preview=True
    )


# --------------------------------------------------------------------------- #
# Приём фото / картинок
# --------------------------------------------------------------------------- #
async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Фото (filters.PHOTO) или картинка-документ (filters.Document.IMAGE).

    Скачиваем, кодируем в base64 data-URI (image/jpeg) и кладём в сессию.
    """
    if not await _guard(update, context):
        return

    message = update.effective_message
    user_id = update.effective_user.id

    # Определяем file_id: для photo берём самое крупное представление,
    # для документа-картинки — сам документ.
    file_name = "photo.jpg"
    if message.photo:
        file_id = message.photo[-1].file_id
    elif message.document is not None:
        file_id = message.document.file_id
        file_name = message.document.file_name or "image"
    else:  # на всякий случай
        await message.reply_text("Не вижу картинку. Пришли фото 🙂")
        return

    try:
        tg_file = await context.bot.get_file(file_id)
        raw = await tg_file.download_as_bytearray()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не удалось скачать фото")
        await message.reply_text(f"❌ Не смог скачать фото: {exc}")
        return

    # base64 data-URI. Для Telegram-фото это всегда JPEG.
    b64 = base64.b64encode(bytes(raw)).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64}"

    session = store.get(user_id)
    session.image_b64 = data_uri
    session.image_name = file_name
    # Новое фото => сбрасываем выбранную фичу / prompt / ожидание ввода.
    session.feature_key = None
    session.prompt = None
    session.awaiting = None

    await message.reply_text("Фото получено ✅ выбери фичу:", reply_markup=main_menu())


# --------------------------------------------------------------------------- #
# Текстовые сообщения (не команды)
# --------------------------------------------------------------------------- #
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return

    message = update.effective_message
    user_id = update.effective_user.id
    session = store.get(user_id)
    text = (message.text or "").strip()

    # 1) Ждём seed.
    if session.awaiting == "seed":
        try:
            value = int(text)
        except ValueError:
            await message.reply_text("Нужно целое число. Попробуй ещё раз 🙂")
            return
        session.settings.set_seed(value)
        session.awaiting = None
        await message.reply_text(
            f"Seed зафиксирован: {session.settings.seed} (рандом выключен).",
            reply_markup=settings_keyboard(session.settings),
        )
        return

    # 2) Ждём prompt — сразу генерируем с этим текстом.
    if session.awaiting == "prompt":
        session.awaiting = None
        session.prompt = text
        await run_generation(update, context, randomize_override=False)
        return

    # 3) Фича уже выбрана и есть фото — трактуем текст как кастомный prompt.
    if session.feature_key is not None and session.image_b64 is not None:
        session.prompt = text
        await run_generation(update, context, randomize_override=False)
        return

    # 4) Иначе — мягкая подсказка.
    if session.image_b64 is None:
        await message.reply_text("Сначала пришли фото 📷, потом выбери фичу.")
    else:
        await message.reply_text(
            "Выбери фичу из меню 👇", reply_markup=main_menu()
        )


# --------------------------------------------------------------------------- #
# Callback-кнопки
# --------------------------------------------------------------------------- #
def _ready_keyboard() -> InlineKeyboardMarkup:
    """Маленькая клавиатура состояния «готов генерить»."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✨ Сгенерировать", callback_data="gen"),
                InlineKeyboardButton("🎚 Настройки", callback_data="set"),
            ],
            [InlineKeyboardButton("📋 Меню", callback_data="menu")],
        ]
    )


async def _show(query, text: str, reply_markup=None) -> None:
    """Показать текст+клавиатуру, корректно работая с медиа-сообщениями.

    Кнопки result_keyboard() висят под фото/документом результата. Telegram
    НЕ позволяет редактировать текст у медиа-сообщения (edit_message_text
    кидает BadRequest), поэтому при неудаче отправляем новое сообщение.
    """
    try:
        await query.edit_message_text(text, reply_markup=reply_markup)
    except BadRequest:
        await query.message.reply_text(text, reply_markup=reply_markup)


async def _show_ready(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сообщение «готов генерить» с подсказкой про свой prompt."""
    user_id = update.effective_user.id
    session = store.get(user_id)
    feature = FEATURES_BY_KEY[session.feature_key]
    text = (
        f"Выбрано: {feature.label}\n"
        f"Prompt: {session.prompt}\n\n"
        "Жми ✨ Сгенерировать — или пришли свой текст-prompt, "
        "и я сразу сгенерирую с ним."
    )
    query = update.callback_query
    if query is not None:
        await _show(query, text, reply_markup=_ready_keyboard())
    else:
        await update.effective_message.reply_text(
            text, reply_markup=_ready_keyboard()
        )


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return

    query = update.callback_query
    await query.answer()  # всегда подтверждаем callback
    data = query.data or ""
    user_id = update.effective_user.id
    session = store.get(user_id)

    # --- Выбор фичи: "f:<key>" ---------------------------------------- #
    if data.startswith("f:"):
        key = data[2:]
        feature = FEATURES_BY_KEY.get(key)
        if feature is None:
            # callback уже подтверждён выше (строка с query.answer()); второй answer
            # на тот же query упал бы BadRequest — отвечаем обычным сообщением.
            await query.message.reply_text("Неизвестная фича")
            return
        session.feature_key = key
        session.awaiting = None
        if feature.directions:
            session.prompt = None
            await _show(
                query,
                f"{feature.label}: выбери направление 👇",
                reply_markup=directions_keyboard(key),
            )
        else:
            session.prompt = feature.default_prompt
            await _show_ready(update, context)
        return

    # --- Выбор направления: "d:<key>:<idx>" --------------------------- #
    if data.startswith("d:"):
        try:
            _, key, idx_str = data.split(":", 2)
            idx = int(idx_str)
            feature = FEATURES_BY_KEY[key]
            label, prompt = feature.directions[idx]
        except (ValueError, KeyError, IndexError, TypeError):
            await query.message.reply_text("Неизвестное направление")
            return
        session.feature_key = key
        session.prompt = prompt
        session.awaiting = None
        await _show_ready(update, context)
        return

    # --- Генерация ----------------------------------------------------- #
    if data == "gen":
        await run_generation(update, context, randomize_override=None)
        return

    if data == "again":
        # «Повторить» — форсируем новый случайный seed.
        await run_generation(update, context, randomize_override=True)
        return

    # --- Настройки ----------------------------------------------------- #
    if data == "set":
        await _show(
            query,
            "⚙️ Настройки генерации:",
            reply_markup=settings_keyboard(session.settings),
        )
        return

    if data.startswith("s:"):
        await _handle_settings(update, context, data)
        return

    # --- Навигация ----------------------------------------------------- #
    if data == "menu":
        session.awaiting = None
        await _show(query, "Выбери фичу 👇", reply_markup=main_menu())
        return

    if data == "newphoto":
        store.reset(user_id)
        await _show(query, "Пришли новое фото 📷")
        return

    # --- Лимиты GPU ---------------------------------------------------- #
    if data == "limits":
        await _show(query, LIMITS_TEXT, reply_markup=main_menu())
        return

    # Неизвестный callback — молча игнорируем (уже ответили на query).
    logger.debug("Неизвестный callback_data: %s", data)


async def _handle_settings(
    update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
) -> None:
    """Мутации настроек по схеме s:* с clamp'ом и пере-рендером клавиатуры."""
    from state import GUIDANCE_STEP, STEPS_STEP

    query = update.callback_query
    user_id = update.effective_user.id
    session = store.get(user_id)
    settings = session.settings

    if data == "s:steps:-":
        settings.steps = _clamp_steps(settings.steps - STEPS_STEP)
    elif data == "s:steps:+":
        settings.steps = _clamp_steps(settings.steps + STEPS_STEP)
    elif data == "s:cfg:-":
        settings.guidance = _clamp_guidance(settings.guidance - GUIDANCE_STEP)
    elif data == "s:cfg:+":
        settings.guidance = _clamp_guidance(settings.guidance + GUIDANCE_STEP)
    elif data == "s:rand":
        settings.randomize_seed = not settings.randomize_seed
    elif data == "s:seed":
        session.awaiting = "seed"
        await _show(query, "✏️ Пришли число для seed (целое 0..2147483647):")
        return
    elif data == "s:back":
        session.awaiting = None
        await _show(query, "Выбери фичу 👇", reply_markup=main_menu())
        return
    else:
        return

    # Пере-рендерим клавиатуру настроек с актуальными значениями.
    try:
        await query.edit_message_reply_markup(
            reply_markup=settings_keyboard(settings)
        )
    except BadRequest:
        await query.message.reply_text(
            "⚙️ Настройки генерации:",
            reply_markup=settings_keyboard(settings),
        )


def _clamp_steps(value: int) -> int:
    """Clamp шагов через state.Settings (на случай локальной правки)."""
    from state import STEPS_MAX, STEPS_MIN

    return max(STEPS_MIN, min(STEPS_MAX, int(value)))


def _clamp_guidance(value: float) -> float:
    from state import GUIDANCE_MAX, GUIDANCE_MIN

    return round(max(GUIDANCE_MIN, min(GUIDANCE_MAX, float(value))), 1)


# --------------------------------------------------------------------------- #
# Генерация
# --------------------------------------------------------------------------- #
async def run_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    randomize_override: bool | None = None,
) -> None:
    """Запуск инференса в HF Space через asyncio.to_thread.

    randomize_override:
      - None  => использовать settings.randomize_seed как есть;
      - True  => форсировать новый случайный seed (кнопка «Повторить»);
      - False => не трогать флаг (после ввода своего prompt/seed).
    """
    message = update.effective_message
    user_id = update.effective_user.id
    session = store.get(user_id)
    settings = session.settings

    # Валидация наличия данных.
    if session.image_b64 is None:
        await _reply(update, "Сначала пришли фото 📷.")
        return
    if session.feature_key is None:
        await _reply(update, "Выбери фичу 👇", reply_markup=main_menu())
        return
    if not session.prompt:
        await _reply(
            update,
            "Не задан prompt. Выбери фичу или пришли свой текст.",
            reply_markup=main_menu(),
        )
        return

    feature = FEATURES_BY_KEY[session.feature_key]
    lora = feature.lora

    # Решаем, рандомить ли seed.
    if randomize_override is True:
        randomize = True
    elif randomize_override is False:
        randomize = False
    else:
        randomize = settings.randomize_seed

    # Информируем пользователя.
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_PHOTO)
    status = await _reply(update, "⏳ Генерирую…")

    space = _get_space(context)
    config = _get_config(context)

    try:
        png_bytes, used_seed = await asyncio.to_thread(
            space.generate,
            session.image_b64,
            session.prompt,
            lora,
            settings.seed,
            randomize,
            settings.guidance,
            settings.steps,
        )
    except QuotaExceeded:
        logger.warning("ZeroGPU quota exceeded", exc_info=True)
        await _safe_edit(
            status,
            "⚠️ Лимит ZeroGPU на сегодня исчерпан. "
            "Квота сбрасывается через ~24ч после первого использования.",
        )
        return
    except SpaceError as exc:
        logger.exception("SpaceError при генерации")
        await _safe_edit(
            status,
            f"❌ Не получилось: {_short(str(exc))}. "
            "Попробуй ещё раз (🔄) или другое фото.",
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Неожиданная ошибка при генерации")
        await _safe_edit(
            status,
            f"❌ Не получилось: {_short(str(exc))}. "
            "Попробуй ещё раз (🔄) или другое фото.",
        )
        return

    # Сохраняем использованный seed, выключаем рандом — чтобы «Повторить»
    # мог переиспользовать или, при нажатии again, заново заролить.
    session.settings.seed = int(used_seed)
    session.settings.randomize_seed = False

    caption = (
        f"{feature.label}\n"
        f"{session.prompt.strip()}\n"
        f"seed: {used_seed}\n"
        f"steps: {settings.steps}, cfg: {settings.guidance}"
    )

    # Убираем статусное сообщение (best-effort) и отправляем результат.
    await _safe_delete(status)

    bio = BytesIO(png_bytes)
    bio.name = "result.png"
    bio.seek(0)
    try:
        if feature.is_upscale:
            # Апскейлы — без сжатия, документом.
            await context.bot.send_document(
                chat_id=chat_id,
                document=bio,
                filename="result.png",
                caption=caption,
                reply_markup=result_keyboard(),
            )
        else:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=bio,
                caption=caption,
                reply_markup=result_keyboard(),
            )
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось отправить результат")
        await context.bot.send_message(
            chat_id,
            "❌ Сгенерировал, но не смог отправить результат. Попробуй ещё раз.",
        )


# --------------------------------------------------------------------------- #
# Вспомогательные функции отправки
# --------------------------------------------------------------------------- #
async def _reply(update: Update, text: str, reply_markup=None):
    """Отправить новое сообщение в чат (универсально для message/callback)."""
    if update.effective_message is not None:
        return await update.effective_message.reply_text(
            text, reply_markup=reply_markup
        )
    chat = update.effective_chat
    if chat is not None:
        return await chat.send_message(text, reply_markup=reply_markup)
    return None


async def _safe_edit(message, text: str) -> None:
    """Аккуратно отредактировать статусное сообщение."""
    if message is None:
        return
    try:
        await message.edit_text(text)
    except Exception:  # noqa: BLE001
        # Если редактирование не удалось — шлём новым сообщением.
        try:
            await message.reply_text(text)
        except Exception:  # noqa: BLE001
            logger.debug("Не удалось доставить статус-сообщение", exc_info=True)


async def _safe_delete(message) -> None:
    if message is None:
        return
    try:
        await message.delete()
    except Exception:  # noqa: BLE001
        pass


def _short(msg: str, limit: int = 200) -> str:
    """Короткий однострочный фрагмент ошибки для пользователя."""
    one_line = " ".join(str(msg).split())
    return one_line[:limit]


# --------------------------------------------------------------------------- #
# Глобальный обработчик ошибок
# --------------------------------------------------------------------------- #
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Необработанная ошибка в хендлере", exc_info=context.error)


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Конфиг и SpaceClient создаём ВНУТРИ main (никогда на import).
    config = load_config()
    space = SpaceClient(
        space_id=config.space_id,
        hf_token=config.hf_token,
        timeout=config.request_timeout,
    )

    application = Application.builder().token(config.bot_token).build()
    application.bot_data[BD_CONFIG] = config
    application.bot_data[BD_SPACE] = space

    # Команды.
    application.add_handler(CommandHandler(["start", "help"], cmd_start))
    application.add_handler(CommandHandler("limits", cmd_limits))

    # Фото и картинки-документы.
    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, on_photo)
    )

    # Текст (не команды).
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text)
    )

    # Callback-кнопки.
    application.add_handler(CallbackQueryHandler(on_callback))

    application.add_error_handler(on_error)

    n = len(FEATURES)
    logger.info("Бот запускается. Фич загружено: %d. Space: %s", n, config.space_id)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
