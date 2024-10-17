import logging
import time
import os

import telegram
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    AIORateLimiter,
    filters,
)
from typing import Any, Callable, Coroutine, Tuple, Optional
import handlers

logging.basicConfig(
    filename="/logs/groupai.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

logger = logging.getLogger(__name__)


def run_bot(bot: Application) -> None:
    allowed_list: list[int] = []
    
    for key, value in os.environ.items():
        if key.startswith("ALLOWED_"):
            allowed_list.append(int(value))
    
    allowed_list = filters.User(user_id=allowed_list)
    bot.add_handler(MessageHandler(filters.ALL & allowed_list, handlers.middleware_function), group=0)
    # bot.add_handler(CommandHandler("export", handlers.export_handler, filters=allowed_list), group=1)
    bot.add_handler(CommandHandler("help", handlers.help_handler, filters=allowed_list), group=1)
    bot.add_handler(CommandHandler("ask", handlers.ask_handler, filters=allowed_list), group=1)
    bot.add_handler(MessageHandler(filters.TEXT & allowed_list, handlers.message_handler), group=1)
    bot.add_error_handler(handlers.error_handler)
    bot.run_polling(poll_interval=0)


def build(
    token: str,
    cto: float,
    rto: float,
    wto: float,
    media_wto: float,
    pto: float,
    rate_limiter: AIORateLimiter,
    post_init_callback: Callable[[Application], Coroutine[Any, Any, None]],
) -> Application:
    return (
        ApplicationBuilder()
        .token(token)
        .concurrent_updates(True)
        .connect_timeout(cto)
        .read_timeout(rto)
        .write_timeout(wto)
        .media_write_timeout(media_wto)
        .pool_timeout(pto)
        .rate_limiter(rate_limiter)
        .post_init(post_init_callback)
        .build()
    )


def update_timeout_factor(tf: float, mf: float = 1.2, _max: float = 10) -> float:
    return round(min(tf * mf, _max), 1)


def update_delay(d: float, mf: float = 1.2, _max: float = 60.0) -> float:
    """Update delay time. (seconds)"""
    return round(min(d * mf, _max), 1)


async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([("/help", "Help Message")])


def update_timeout(
    factor: float,
    _connect_timeout: float,
    _read_timeout: float,
    _write_timeout: float,
    _media_write_timeout: float,
    _pool_timeout: float,
) -> Tuple[float, float, float, float, float]:
    return (
        _connect_timeout * factor,
        _read_timeout * factor,
        _write_timeout * factor,
        _media_write_timeout * factor,
        _pool_timeout * factor,
    )


if __name__ == "__main__":
    connect_timeout, pool_timeout = 5.0, 1.0
    read_timeout, write_timeout, media_write_timeout = 5.0, 5.0, 20.0
    timeout_factor = 1.2
    delay, delay_factor = 5.0, 1.5

    token = os.getenv("TLG_TOKEN")
    max_retry = int(os.getenv("MAX_RETRY", 5))

    while True:
        try:
            aio_rate_limiter = AIORateLimiter(
                overall_max_rate=10, overall_time_period=1, max_retries=max_retry
            )
            application = build(
                token,
                connect_timeout,
                read_timeout,
                write_timeout,
                media_write_timeout,
                pool_timeout,
                aio_rate_limiter,
                post_init,
            )
            run_bot(application)
            break
        except telegram.error.TimedOut as error:
            logger.error(f"{type(error)}: {str(error)}")  # AttributeError: type object 'TimedOut' has no attribute 'name'
            # Update timeout
            timeout_factor = update_timeout_factor(timeout_factor)
            (
                connect_timeout,
                read_timeout,
                write_timeout,
                media_write_timeout,
                pool_timeout,
            ) = update_timeout(
                timeout_factor,
                connect_timeout,
                read_timeout,
                write_timeout,
                media_write_timeout,
                pool_timeout,
            )
            # Update delay
            delay = update_delay(delay, delay_factor)
            time.sleep(delay)
