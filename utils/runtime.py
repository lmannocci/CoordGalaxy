from __future__ import annotations

import os
import sys
from types import TracebackType
from typing import Type

try:
    from telegram_send_message import telegram_send as telegram_client
except ModuleNotFoundError:
    class _TelegramFallback:
        def send(self, *_args, **_kwargs) -> None:
            return None

        def send_document(self, *_args, **_kwargs) -> None:
            return None

    telegram_client = _TelegramFallback()

from utils.LogManager.LogManager import LogManager


lm = LogManager("main")


def send_log() -> None:
    """
    Send available log files through the configured Telegram client.

    :return: None.
    """
    path_log = f".{os.sep}..{os.sep}logCBPlusPlus.txt"
    path_main_log = f".{os.sep}utils{os.sep}LogManager{os.sep}log{os.sep}log_main.txt"

    if os.path.exists(path_main_log):
        try:
            telegram_client.send_document(path_main_log)
        except Exception:
            lm.printl("Error. Impossible sending log_main.txt.")
            # lm.printl(error)
    if os.path.exists(path_log):
        try:
            telegram_client.send_document(path_log)
        except Exception:
            lm.printl("Error. Impossible sending logCBPlusPlus.txt.")
            # lm.printl(error)


def redefine_exception() -> None:
    """
    Install the framework exception hook.

    :return: None.
    """
    # Set your custom excepthook
    sys.excepthook = custom_excepthook


def custom_excepthook(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    traceback: TracebackType | None,
) -> None:
    """
    Log uncaught exceptions through the framework log manager.

    :param exc_type: [Type[BaseException]] Exception type.
    :param exc_value: [BaseException] Exception instance.
    :param traceback: [TracebackType | None] Exception traceback.
    :return: None.
    """
    # lm.printl("Custom excepthook invoked:")
    # lm.printl(f"Exception Type: {exc_type}")
    lm.printl(f"Exception Value: {exc_value}")
    # Add your custom handling logic here
