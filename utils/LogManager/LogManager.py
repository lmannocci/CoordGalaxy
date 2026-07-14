from __future__ import annotations

import os
from datetime import datetime
from typing import Any

try:
    from telegram_send_message import telegram_send as t
except ModuleNotFoundError:
    class _TelegramFallback:
        def send(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def send_document(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    t = _TelegramFallback()


absolute_path = os.path.dirname(__file__)
config = os.path.join(absolute_path, f"config{os.sep}")
log = os.path.join(absolute_path, f"log{os.sep}")


class LogManager:
    def __init__(self, username: str) -> None:
        """
        Create a log manager for a named log file.

        :param username: [str] Log namespace, for example main.
        :return: None.
        """
        self.persistent_log = log + "log_" + username + ".txt"
        self.temp = log + "temp_" + username + ".txt"
        #
        # if os.path.exists(self.temp):
        #     os.remove(self.temp)
        # # create file
        # open(self.temp, "x")

        self.username = username

        if not os.path.exists(log):
            os.makedirs(log, exist_ok=True)

        # if persistent log does not exist, create it
        if not os.path.exists(self.persistent_log):
            open(self.persistent_log, "x").close()

    def printl(self, s: Any, verbose: int = 0) -> None:
        """
        Print a message, append it to the persistent log, and optionally send it through Telegram.

        :param s: [Any] Message or object to log.
        :param verbose: [int] Reserved verbosity flag.
        :return: None.
        """
        sn = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]: ") + str(s) + "\n"
        # with open(self.temp, 'a') as f:
        #     f.write(sn)
        with open(self.persistent_log, "a") as f:
            f.write(sn)
        print(s)
        try:
            t.send(s)
        except Exception as e:
            print(f"ERROR: Impossible sending message on telegram. {e}.")

    def printK(self, index: int, K: int, s: Any) -> None:
        """
        Log a message every K iterations.

        :param index: [int] Current iteration index.
        :param K: [int] Logging frequency.
        :param s: [Any] Message or object to log.
        :return: None.
        """
        if index % K == 0:
            self.printl(s)

    def printTemp(self, s: str) -> None:
        """
        Append a string to the temporary log file when one is configured.

        :param s: [str] Message to append.
        :return: None.
        """
        if isinstance(s, str):
            sn = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]: ") + s + "\n"
            with open(self.temp, "a") as f:
                f.write(sn)
        else:
            print("ERROR: you must specify a str parameter")
