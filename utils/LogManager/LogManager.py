import time
from datetime import datetime, timedelta
import os
from telegram_send_message import telegram_send as t

absolute_path = os.path.dirname(__file__)
config = os.path.join(absolute_path, f'config{os.sep}')
log = os.path.join(absolute_path, f'log{os.sep}')

class LogManager:
    def __init__(self, username):
        self.persistent_log = log + "log_" + username + ".txt"
        # self.temp = log + "temp_" + username + ".txt"
        #
        # if os.path.exists(self.temp):
        #     os.remove(self.temp)
        # # create file
        # open(self.temp, "x")

        self.username = username

        # if persistent log does not exist, create it
        if not os.path.exists(self.persistent_log):
            open(self.persistent_log, "x")

    def printl(self, s, verbose=0):
        sn = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]: ") + str(s) + "\n"
        # with open(self.temp, 'a') as f:
        #     f.write(sn)
        with open(self.persistent_log, 'a') as f:
            f.write(sn)
        print(s)
        try:
            t.send(s)
        except Exception as e:
            print(f"ERROR: Impossible sending message on telegram. {e}.")

    def printK(self, index, K, s):
        if index % K == 0:
            self.printl(s)

    def printTemp(self, s):
        if isinstance(s, str):
            sn = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]: ") + s + "\n"
            with open(self.temp, 'a') as f:
                f.write(sn)
        else:
            print("ERROR: you must specify a str parameter")