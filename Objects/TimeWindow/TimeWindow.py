import re
from datetime import datetime, timedelta

from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.Checkpoint.Checkpoint import *
from typing import Any

file_name = os.path.splitext(os.path.basename(__file__))[0]

class TimeWindow:
    def __init__(
        self,
        type_output_network: str,
        type_time_window: str,
        tw_str: str,
        tw_slide_interval_str: str,
        type_merge: str | None = None
    ) -> None:
        """
            TimeWindow constructor.
            :param type_output_network: [str] The type of network in output. "temporal": in output one network for each computed time window.
             "merged": in output a unique merged network.
            Admissible values for parameter:
            - merged
            - temporal
            :param type_time_window: [str] The type of time window can be: ATW (Adjacent Time Window), OTW (Overlapping Time Window),
            ANY (no time window). The ATW exploits only tw_str, since tw_slide_interval_str is equal to tw_str.
            :param tw_str: [str] Length of the window, e.g., 1d, 1h, 30s.
            :param tw_slide_interval_str: [str] Size of the slide of the window. How much the window scrolls each time.
            :param type_merge: [str | None] Merge strategy used when the output network is merged.
            :return: None.
        """
        self.lm = LogManager("main")
        self.ch = Checkpoint()
        self.type_output_network = type_output_network
        self.type_time_window = type_time_window
        self.tw_str = tw_str
        self.tw_slide_interval_str = tw_slide_interval_str
        self.type_merge = type_merge

        self.icm = IntegrityConstraintManager(file_name)

        self.icm.check_type_output(type_output_network, type_merge)

        if self.type_time_window == "ANY":
            self.tw_str = 'none'
            self.tw_slide_interval_str = 'none'
        elif self.type_time_window == "OTW":
            pass
        elif self.type_time_window == "ATW":
            self.tw_slide_interval_str = tw_str

        self.tw = self.get_time_window(self.tw_str)
        self.tw_slide_interval = self.get_time_window(self.tw_slide_interval_str)


    def _compute_windows(self, min_time: Any, max_time: Any, df: Any, path: str) -> list[Any]:
        """
            Compute the lists of the window according to the parameter for the window and the dates of the dataframe.
            :param min_time: [datetime] Oldest date in the dataframe.
            :param max_time: [datetime] Most recent date in the dataframe.
            :param df: [pd.DataFrame] Co-action dataframe to split into time windows.
            :param path: [str] Directory where the cached window list is saved.
            :return: [list] Time-window tuples containing start/end dates and the filtered dataframe.
        """
        window_list = []

        start_date_list = []
        end_date_list = []
        # first time window start=min / end = start + tw
        temp_start_time = min_time
        temp_end_time = temp_start_time + timedelta(seconds=self.tw)

        if self.type_time_window == "ATW" or self.type_time_window == "OTW":
            max_exceeded = False

            while max_exceeded == False:
                # I use an if, because I want to do the last iteration even
                # if temp_end_time is greater than max_time, to consider the last window
                if temp_end_time >= max_time:
                    max_exceeded = True
                temp_start_time_str = temp_start_time.strftime('%Y-%m-%d %H:%M:%S')
                temp_end_time_str = temp_end_time.strftime('%Y-%m-%d %H:%M:%S')
                filtered_df = df[(df['created'] >= temp_start_time_str) & (df['created'] <= temp_end_time_str)]

                temp_window_tuple = (temp_start_time, temp_end_time, temp_start_time_str, temp_end_time_str, filtered_df)
                window_list.append(temp_window_tuple)
                # save the current window
                start_date_list.append(temp_start_time)
                end_date_list.append(temp_end_time)
                # compute next window. start = old_start+tw_slide_interval, end=start+tw
                temp_start_time = temp_start_time + timedelta(seconds=self.tw_slide_interval)
                temp_end_time = temp_start_time + timedelta(seconds=self.tw)
        elif self.type_time_window == "ANY":
            start_time_str = min_time.strftime('%Y-%m-%d %H:%M:%S')
            end_time_str = max_time.strftime('%Y-%m-%d %H:%M:%S')
            temp_tuple = (min_time, max_time, start_time_str, end_time_str, df)
            window_list.append(temp_tuple)

        self.ch.save_object(window_list, path + "window_list.p")
        # self.ch.save_object(start_date_list, path + "start_date_list.p")
        # self.ch.save_object(end_date_list, path + "end_date_list.p")
        window_sizes = [window[4].shape[0] for window in window_list]
        self.lm.printl(
            "[SIM][WINDOW BUILD DONE] "
            f"total_windows={len(window_list)} non_empty_windows={sum(size > 0 for size in window_sizes)} "
            f"min_rows={min(window_sizes) if window_sizes else 0} "
            f"max_rows={max(window_sizes) if window_sizes else 0} "
            f"total_window_rows={sum(window_sizes)} cache={path}window_list.p"
        )
        return window_list

    # def __from_datetime_to_str(self, start_date_list, end_date_list):
    #     start_date_str_list = []
    #     end_date_str_list = []
    #     for start_date, end_date in zip(start_date_list, end_date_list):
    #         start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    #         end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
    #         start_date_str_list.append(start_date_str)
    #         end_date_str_list.append(end_date_str)
    #     return start_date_str_list, end_date_str_list

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    def get_time_window(self, TW: str) -> int:
        """
            Convert a time-window string to seconds.
            :param TW: [str] Time-window value with unit, for example 1d, 1h, 30m, or 10s.
            :return: [int] Time-window length in seconds.
        """
        array = re.findall(r'[A-Za-z]+|\d+', TW)
        value = array[0]
        unit = array[1]
        if unit == 'd':
            tw = int(value) * 60 * 60 * 24
        elif unit == 'h':
            tw = int(value) * 60 * 60
        elif unit == 'm':
            tw = int(value) * 60
        elif unit == 's':
            tw = int(value)
        return tw

    def get_type_output_network(self) -> str:
        """
            Return the output network type.
            :return: [str] Output network type, for example temporal or merged.
        """
        return self.type_output_network

    def get_type_time_window(self) -> str:
        """
            Return the configured time-window type.
            :return: [str] Time-window type, for example ATW, OTW, or ANY.
        """
        return self.type_time_window

    def get_tw_str(self) -> str:
        """
            Return the original time-window string.
            :return: [str] Time-window string, for example 1d or 1h.
        """
        return self.tw_str

    def get_tw(self) -> int:
        """
            Return the time-window length in seconds.
            :return: [int] Time-window length in seconds.
        """
        return self.tw

    def get_tw_slide_interval_str(self) -> str:
        """
            Return the original time-window slide interval string.
            :return: [str] Slide interval string, for example 1d or 30m.
        """
        return self.tw_slide_interval_str

    def get_tw_slide_interval(self) -> int:
        """
            Return the time-window slide interval in seconds.
            :return: [int] Slide interval in seconds.
        """
        return self.tw_slide_interval

    def get_type_merge(self) -> str | None:
        """
            Return the merge strategy for merged output networks.
            :return: [str | None] Merge strategy, or None when no merge strategy is configured.
        """
        return self.type_merge

    def compute_time_windows(self, df: Any, path: str) -> list[Any]:
        """
            Split a co-action dataframe into the configured analysis time windows.
            :param df: [pd.DataFrame] Co-action dataframe containing a created column formatted as '%Y-%m-%d %H:%M:%S'.
            :param path: [str] Directory where the cached window list is saved.
            :return: [list] Time-window tuples containing start/end dates and the filtered dataframe.
        """
        min_time = datetime.strptime(df['created'].min(), '%Y-%m-%d %H:%M:%S')
        max_time = datetime.strptime(df['created'].max(), '%Y-%m-%d %H:%M:%S')
        self.lm.printl(
            "[SIM][WINDOW BUILD START] "
            f"type={self.type_time_window} rows={df.shape[0]} "
            f"range=[{min_time} -> {max_time}] tw={self.tw_str}({self.tw}s) "
            f"slide={self.tw_slide_interval_str}({self.tw_slide_interval}s) cache={path}window_list.p"
        )

        temp_start_time = min_time
        temp_end_time = temp_start_time + timedelta(seconds=self.tw)
        # if the right end of the range exceeds the maximum date value in the dataframe, raise error
        if temp_end_time > max_time:
            m = (
                "[SIM][WINDOW BUILD ERROR] "
                f"tw={self.tw}s is longer than the analysis period range=[{min_time} -> {max_time}]"
            )
            self.lm.printl(m)
            raise Exception(m)

        # compute the interval of each time window (between min_time and max_time)
        window_list = self._compute_windows(min_time, max_time, df, path)

        # return start_date_list, end_date_list, start_date_str_list, end_date_str_list
        return window_list
