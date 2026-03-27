import re
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.Checkpoint.Checkpoint import *

file_name = os.path.splitext(os.path.basename(__file__))[0]

class TimeWindow:
    def __init__(self, type_output_network, type_time_window, tw_str, tw_slide_interval_str, type_merge=None):
        """
            TimeWindow constructor.
            :param ch: [Checkpoint] Checkpoint instance to save object.
            :param type_output_network: [str] The type of network in output. "temporal": in output one network for each computed time window.
             "merged": in output a unique merged network.
            Admissible values for parameter:
            - merged
            - temporal
            :param type_time_window: [str] The type of time window can be: ATW (Adjacent Time Window), OTW (Overlapping Time Window),
            ANY (no time window). The ATW exploits only tw_str, since tw_slide_interval_str is equal to tw_str.
            :param tw_str: [str] Length of the window, e.g., 1d, 1h, 30s.
            :param tw_slide_interval_str: [str] Size of the slide of the window. How much the window scrolls each time.
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


    def __compute_windows(self, min_time, max_time, df, path):
        """
            Compute the lists of the window according to the parameter for the window and the dates of the dataframe.
            :param min_time: [datetime] Oldest date in the dataframe.
            :param min_time: [datetime] Most recent date in the dataframe.
            :return: [list, list] Return two list start_date_list, end_date_list. The first one containing the datetime
            of the start of each window, and the second one the end.
        """
        self.lm.printl(f"{file_name}. __compute_windows start.")

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
        self.lm.printl(f"{file_name}. __compute_windows completed. Number of time windows: {str(len(window_list))}")
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
    def get_time_window(self, TW):
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

    def get_type_output_network(self):
        return self.type_output_network

    def get_type_time_window(self):
        return self.type_time_window

    def get_tw_str(self):
        return self.tw_str

    def get_tw(self):
        return self.tw

    def get_tw_slide_interval_str(self):
        return self.tw_slide_interval_str

    def get_tw_slide_interval(self):
        return self.tw_slide_interval

    def get_type_merge(self):
        return self.type_merge

    def compute_time_windows(self, df, path):
        self.lm.printl(f"{file_name}. compute_time_windows start.")
        min_time = datetime.strptime(df['created'].min(), '%Y-%m-%d %H:%M:%S')
        max_time = datetime.strptime(df['created'].max(), '%Y-%m-%d %H:%M:%S')

        temp_start_time = min_time
        temp_end_time = temp_start_time + timedelta(seconds=self.tw)
        # if the right end of the range exceeds the maximum date value in the dataframe, raise error
        if temp_end_time > max_time:
            m = f"{file_name}. compute_similarity: {self.tw} too long for the period of analysis: {str(min_time)} - {str(max_time)}"
            self.lm.printl(m)
            raise Exception(m)

        # compute the interval of each time window (between min_time and max_time)
        window_list = self.__compute_windows(min_time, max_time, df, path)

        self.lm.printl(f"{file_name}. compute_time_windows completed.")

        # return start_date_list, end_date_list, start_date_str_list, end_date_str_list
        return window_list
