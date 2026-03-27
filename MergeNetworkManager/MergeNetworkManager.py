from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.ConversionManager.ConversionManager import *
from utils.Checkpoint.Checkpoint import *

import os
# import shutil
import pandas as pd


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
data_path = os.path.join(absolute_path, f".{os.sep}..{os.sep}data{os.sep}")
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class MergeNetworkManager:
    def __init__(self, dm, dataset_name, user_fraction, type_filter, tw, ca):
        """
            :param ch: [Checkpoint] Checkpoint instance to save object.
            :param type_time_window: [str] The type of time window can be: ATW (Adjacent Time Window), OTW (Overlapping Time Window),
            ANY (no time window. The ATW exploits only tw_str, since tw_slide_interval_str is equal to tw_str.
            :param tw_str: [str] Length of the window, e.g., 1d, 1h, 30s.
            :param tw_slide_interval_str: [str] Size of the slide of the window. How much the window scrolls each time.
            :param save_info: [bool, optional, default=False] If true, in case of overlapping similarity function, save info_edge_list.
            :param df: [bool, optional] If true, compute the similarity function using the sparse implementation, memorizing
            the whole similarity matrix, before discarding zero values. Instead, if false, it computes the similarity for each couples
            of user and only if nonzero value, it is saved in memory.

        """
        self.lm = LogManager('main')
        self.dm = dm

        self.ch = Checkpoint()
        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.ca = ca

        self.cm = ConversionManager()

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    def merge_edge_list(self, path_edge_list_temporal, path_edge_list):
        """
            In __computing_time_window_similarity, we compute the edges for each time window, saving each time window in a different file. In case of "merged" option, these files
            must be joined in a unique one. In this case, the output is a unique graph, where the edges are the union of the edges of all time windows, and the weights are
            the sum of the weights. The filename in output is given by the concatenation of the start_date of the first time window and the end_date of the last time window.
        """
        self.lm.printl(f"{file_name}. merge_edge_list start.")

        edge_list_files = [pos_csv for pos_csv in os.listdir(path_edge_list_temporal) if pos_csv.endswith('.p')]

        if len(edge_list_files) == 0:
            m = f"No edge_list files to be merged."
            self.lm.printl(m)
            raise Exception(m)

        # sort the file in alphabetical order, which corresponds to the chronological order
        edge_list_files = sorted(edge_list_files)
        f1 = edge_list_files[0]
        f2 = edge_list_files[-1]

        # I take the files of the first and last time window. the name of each file is in format {start_date}_{end_date}.p
        # For the first time window, I take the first element, that is the start_date and for the last time window, I take the end_date.
        start_date = f1.split('_')[0]
        end_date = f2.split('_')[1]
        filename = start_date + '_' + end_date
        # df_list = []
        combined_df = pd.DataFrame()
        for elf in edge_list_files:
            edge_list = self.ch.load_object(path_edge_list_temporal + elf)
            # there are cases in which the edge_list is empty, since in that time window, no edges have been created
            if len(edge_list) == 0:
                continue
            # sum, because in each window, each edge appears only once, so it is already the right dataframe, which can
            # be concat with combined_df, which has the following columns: userId1, userId2, sum, nAction, count
            max_index = len(edge_list[0])
            columns = list(tuple_index.keys())[0:max_index]
            temp_df = pd.DataFrame(edge_list, columns=columns)
            # I add 1 for all edges, so that in the concat dataframe, I can simply sum the current twCount + 1
            temp_df['twCount'] = 1
            # I rename the column 'sum' in 'weightSum', so that in the concat dataframe, I can simply sum the current weightSum + sum
            temp_df = temp_df.rename(columns={'w_': 'weightSum'})

            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

            # Group by userId1 and userId2 and aggregate the data
            combined_df = combined_df.groupby(['userId1', 'userId2']).agg(
                weightSum=('weightSum', 'sum'),
                nAction=('nAction', 'sum'),
                twCount=('twCount', 'sum')
            ).reset_index()

            # combined_df = combined_df.groupby(['userId1', 'userId2'], as_index=False).agg(
            #     {'sum': 'sum', 'count': 'sum'})

        # del group_df
        del edge_list
        combined_df['weightAverage'] = combined_df['weightSum'] / combined_df['twCount']
        # for each edge in the edge_list, format (userid1, userId2, weight)

        combined_df = combined_df[['userId1', 'userId2', 'weightSum', 'weightAverage', 'nAction', 'twCount']]

        self.lm.printl(f"{file_name}. All files have been read and processed.")
        if self.tw.get_type_merge() == "sum":
            combined_df = combined_df.drop(columns=['weightAverage'])
        elif self.tw.get_type_merge() == "average":
            combined_df = combined_df.drop(columns=['weightSum'])
        self.lm.printl(f"{file_name}. columns dropped.")

        combined_df['userId1'] = combined_df['userId1'].astype(str)
        combined_df['userId2'] = combined_df['userId2'].astype(str)

        merged_edge_list = self.cm.from_df_to_edge_list(combined_df)

        # if combined_df is very large, I first save a temporary dataframe, I delete and I read it in chunk,
        # transforming in list in chunks
        # combined_df.to_csv(f"{self.dm.path_edge_list}temporary_df.csv", index=False)
        # self.lm.printl(f"{file_name}. Dataframe temporary has been saved. Shape: {str(combined_df.shape)}")
        # del combined_df
        # chunksize = 500000
        # # read DataFrame in chunks and transform into list of tuples
        # chunk_iter = pd.read_csv(f"{self.dm.path_edge_list}temporary_df.csv", chunksize=chunksize, dtype=dtype)  # Adjust this based on your data
        # # combined_df = pd.read_csv(f"{self.dm.path_edge_list}temporary_df.csv", dtype=dtype)
        # # self.lm.printl(f"{file_name}. Dataframe temporary has been saved. Shape: {str(combined_df.shape)}")
        # merged_edge_list = []
        # index = 0
        # for chunk in chunk_iter:
        #     # Transform each chunk into a list of tuples
        #     tuples = [tuple(row) for row in chunk.values]
        #     merged_edge_list.extend(tuples)

        # this must be called if I want to normalize the weights
        # merged_normalized_edge_list = self.__normalize_edge_list(merged_edge_list)
        # self.ch.save_object(merged_normalized_edge_list, self.dm.path_edge_list + filename)

        self.ch.save_object(merged_edge_list, f"{path_edge_list}{filename}")

        # finally, I can remove the temporary dataframe
        # os.remove(f"{self.dm.path_edge_list}temporary_df.csv")
        # self.lm.printl(f"{file_name}. Dataframe temporary has been removed.")

        self.lm.printl(
            f"{file_name}. merge_edge_list finish merging edge lists for window {start_date}_{end_date}. Number of edges: {str(len(merged_edge_list))}.")

    # def merge_info_edge_list(self):
    #     """
    #         In __computing_time_window_similarity, we compute the info edges for each time window, saving each time window in a different file. In case of "merged" option, these files
    #         must be joined in a unique one. In this case, the output is a unique graph, where the edges are the union of the info edges of all time windows.
    #          The filename in output is given by the concatenation of the start_date of the first time window and the end_date of the last time window.
    #          It is currently implemented the saving of info_edge_list only for overlapping measure.
    #     """
    #     self.lm.printl(f"{file_name}. __merge_info_edge_list start.")
    #     info_edge_list_files = [pos_csv for pos_csv in os.listdir(self.dm.path_info_edge_list_temporal) if
    #                             pos_csv.endswith('.csv')]
    #
    #     if len(info_edge_list_files) == 0:
    #         m = f"No info_edge_list files to be merged."
    #         self.lm.printl(m)
    #         raise Exception(m)
    #
    #     # sort the file in alphabetical order, which corresponds to the chronological order
    #     info_edge_list_files = sorted(info_edge_list_files)
    #     f1 = info_edge_list_files[0]
    #     f2 = info_edge_list_files[-1]
    #
    #     # I take the files of the first and last time window. the name of each file is in format {start_date}_{end_date}.p
    #     # For the first time window, I take the first element, that is the start_date and for the last time window, I take the end_date.
    #     start_date = f1.split('_')[0]
    #     end_date = f2.split('_')[1]
    #     filename = start_date + '_' + end_date
    #
    #     # result_df = pd.DataFrame()
    #     for index, elf in enumerate(info_edge_list_files):
    #         temp_df = self.ch.read_dataframe(self.dm.path_info_edge_list_temporal + elf, dtype=dtype)
    #         self.lm.printl(f"{file_name}. Read file {elf}.")
    #         if index == 0:
    #             temp_df.to_csv(self.dm.path_info_edge_list + filename, index=False)
    #         else:
    #             temp_df.to_csv(self.dm.path_info_edge_list + filename, index=False, mode='a', header=False)
    #         # result_df = pd.concat([result_df, temp_df], ignore_index=True)
    #
    #     # result_df.to_csv(self.dm.path_info_edge_list + filename, index=False)
    #     self.lm.printl(f"{file_name}. All files have been processed.")
    #
    #     self.lm.printl(f"{file_name}. __merge_info_edge_list finish merging info edge lists.")

    # def __normalize_edge_list(self, edge_list):
    #     # Extract the values from the tuples
    #     values = [x[2] for x in edge_list]
    #
    #     # Find the minimum and maximum values
    #     min_value = min(values)
    #     max_value = max(values)
    #
    #     # Normalize the values
    #     normalized_edge_list = []
    #     for item in edge_list:
    #         user1, user2, value = item
    #         # normalized_value = (value - min_value) / (max_value - min_value)
    #         normalized_value = value / max_value
    #         normalized_edge_list.append((user1, user2, normalized_value))
    #
    #     return normalized_edge_list


