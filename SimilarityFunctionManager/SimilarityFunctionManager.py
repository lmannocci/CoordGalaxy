from SimilarityFunctionManager.methods.similarityFunction import *
from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.ConversionManager.ConversionManager import *
from utils.Checkpoint.Checkpoint import *
from MergeNetworkManager import MergeNetworkManager
from Objects.TimeWindow.TimeWindow import *

from multiprocessing import Pool, Manager
from functools import partial
from itertools import combinations
import os
# import shutil
import multiprocessing
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
data_path = os.path.join(absolute_path, f".{os.sep}..{os.sep}data{os.sep}")
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class SimilarityFunctionManager:
    def __init__(self, dataset_name, user_fraction, type_filter, tw: TimeWindow, ca, sparse_computation=False, save_info=False,
                 parallelize_window=False, parallelize_similarity=False):
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
        self.icm = IntegrityConstraintManager(file_name)
        self.dm = DirectoryManager(file_name, dataset_name, data_path=data_path, results=results,
                                   user_fraction=user_fraction, type_filter=type_filter, tw=tw, ca=ca)

        self.ch = Checkpoint()
        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.ca = ca
        self.sparse_computation = sparse_computation
        self.save_info = save_info  # if sparse_computation = True, save_info not implemented
        self.__set_parallelize_window(parallelize_window)
        self.parallelize_similarity = parallelize_similarity

        # check if the chosen sparse_computation is implemented for the chosen similarity function
        self.icm.check_sparse_computation(ca, sparse_computation, save_info, parallelize_similarity)

        self.cm = ConversionManager()

        self.mm = MergeNetworkManager(self.dm, dataset_name, user_fraction, type_filter, tw, ca)

    def __set_parallelize_window(self, parallelize_window):
        if isinstance(parallelize_window, bool):
            self.parallelize_window = parallelize_window
            if parallelize_window:
                # Determine the number of processes to use
                self.num_processes = multiprocessing.cpu_count()
            else:
                self.num_processes = 1
        elif isinstance(parallelize_window, int):
            self.num_processes = parallelize_window
            self.parallelize_window = True
        self.lm.printl(f"Parallelization window: {str(parallelize_window)} with {str(self.num_processes)} processes.")

    def __tf_idf(self, co_action_df, c):
        def concatenate_strings(set_of_strings):
            return ' '.join(set_of_strings)

        co_action_df[c + "concat"] = co_action_df[c].apply(concatenate_strings)
        documents = co_action_df[c + "concat"].values.tolist()

        # Create a TfidfVectorizer
        vectorizer = TfidfVectorizer()

        # Get feature names (terms)
        # terms = vectorizer.get_feature_names_out()

        # Fit and transform the documents
        tfidf_matrix = vectorizer.fit_transform(documents)

        return tfidf_matrix

    # NOT PARALELL IMPLEMENTATION
    def __extract_edge_list(self, combinations_sets):
        """
            Compute the edge list for the given time window.
            :param co_action_sets: [List(tuple(str, set))] List of tuples, containing userId (first element) and list of items (second element) of actions performed
            by the userId. Examples:  (781597, {'1194046521622306817', '1194041159884181504'}), (1588291, {'1193888404343275521', '1193968350793216008'}),
            :param filename: [str] Start of the current window (string format).
            :return: [list] Return list of tuples, containing edges in format (userId1, userId2, weight)
        """
        edge_list = []
        intersection_list = []
        userId_list1 = []
        userId_list2 = []

        # combinations(co_action_sets, 2): all possible combinations of length 2 of the elements of co_action_sets
        for user_tuple1, user_tuple2 in combinations_sets:
            userId1 = user_tuple1[0]
            userId2 = user_tuple2[0]

            set1 = user_tuple1[1]
            set2 = user_tuple2[1]

            sim = 0
            nCommonAction = 0
            if self.ca.get_similarity_function() == "tfidf_cosine_similarity":
                # v1, v2 are vectors of weight of tf-idf (only  tfidf_cosine_similarity has the third element in the tuple)
                v1 = user_tuple1[2]
                v2 = user_tuple2[2]
                # cosine similarity needs this shape
                v1 = v1.reshape(1, -1)
                v2 = v2.reshape(1, -1)
                sim = my_cosine_similarity(v1, v2)  # my implementation
                intersection, nCommonAction, _ = overlapping_coefficient(set1, set2)
            elif self.ca.get_similarity_function() == "overlapping_coefficient" or self.ca.get_similarity_function() == "overlapping":
                if self.ca.get_similarity_function() == "overlapping_coefficient":
                    # set1, set2 are set of ids
                    intersection, nCommonAction, sim = overlapping_coefficient(set1, set2)  # self implemented
                elif self.ca.get_similarity_function() == "overlapping":
                    intersection, nCommonAction, sim = overlapping_coefficient(set1, set2)

            # save retweetId, replyId, url, contributing to the edge between userId1 and userId2
            # I save retweetId, userId1, userId2 in a dataframe
            # for each couple of user, i concat such dataframe in a final one, for current time window
            # currently implemented only for overlapping measure.
            # I save the information only if the similarity between the two users is greater than zero
            if sim > 0:
                intersection_list.extend(list(intersection))
                temp_list1 = [userId1] * len(intersection)
                temp_list2 = [userId2] * len(intersection)
                userId_list1.extend(temp_list1)
                userId_list2.extend(temp_list2)
                # self.lm.printl(f"{str(userId1)}, {str(userId2)}, {str(sim)}")

            # if the weight is zero, we do not add the edge at all
            if sim > 0:
                edge_list.append((userId1, userId2, sim, nCommonAction))

        # only in case sparse_computation = False
        if self.save_info == True:
            info_edge_list_df = pd.DataFrame(
                {NODE1_VAR: userId_list1, NODE2_VAR: userId_list2, 'id': intersection_list})
        else:
            info_edge_list_df = pd.DataFrame()
        return edge_list, info_edge_list_df

    def __parallelized_extract_edge_list(self, tuples):
        user_tuple1, user_tuple2 = tuples

        userId1 = user_tuple1[0]
        userId2 = user_tuple2[0]

        set1 = user_tuple1[1]
        set2 = user_tuple2[1]

        if self.ca.get_similarity_function() == "tfidf_cosine_similarity":
            # v1, v2 are vectors of weight of tf-idf
            v1 = user_tuple1[2]
            v2 = user_tuple2[2]

            # cosine similarity needs this shape
            v1 = v1.reshape(1, -1)
            v2 = v2.reshape(1, -1)
            sim = my_cosine_similarity(v1, v2)  # my implementation
            intersection, nCommonAction, _ = overlapping_coefficient(set1, set2)
        elif self.ca.get_similarity_function() == "overlapping_coefficient" or self.ca.get_similarity_function() == "overlapping":
            if self.ca.get_similarity_function() == "overlapping_coefficient":
                # v1, v2 are set of ids
                intersection, nCommonAction, sim = overlapping_coefficient(set1, set2)
            elif self.ca.get_similarity_function() == "overlapping":
                intersection, nCommonAction, _ = overlapping_coefficient(set1, set2)
                sim = nCommonAction  # in case of overlapping, nCommonAction and similarity are the same thing

        # I save the information only if the similarity between the two users is greater than zero
        if sim > 0:
            return userId1, userId2, sim, nCommonAction, list(intersection)
        else:
            return None

    def __computing_time_window_similarity(self, df):
        """
            Compute the edge list for all the time windows, launching on thread for each window.
            :param df: [Dataframe] Dataframe of posts to compute the edge list.
        """
        self.lm.printl(f"{file_name}. __computing_time_window_similarity start.")

        window_list = self.tw.compute_time_windows(df, self.dm.path_info_tw)
        self.n_windows = len(window_list)
        # delete the dataframe to free memory, once I split it according to the time window
        del df

        if not self.parallelize_window:
            # Create a shared counter using Manager
            with multiprocessing.Manager() as manager:
                shared_counter = manager.Value('i', 0)  # Shared integer initialized to 0
                for window in window_list:
                    shared_counter = self.window_edge_list(window, shared_counter)
        else:
            # Create a shared counter using Manager
            with Manager() as manager:
                shared_counter = manager.Value('i', 0)  # Shared integer initialized to 0
                with Pool(processes=self.num_processes) as pool:
                    pool.map(partial(self.window_edge_list, shared_counter=shared_counter), window_list)

        # if the type of output is merged (w.r.t. the temporal axis) I have to merge the edges among the time windows, outputting one edge_list
        if self.tw.get_type_output_network() == 'merged':
            self.mm.merge_edge_list(self.dm.path_edge_list_temporal, self.dm.path_edge_list)
            # if self.ca.get_similarity_function() == "overlapping" or self.ca.get_similarity_function() == "overlapping_coefficient":
            #      if save_info == True:
            #         # info_edge_list implemented only for overlapping measure
            #         self.merge_info_edge_list()

        self.lm.printl(f"{file_name}. __computing_time_window_similarity completed.")

    def window_edge_list(self, window, shared_counter):
        """
            Compute the edge list for the given time window.
            :param window: [dict] It is a dictionary, containing the following parameters
            - filtered_df: [Dataframe] Dataframe of posts of the current time window to compute the edge list.
            - start_date_str: [str] Start of the current window (string format).
            - end_date_str: [str] End of the current window (string format).
            - start_date: [DateTime] Start of the current window (DateTime format).
            - end_date: [DateTime] End of the current window (DateTime format).
            :param shared_counter: [int] Shared variable among process, which is used to count the number of time windows
            processed. It works also with one process.
        """
        start_date = window[2]
        end_date = window[3]
        df = window[4]

        self.lm.printl(f"""{file_name}. window_edge_list start computing edge lists for window {start_date}_{end_date}.
                       Number of {action_map[self.ca.get_co_action()]}: {str(df.shape[0])}
                       Number of users {str(df['userId'].nunique())}
                       """)

        if df.shape[0] > 0:
            # name of the file of the edge list
            # I replace : characters because it is bad read both on windows and mac
            filename = f"{start_date.replace(':', '-')}_{end_date.replace(':', '-')}.p"

            # get the column name in the dataframe (c) and the action name, which is used for pretty plot and print
            c = co_action_column[self.ca.get_co_action()]

            # extract the set of retweet for each user.
            co_action_df = df.groupby("userId")[c].apply(set).reset_index()
            userIds = co_action_df["userId"].values.tolist()

            edge_list = []
            if self.sparse_computation:
                if self.ca.get_similarity_function() == "tfidf_cosine_similarity":
                    tfidf_matrix = self.__tf_idf(co_action_df, c)
                    sim_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

                    # Get the upper triangular indices (sim_matrix is symmetric)
                    upper_tri_indices = np.triu_indices(sim_matrix.shape[0], k=1)

                    # Create a list of tuples (userId1, userId2, similarity)
                    edge_list = []
                    for i, j in zip(*upper_tri_indices):
                        sim = sim_matrix[i, j]
                        if sim > 0:
                            edge_list.append((userIds[i], userIds[j], sim, None))
            else:
                co_action_sets = []
                if self.ca.get_similarity_function() == "tfidf_cosine_similarity":
                    tfidf_matrix = self.__tf_idf(co_action_df, c)
                    # convert the sparse matrix of tf-idf vectors to an array (n_users, n_features), where the n_features is the number of
                    # different retweetIds (for example), i.e., the corresponding of a term in a document.
                    mat = tfidf_matrix.toarray()

                    # (781597, {'1194046521622306817', '1194041159884181504'}),
                    # list_user_sets = list(co_action_df.to_records(index=False))
                    list_user_sets = list(co_action_df[c].values)

                    # List of tuples:
                    # 0) userId 781597
                    # 1) set of action {'1194046521622306817', '1194041159884181504'}
                    # 2) tf-idf vector of the user array([0, 0.05])
                    co_action_sets = list(zip(userIds, list_user_sets, mat))
                elif self.ca.get_similarity_function() == "overlapping_coefficient" or self.ca.get_similarity_function() == "overlapping":
                    # example of co_action_sets. Tuple where the first element is the userId and the second one is the set of the user
                    # (781597, {'1194046521622306817', '1194041159884181504'}),
                    #  (1588291, {'1193888404343275521', '1193968350793216008'}),
                    co_action_sets = list(co_action_df.to_records(index=False))

                if not self.parallelize_similarity:
                    edge_list, info_edge_list_df = self.__extract_edge_list(combinations(co_action_sets, 2))
                else:
                    # Parallelize using multiprocessing
                    with Pool() as pool:
                        results_pool = pool.map(self.__parallelized_extract_edge_list, combinations(co_action_sets, 2))

                    edge_list = []
                    intersection_list = []
                    userId_list1 = []
                    userId_list2 = []
                    # Process the results
                    for result in results_pool:
                        if result is not None:
                            userId1, userId2, sim, nCommonAction, intersection = result
                            edge_list.append((userId1, userId2, sim, nCommonAction))

                            if self.save_info:
                                intersection_list.extend(intersection)
                                temp_list1 = [userId1] * len(intersection)
                                temp_list2 = [userId2] * len(intersection)
                                userId_list1.extend(temp_list1)
                                userId_list2.extend(temp_list2)
                    if self.save_info:
                        info_edge_list_df = pd.DataFrame(
                            {NODE1_VAR: userId_list1, NODE2_VAR: userId_list2, 'id': intersection_list})

            # save edge_list
            self.ch.save_object(edge_list, self.dm.path_edge_list_temporal + filename)

            # info_edge_list are very large files. it is better not to save if it is not necessary
            if self.save_info:
                # filename is in format startDate_endDate.p, it is a pickle format
                info_edge_list_df.to_csv(self.dm.path_info_edge_list_temporal + filename.split('.')[0] + '.csv',
                                         index=False)
        else:
            edge_list = []

        # Increment the shared counter by 1
        # with shared_counter.get_lock():  # Locking to ensure safe update
        shared_counter.value += 1
        self.lm.printl(f"""{file_name}. window_edge_list {str(shared_counter.value)}/{str(self.n_windows)}.
                       Completed {start_date}_{end_date}. Number of edges: {str(len(edge_list))}.""")

        return start_date, end_date

    def __read_co_action_dataset(self):
        """
            Get the filtered dataframe, removing missing values for the correct columns of the dataset.
            :param df: [DataFrame] Pandas dataframe to be filtered according to the co-action type.
            :return: [DataFrame] Filtered pandas dataframe, according to the co-action type.
        """
        ca_type = self.ca.get_co_action()
        df = self.ch.read_dataframe(f"{self.dm.path_dataset}{self.user_fraction}_{self.type_filter}_{self.dataset_name}_{ca_type}.csv", dtype)
        if co_action_column[ca_type] not in df.columns:
            m = f"{co_action_column[ca_type]} column is not among the columns of the dataframe."
            self.lm.printl(m)
            raise ValueError(m)

        # Remove null values
        df = df[df[co_action_column[ca_type]].isnull() == False]
        return df

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------

    def compute_similarity(self):
        """
            Compute the edge list for all the time windows, launching on thread for each window.
        """
        self.lm.printl(f"{file_name}. compute_similarity start.")
        start_time = time.time()

        df = self.__read_co_action_dataset()

        self.__computing_time_window_similarity(df)

        self.lm.printl(
            f"{file_name}. compute_similarity: {self.tw.get_type_time_window()} tw={self.tw.get_tw()}, tw_slide={self.tw.get_tw_slide_interval()}, co_action={self.ca.get_co_action()} completed.")
        finish_time = time.time()
        delta_time = finish_time - start_time
        self.lm.printl(f"{file_name}. compute_similarity completed in %s seconds" % delta_time)
