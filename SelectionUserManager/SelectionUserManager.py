from IntegrityConstraintManager.IntegrityConstraintManager import *
from DirectoryManager import DirectoryManager
from utils.Checkpoint.Checkpoint import *
from utils.PlotManager.PlotManager import *
from SimilarityFunctionManager.methods.similarityFunction import *

from itertools import combinations
import matplotlib.pyplot as plt
import json
import numpy as np
import math

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
data_path = os.path.join(absolute_path, f".{os.sep}..{os.sep}data{os.sep}")
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class SelectionUserManager:
    def __init__(self, dataset_name, user_fraction, type_filter, co_action_list):
        self.lm = LogManager("main")
        self.icm = IntegrityConstraintManager(file_name)

        self.ch = Checkpoint()
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.dataset_name = dataset_name
        self.co_action_list = co_action_list

        self.icm.check_type_filter(type_filter)
        self.icm.check_user_fraction(user_fraction)
        self.icm.check_list_co_action(co_action_list)
        self.dm = DirectoryManager(file_name, dataset_name, data_path=data_path, results=results, user_fraction=user_fraction, type_filter=type_filter)

        self.pm = PlotManager()

    def __get_distinct_users(self, df):
        number_distinct_users = len(df['userId'].unique())
        filter_len = int(number_distinct_users * self.user_fraction)
        # Count the number of posts for each user. Sort the users according to this count. Select the first filter_len users (most active users)
        # Select only the userId column, useful to perform the inner join with the original dataset, selecting posts of the most active users
        top_users = df.groupby("userId").size().reset_index(name="count").sort_values(by=["count"], ascending=False)[0:filter_len][["userId"]]

        return top_users

    def __plot_distribution(self, c, df):
        distribution_url = df.groupby('userId')[c].count().reset_index().rename(columns={0: 'count'})
        # Plotting the distribution
        plt.figure()
        plt.hist(distribution_url[c], bins=40, edgecolor='black')
        plt.title(f"Distribution of {c}")
        plt.xlabel(f"Number {c} per user")
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f"{self.dm.path_data_analysis}{self.user_fraction}_{self.type_filter}_{c}_distribution.png", dpi=dpi)
        plt.show()

    def __save_filtered_users(self, key, df, top_users, save_dataset):
        self.lm.printl(f"{file_name}: __save_filtered_users start.")

        filename = f"{self.user_fraction}_{self.type_filter}_{self.dataset_name}_{key}.csv"
        filter_df = pd.merge(df, top_users, on="userId", how="inner")

        # we can compute the dataframe, without saving it. It can be useful, if we want to save only the statistics
        # in preliminary experiments
        if save_dataset == True:
            self.ch.save_dataframe(filter_df, self.dm.path_dataset + filename)

        self.lm.printl(f"{file_name}: __save_filtered_users completed.")
        return filter_df


    def __save_info_overlapping(self, users_df_dict, list_name_co_action):
        self.lm.printl(f"{file_name}: __save_info_overlapping start.")
        data_dict = {"userFraction": [], "coAction1": [], "coAction2": [], "overlapping": [], "percOverlapping": []}

        for c1, c2 in combinations(list_name_co_action, 2):
            user_set1 = set(users_df_dict[c1]['userId'].unique())
            user_set2 = set(users_df_dict[c2]['userId'].unique())

            _, absolute_o, o_coefficient = overlapping_coefficient(user_set1, user_set2)
            o_perc = round(o_coefficient * 100)
            c1_name = co_action_map[c1]
            c2_name = co_action_map[c2]

            data_dict["userFraction"].append(self.user_fraction)
            data_dict["coAction1"].append(c1_name)
            data_dict["coAction2"].append(c2_name)
            data_dict["overlapping"].append(absolute_o)
            data_dict["percOverlapping"].append(o_perc)
        df = pd.DataFrame(data_dict)
        self.ch.update_dataframe(df, self.dm.path_data_analysis + f"{self.type_filter}_info_overlapping_users.csv", dtype=dtype)
        self.lm.printl(f"{file_name}: __save_info_overlapping completed.")

    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    def __save_info_dataset(self, key, original_df, filtered_df):
        self.lm.printl(f"{file_name}: __save_info_dataset start.")
        c = co_action_column[key]
        # self.__plot_distribution(c, filtered_df)

        row_dict = {}
        row_dict['co_action'] = key
        row_dict['type_filter'] = self.type_filter
        row_dict['user_fraction'] = self.user_fraction

        row_dict['nElements'] = original_df.shape[0]
        row_dict['nFilteredElements'] = filtered_df.shape[0]

        row_dict['nDistinctElements'] = len(original_df[c].unique())
        row_dict['nFilteredDistinctElements'] = len(filtered_df[c].unique())

        row_dict['nUsers'] = len(original_df['userId'].unique())
        row_dict['nFilteredUsers'] = len(filtered_df['userId'].unique())

        df_info = pd.DataFrame([row_dict])
        self.ch.update_dataframe(df_info, self.dm.path_data_analysis+f"{self.type_filter}_info_filter_users.csv", dtype=dtype)
        self.lm.printl(json.dumps(row_dict, indent=4))

        self.lm.printl(f"{file_name}: __save_info_dataset completed.")


    def filter_users(self, filter_dataset, save_dataset=True, save_info=False):
        """
            Given a dataframe of posts published by several users, it gets posts by the top user_fraction of top retweeters.
            :param df: [DataFrame] DataFrame of posts.
            :param user_fraction: [double] User fraction to select users. E.g., 0.01=1% users (most active users)
            :param filename: [str] Csv file to save the pandas dataframe
            :param type_filter: [str] Type of filtering.
                Admissible values for parameter:
            - most_active_users
            - top_retweeters
            - top_tweeters
            - top_co_action_original
            - top_co_action_merge_original
            - top_co_action_
            - top_co_action_merge
            :param filter_dataset: [dict] Dictionary with key the co-action, and value True/False, whether to read filtered dataset or not.
            :param save_dataset: [bool, default=True] Whether to save the dataframe after the computation.
            Default is True, but if you are interested in only computing the statistics, without saving the dataframe,
            you can set it to False.
            :param save_info: [bool, default=False] Whether to save the information on the dataset. Default is False,
             but if you are interested in computing the plots of the overlapping between co-actions, or the statistics
             on the dataset, you can set it to True.
            :return: [DataFrame] Return a pandas DataFrame with posts of the most active users.
        """
        self.lm.printl(f"{file_name}: filter_users start.")

        dict_df = {}

        # dict_df["df_retweet"] = df_all[df_all['retweetId'].isnull() == False]
        # dict_df['df_reply'] = df_all.loc[(df_all['replyId'].isnull() == False)]
        for co_action in self.co_action_list:
            self.lm.printl(f"{file_name}: Reading dataframe for co-action {co_action}.")
            suffix = "filtered" if filter_dataset[co_action] == True else ""
            path_df = f"{self.dm.path_dataset}2_{self.dataset_name}_normalized_{action_map[co_action]}{suffix}.csv"
            dict_df[co_action] = self.ch.read_dataframe(path_df, dtype=dtype)

        # if co_action == 'co-retweet':
        #         dict_df["co-retweet"] = self.ch.read_dataframe(f"{self.dm.path_dataset}2_uk_normalized_retweet.csv", dtype=dtype)
        #     elif co_action == 'co-reply':
        #         dict_df["co-reply"] = self.ch.read_dataframe(f"{self.dm.path_dataset}2_uk_normalized_reply.csv", dtype=dtype)
        #     elif co_action == 'co-url-domain':
        #         dict_df["co-url-domain"] = self.ch.read_dataframe(f"{self.dm.path_dataset}2_uk_normalized_url{suffix}.csv", dtype=dtype)
        #     elif co_action == 'co-mention':
        #         dict_df[co_action"] = self.ch.read_dataframe(f"{self.dm.path_dataset}2_{self.dataset_name}_normalized_{action_map[co_action]}{suffix}.csv", dtype=dtype)
        #     elif co_action == 'co-hashtag':
        #         dict_df["co-hashtag"] = self.ch.read_dataframe(f"{self.dm.path_dataset}2_uk_normalized_hashtag{suffix}.csv", dtype=dtype)

        # with this filter, only original tweets are considered for co-actions url, mention and hashtag
        if self.type_filter == 'top_co_action_original' or self.type_filter == 'top_co_action_merge_original':
            for co_action in self.co_action_list:
                if co_action in ['co-url-domain', 'co-mention', 'co-hashtag']:
                    dict_df[co_action] = dict_df[co_action][dict_df[co_action]['type'] == 'original']
                # if co_action == 'co-url-domain':
                #     dict_df["co-url-domain"] = dict_df["co-url-domain"][dict_df["co-url-domain"]['type'] == 'original']
                # if co_action == 'co-mention':
                #     dict_df["co-mention"] = dict_df["co-mention"][dict_df["co-mention"]['type'] == 'original']
                # if co_action == 'co-hashtag':
                #     dict_df["co-hashtag"] = dict_df["co-hashtag"][dict_df["co-hashtag"]['type'] == 'original']

        if self.type_filter == "top_co_action":
            top_users_dict = {}  # list of dataframes, containing only userId column
            for key, df in dict_df.items():
                top_users = self.__get_distinct_users(df)
                top_users_dict[key] = top_users
                final_df = self.__save_filtered_users(key, df, top_users, save_dataset)
                if save_info == True:
                    self.__save_info_dataset(key, df, final_df)

            if save_info == True:
                self.__save_info_overlapping(top_users_dict, list(dict_df.keys()))

        if self.type_filter == "top_co_action_merge":
            # for each co-action, I select the top users, then I merge these lists, to create a unique list of users, which
            # is used as initial set of users for all co-actions. Ths method allows to select more likely a set of overlapping
            # users between the different co-actions. Indeed, "top_co_action" uses a different set for each co-action,
            # which probably brings to a low overlapping between the set of users of the different co-actions
            top_users_dict = {}  # list of dataframes, containing only userId column
            for key, df in dict_df.items():
                top_users = self.__get_distinct_users(df)
                top_users_dict[key] = top_users

            top_users_list = list(top_users_dict.values())
            if save_info == True:
                self.__save_info_overlapping(top_users_dict, list(dict_df.keys()))

            # Concat dataframes userIds, in a unique df
            merged_top_user_df = pd.concat(top_users_list, ignore_index=True)
            # Remove duplicates based on 'userId'
            merged_top_user_df = merged_top_user_df.drop_duplicates(subset=['userId'])
            # Optionally, you can reset the index
            merged_top_user_df = merged_top_user_df.reset_index(drop=True)

            for key, df in dict_df.items():
                filter_df = self.__save_filtered_users(key, df, merged_top_user_df, save_dataset)
                if save_info == True:
                    self.__save_info_dataset(key, df, filter_df)

        if self.type_filter in ['most_active_users', 'top_tweeters', 'top_retweeters']:
            df_all = self.ch.read_dataframe(f"2_{self.dataset_name}_normalized_tweets.csv", dtype=dtype)
            if self.type_filter == "most_active_users":
                top_users = self.__get_distinct_users(df_all)
            if self.type_filter == "top_tweeters":
                df_tweeters = df_all.loc[(df_all['reply'].isnull() == True) & (df_all['retweet'].isnull() == True)]
                top_users = self.__get_distinct_users(df_tweeters)
            if self.type_filter == "top_retweeters":
                top_users = self.__get_distinct_users(dict_df["co-retweet"])

            new_rows = []
            for key, df in dict_df.items():
                final_df = self.__save_filtered_users(key, df, top_users, save_dataset)
                if save_info == True:
                    self.__save_info_dataset(key, df, final_df)

        self.lm.printl(f"{file_name}: filter_users completed.")

    def plot_overlapping_percentage_users(self):
        self.lm.printl(f"{file_name}: plot_overlapping_percentage_users started")
        df = self.ch.read_dataframe(f"{self.dm.path_data_analysis}top_co_action_merge_info_overlapping_users.csv",
                                    dtype=dtype)
        self.pm.plot_grid_combinations(df, self.dm.path_data_analysis,  "top_co_action_merge_info_overlapping_users.png",
                                       "coAction1", "coAction2", 'userFraction',
                                       'percOverlapping', 'userFraction', 'percOverlapping',
                                       0.01)
        self.lm.printl(f"{file_name}: plot_overlapping_percentage_users completed.")

    def plot_number_users(self):
        self.lm.printl(f"{file_name}: plot_number_users start.")
        df = self.ch.read_dataframe(f"{self.dm.path_data_analysis}top_co_action_merge_info_filter_users.csv", dtype=dtype)

        # Get unique co_actions
        co_actions = df['co_action'].unique()
        num_plots = len(co_actions)

        # Calculate number of columns and rows
        num_cols = 2  # Initial assumption of number of columns
        num_rows = math.ceil(num_plots / num_cols)  # Calculate number of rows needed

        # Create the grid of subplots
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharey='col')

        # Flatten the axes array if it's more than one row
        if num_rows > 1:
            axes = axes.flatten()

        # Iterate over each co_action and create a subplot
        for i, co_action in enumerate(co_actions):
            subset = df[df['co_action'] == co_action]
            ax = axes[i]

            ax.plot(subset['user_fraction'], subset['nUsers'], marker='o', linestyle='-', label='nUsers')
            ax.plot(subset['user_fraction'], subset['nFilteredUsers'], marker='x', linestyle='-',
                    label='nFilteredUsers')

            ax.set_xlabel('user_fraction')
            ax.set_ylabel('nUsers')
            ax.set_title(f'co_action: {co_action}')
            ax.legend()
            ax.grid(True)

        # Hide any unused subplots
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()
        {self.dm.path_data_analysis}
        plt.savefig(f"{self.dm.path_data_analysis}top_co_action_merge_number_users.png", dpi=dpi)
        self.lm.printl(f"{file_name}: plot_number_users completed.")