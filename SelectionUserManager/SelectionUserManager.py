from IntegrityConstraintManager.IntegrityConstraintManager import *
from DirectoryManager import DirectoryManager
from utils.Checkpoint.Checkpoint import *
from utils.PlotManager.PlotManager import *
from utils.decorator_definition import *
from utils.common_variables import normalize_co_action_id

from itertools import combinations
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import math

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
data_path = os.path.join(absolute_path, f".{os.sep}..{os.sep}data{os.sep}")
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


def _overlapping_coefficient(set1: set, set2: set) -> tuple[set, int, float]:
    """
    Compute the overlap coefficient between two sets.

    :param set1: First set of users.
    :param set2: Second set of users.
    :return: Intersection, absolute intersection size, and overlap coefficient.
    """
    intersection = set1 & set2
    absolute_overlap = len(intersection)
    min_cardinality = min(len(set1), len(set2))
    if min_cardinality == 0:
        return intersection, absolute_overlap, 0
    return intersection, absolute_overlap, absolute_overlap / min_cardinality


class SelectionUserManager:
    def __init__(self, dataset_name: str, user_fraction: float | None, type_filter: str, co_action_list: list[str]) -> None:
        """
        Create a manager for selecting users from co-action datasets.

        :param dataset_name: Name of the dataset to process.
        :param user_fraction: Fraction of users to keep, or None when no fraction is used.
        :param type_filter: User-selection strategy name.
        :param co_action_list: List of co-action layer names to process.
        :return: None.
        """
        self.lm = LogManager("main")
        self.icm = IntegrityConstraintManager(file_name)

        self.ch = Checkpoint()
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.dataset_name = dataset_name
        self.co_action_list = [normalize_co_action_id(co_action) for co_action in co_action_list]

        self.icm.check_type_filter(type_filter)
        self.icm.check_user_fraction(user_fraction)
        self.icm.check_list_co_action(co_action_list)
        self.dm = DirectoryManager(file_name, dataset_name, data_path=data_path, results=results, user_fraction=user_fraction, type_filter=type_filter)

        self.pm = PlotManager()

    def _get_distinct_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the most active distinct users according to the configured fraction.

        :param df: Co-action dataframe containing userId values.
        :return: Dataframe with a single userId column for selected users.
        """
        number_distinct_users = len(df['userId'].unique())
        if self.user_fraction is None:
            filter_len = number_distinct_users
        else:
            filter_len = int(number_distinct_users * self.user_fraction)
        # Count the number of posts for each user. Sort the users according to this count. Select the first filter_len users (most active users)
        # Select only the userId column, useful to perform the inner join with the original dataset, selecting posts of the most active users
        top_users = df.groupby("userId").size().reset_index(name="count").sort_values(by=["count"], ascending=False)[0:filter_len][["userId"]]

        return top_users

    def _plot_distribution(self, c: str, df: pd.DataFrame) -> None:
        """
        Plot the distribution of co-action objects per user.

        :param c: Column name containing co-action object values.
        :param df: Co-action dataframe to summarize.
        :return: None. The plot is saved to the analysis directory.
        """
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

    def _save_filtered_users(
        self,
        key: str,
        df: pd.DataFrame,
        top_users: pd.DataFrame,
        save_dataset: bool,
        source_suffix: str = ""
    ) -> pd.DataFrame:
        """
        Filter a co-action dataframe to selected users and optionally save it.

        :param key: Co-action key being filtered.
        :param df: Original co-action dataframe.
        :param top_users: Dataframe containing selected userId values.
        :param save_dataset: Whether to save the filtered dataframe under the selected co-action directory.
        :param source_suffix: Optional suffix used by the source co-action artifact.
        :return: Filtered co-action dataframe.
        """
        filename = self._filtered_co_action_filename(key, extension="csv")
        selected_user_ids = set(top_users["userId"].astype(str))
        selected_row_mask = df["userId"].astype(str).isin(selected_user_ids)
        filter_df = df.loc[selected_row_mask].copy()

        # we can compute the dataframe, without saving it. It can be useful, if we want to save only the statistics
        # in preliminary experiments
        if save_dataset == True and self.user_fraction is not None:
            output_path = self._selected_co_action_data_path()
            self.ch.save_dataframe(filter_df, output_path + filename)
            self._save_filtered_embeddings(key, selected_row_mask, source_suffix)

        return filter_df

    def _save_filtered_embeddings(self, key: str, selected_row_mask: pd.Series, source_suffix: str = "") -> None:
        """
        Filter and save embeddings for text-based co-actions using the selected CSV row mask.

        :param key: Co-action key being filtered.
        :param selected_row_mask: Boolean mask used to filter the corresponding co-action CSV.
        :param source_suffix: Optional suffix used by the source embedding artifact.
        :return: None. The filtered embedding artifact is saved under the selected co-action directory.
        """
        if key not in co_action_embeddings:
            return

        source_path = self._co_action_file_path(action_map[key], source_suffix, extension="npy")
        output_filename = self._filtered_co_action_filename(key, extension="npy")
        embeddings = self.ch.load_object(source_path)
        selected_positions = np.flatnonzero(selected_row_mask.to_numpy())
        filtered_embeddings = embeddings[selected_positions]
        self.ch.save_object(filtered_embeddings, self._selected_co_action_data_path() + output_filename)


    def _save_info_overlapping(self, users_df_dict: dict[str, pd.DataFrame], list_name_co_action: list[str]) -> None:
        """
        Save pairwise overlap statistics among selected user sets.

        :param users_df_dict: Mapping from co-action key to selected users dataframe.
        :param list_name_co_action: Ordered list of co-action keys to compare.
        :return: None. The overlap summary is appended to the analysis file.
        """
        if len(list_name_co_action) < 2:
            self.lm.printl(
                f"{file_name}. Skipping selected-user overlap statistics: "
                "at least two co-actions are required."
            )
            return

        data_dict = {"userFraction": [], "coAction1": [], "coAction2": [], "overlapping": [], "percOverlapping": []}

        for c1, c2 in combinations(list_name_co_action, 2):
            user_set1 = set(users_df_dict[c1]['userId'].unique())
            user_set2 = set(users_df_dict[c2]['userId'].unique())

            _, absolute_o, o_coefficient = _overlapping_coefficient(user_set1, user_set2)
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

    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    def _save_info_dataset(self, key: str, original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
        """
        Save filtering statistics for one co-action dataset.

        :param key: Co-action key being summarized.
        :param original_df: Co-action dataframe before user filtering.
        :param filtered_df: Co-action dataframe after user filtering.
        :return: None. The summary is appended to the analysis file.
        """
        c = co_action_column[key]
        # self._plot_distribution(c, filtered_df)

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

    @log_method
    def analyze_user_selection(self, filter_dataset: dict[str, bool]) -> dict[str, pd.DataFrame]:
        """
        Analyze user selection without saving filtered co-action datasets.

        :param filter_dataset: Mapping from co-action key to whether an already filtered input file should be read.
        :return: Dictionary mapping co-action keys to the filtered dataframes computed for analysis.
        """
        return self._run_user_selection(filter_dataset, save_dataset=False, save_info=True)

    @log_method
    def apply_user_selection(self, filter_dataset: dict[str, bool], save_info: bool = False) -> dict[str, pd.DataFrame]:
        """
        Select users and save filtered co-action datasets.

        :param filter_dataset: Mapping from co-action key to whether an already filtered input file should be read.
        :param save_info: Whether to save analysis summaries while filtering.
        :return: Dictionary mapping co-action keys to filtered dataframes.
        """
        return self._run_user_selection(filter_dataset, save_dataset=True, save_info=save_info)

    @log_method
    def filter_users(
        self,
        filter_dataset: dict[str, bool],
        save_dataset: bool = True,
        save_info: bool = False
    ) -> dict[str, pd.DataFrame]:
        """
        Backward-compatible user-selection entry point.

        :param filter_dataset: Mapping from co-action key to whether an already filtered input file should be read.
        :param save_dataset: Whether to save filtered co-action datasets.
        :param save_info: Whether to save analysis summaries.
        :return: Dictionary mapping co-action keys to filtered dataframes.
        """
        return self._run_user_selection(filter_dataset, save_dataset=save_dataset, save_info=save_info)

    def _load_co_action_datasets(self, filter_dataset: dict[str, bool]) -> dict[str, pd.DataFrame]:
        """
        Load the co-action datasets configured for this selection manager.

        :param filter_dataset: Mapping from co-action key to whether an already filtered input file should be read.
        :return: Dictionary mapping co-action keys to loaded co-action dataframes.
        """
        dict_df = {}
        for co_action in self.co_action_list:
            self.lm.printl(f"{file_name}: Reading dataframe for co-action {co_action}.")
            suffix = "_filtered" if self._filter_dataset_value(filter_dataset, co_action) == True else ""
            path_df = self._co_action_file_path(action_map[co_action], suffix)
            dict_df[co_action] = self.ch.read_dataframe(path_df, dtype=dtype)

        return dict_df

    def _keep_original_content_only(self, dict_df: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Restrict URL, mention, and hashtag co-actions to original content rows.

        :param dict_df: Dictionary mapping co-action keys to co-action dataframes.
        :return: Dictionary with original-only filtering applied where relevant.
        """
        # with this filter, only original tweets are considered for co-actions url, mention and hashtag
        if self.type_filter == 'top_co_action_original' or self.type_filter == 'top_co_action_merge_original':
            for co_action in self.co_action_list:
                if co_action in ['co-url-domain', 'co-mention', 'co-hashtag']:
                    dict_df[co_action] = dict_df[co_action][dict_df[co_action]['contentType'] == 'original']
        return dict_df

    def _read_normalized_posts(self) -> pd.DataFrame:
        """
        Read the normalized post dataframe used by activity-based user-selection strategies.

        :return: Normalized post dataframe.
        """
        path_df_all = f"{self.dm.path_temp_data}{self.dataset_name}_normalized_tweets.csv"
        if not os.path.exists(path_df_all):
            path_df_all = f"{self.dm.path_temp_data}2_{self.dataset_name}_normalized_tweets.csv"
        if not os.path.exists(path_df_all):
            path_df_all = f"{self.dm.path_temp_data}2_{self.dataset_name}_normalized_info_tweets.csv"
        return self.ch.read_dataframe(path_df_all, dtype=dtype)

    def _run_user_selection(
        self,
        filter_dataset: dict[str, bool],
        save_dataset: bool,
        save_info: bool
    ) -> dict[str, pd.DataFrame]:
        """
        Run the configured user-selection strategy and optionally save outputs.

        :param filter_dataset: Mapping from co-action key to whether an already filtered input file should be read.
        :param save_dataset: Whether to save filtered co-action datasets.
        :param save_info: Whether to save analysis summaries.
        :return: Dictionary mapping co-action keys to filtered dataframes.
        """
        dict_df = self._keep_original_content_only(self._load_co_action_datasets(filter_dataset))
        filtered_df_dict = {}
        source_suffix_by_key = {
            co_action: "_filtered" if self._filter_dataset_value(filter_dataset, co_action) == True else ""
            for co_action in self.co_action_list
        }

        if self.type_filter in ["top_co_action", "top_co_action_original"]:
            top_users_dict = {}  # list of dataframes, containing only userId column
            for key, df in dict_df.items():
                top_users = self._get_distinct_users(df)
                top_users_dict[key] = top_users
                final_df = self._save_filtered_users(key, df, top_users, save_dataset, source_suffix_by_key[key])
                filtered_df_dict[key] = final_df
                if save_info == True:
                    self._save_info_dataset(key, df, final_df)

            if save_info == True:
                self._save_info_overlapping(top_users_dict, list(dict_df.keys()))

        if self.type_filter in ["top_co_action_merge", "top_co_action_merge_original"]:
            # for each co-action, I select the top users, then I merge these lists, to create a unique list of users, which
            # is used as initial set of users for all co-actions. Ths method allows to select more likely a set of overlapping
            # users between the different co-actions. Indeed, "top_co_action" uses a different set for each co-action,
            # which probably brings to a low overlapping between the set of users of the different co-actions
            top_users_dict = {}  # list of dataframes, containing only userId column
            for key, df in dict_df.items():
                top_users = self._get_distinct_users(df)
                top_users_dict[key] = top_users

            top_users_list = list(top_users_dict.values())
            if save_info == True:
                self._save_info_overlapping(top_users_dict, list(dict_df.keys()))

            # Concat dataframes userIds, in a unique df
            merged_top_user_df = pd.concat(top_users_list, ignore_index=True)
            # Remove duplicates based on 'userId'
            merged_top_user_df = merged_top_user_df.drop_duplicates(subset=['userId'])
            # Optionally, you can reset the index
            merged_top_user_df = merged_top_user_df.reset_index(drop=True)

            for key, df in dict_df.items():
                filter_df = self._save_filtered_users(key, df, merged_top_user_df, save_dataset, source_suffix_by_key[key])
                filtered_df_dict[key] = filter_df
                if save_info == True:
                    self._save_info_dataset(key, df, filter_df)

        if self.type_filter in ['most_active_users', 'top_tweeters', 'top_retweeters']:
            df_all = self._read_normalized_posts()
            if self.type_filter == "most_active_users":
                top_users = self._get_distinct_users(df_all)
            if self.type_filter == "top_tweeters":
                df_tweeters = df_all.loc[df_all['contentType'] == 'original']
                top_users = self._get_distinct_users(df_tweeters)
            if self.type_filter == "top_retweeters":
                top_users = self._get_distinct_users(dict_df["co-retweet"])

            for key, df in dict_df.items():
                final_df = self._save_filtered_users(key, df, top_users, save_dataset, source_suffix_by_key[key])
                filtered_df_dict[key] = final_df
                if save_info == True:
                    self._save_info_dataset(key, df, final_df)

        return filtered_df_dict

    def _filtered_co_action_filename(self, key: str, extension: str = "csv") -> str:
        """
            Return the filtered co-action artifact filename.
            :param key: [str] Co-action key.
            :param extension: [str] File extension without dot.
            :return: [str] Filtered co-action artifact filename.
        """
        return f"{action_map[key]}.{extension}"

    def _filter_dataset_value(self, filter_dataset: dict[str, bool], co_action: str) -> bool:
        """
            Return the input-filter flag for a co-action using canonical ids and aliases.
            :param filter_dataset: [dict[str, bool]] Dictionary keyed by co-action id, layer name, or alias.
            :param co_action: [str] Canonical co-action id being processed.
            :return: [bool] Whether the filtered source artifact should be read.
        """
        normalized_filter_dataset = {
            normalize_co_action_id(key): value
            for key, value in filter_dataset.items()
        }
        return normalized_filter_dataset.get(co_action, False)

    def _selected_co_action_data_path(self) -> str:
        """
            Return the directory where selected-user co-action artifacts are saved.
            :return: [str] Path to the selected co-action data directory.
        """
        return self.dm.get_co_action_data_path(self.user_fraction, self.type_filter, create=True)

    def _co_action_file_path(self, action_name: str, suffix: str = "", extension: str = "csv") -> str:
        """
            Return the co-action file path, preferring the dataset-directory scoped filename template.
            :param action_name: [str] Co-action filename stem.
            :param suffix: [str] Optional suffix before the extension.
            :param extension: [str] File extension without dot.
            :return: [str] Path to the co-action artifact.
        """
        filename = f"{action_name}{suffix}.{extension}"
        path = f"{self.dm.path_co_action_data}{filename}"
        if os.path.exists(path):
            return path

        legacy_filename = f"{self.dataset_name}_{action_name}{suffix}.{extension}"
        return f"{self.dm.path_co_action_data}{legacy_filename}"

    @log_method
    def plot_overlapping_percentage_users(self) -> None:
        """
        Plot overlap percentages for users selected across co-action layers.

        :return: None. The plot is saved in the analysis directory.
        """
        if len(self.co_action_list) < 2:
            self.lm.printl(
                f"{file_name}. Skipping selected-user overlap plot: "
                "at least two co-actions are required."
            )
            return

        df = self.ch.read_dataframe(f"{self.dm.path_data_analysis}top_co_action_merge_info_overlapping_users.csv",
                                    dtype=dtype)
        if df.empty:
            self.lm.printl(f"{file_name}. Skipping selected-user overlap plot: no overlap rows found.")
            return

        self.pm.plot_grid_combinations(df, self.dm.path_data_analysis,  "top_co_action_merge_info_overlapping_users.png",
                                       "coAction1", "coAction2", 'userFraction',
                                       'percOverlapping', 'userFraction', 'percOverlapping',
                                       0.01)

    @log_method
    def plot_number_users(self) -> None:
        """
        Plot original and filtered user counts for each co-action layer.

        :return: None. The plot is saved in the analysis directory.
        """
        df = self.ch.read_dataframe(f"{self.dm.path_data_analysis}top_co_action_merge_info_filter_users.csv", dtype=dtype)

        # Get unique co_actions
        co_actions = df['co_action'].unique()
        num_plots = len(co_actions)
        if num_plots == 0:
            self.lm.printl(f"{file_name}. Skipping selected-user count plot: no rows found.")
            return

        # Calculate number of columns and rows
        num_cols = 2  # Initial assumption of number of columns
        num_rows = math.ceil(num_plots / num_cols)  # Calculate number of rows needed

        # Create the grid of subplots
        fig, axes = plt.subplots(nrows=max(num_rows, 1), ncols=num_cols, figsize=(15, 10), sharey='col')
        axes = np.atleast_1d(axes).flatten()

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
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{self.dm.path_data_analysis}top_co_action_merge_number_users.png", dpi=dpi)
