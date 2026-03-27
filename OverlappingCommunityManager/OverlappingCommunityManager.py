from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D  # For custom legends
from itertools import combinations
import matplotlib.ticker as mticker
from statannotations.Annotator import Annotator


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class OverlappingCommunityManager:
    def __init__(self, dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, file_prefix, chm_x=None,
                 chm_y=None, community_size_th=None):
        """
            Given a dataframe of pasts published by several users, it gets posts by the top user_fraction users.
            :param type_time_window: [str] The type of time window can be: ATW (Adjacent Time Window), OTW (Overlapping Time Window),
            ANY (no time window). The ATW exploits only tw_str, since tw_slide_interval_str is equal to tw_str.
            :param tw_str: [str] Length of the window, e.g., 1d, 1h, 30s.
            :param tw_slide_interval_str: [str] Size of the slide of the window. How much the window scrolls each time.
            :return: [str] Return the path of the directory according to the parameters. ATW is based only on tw_str parameter,
            while OTW both on tw_str and tw_slide_interval_str. Instead, ANY implies no window at all, so it is free parameter.
        """
        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter

        self.list_ca = list_ca
        self.available_list_ca = list(available_co_action.keys())
        self.dict_ca_filter = dict_ca_filter
        self.icm = IntegrityConstraintManager(file_name)
        # check if for each co_action in the list, it is passed the corresponding threshold in the dictionary of the threshold.
        self.icm.check_co_action(list_ca, dict_ca_filter)

        self.file_prefix = file_prefix
        self.community_size_th = community_size_th

        self.lm = LogManager('main')
        self.ch = Checkpoint()
        self.cm = ConversionManager()

        # General information, common to all chm_x, chm_y. useful to extract the info for the file name
        self.dm = DirectoryManager(file_name, dataset_name, results=results, user_fraction=user_fraction,
                                   type_filter=type_filter,
                                   tw=tw, list_ca=list_ca, dict_ca_filter=dict_ca_filter)
       

        if chm_x is not None:
            self.chm_x = chm_x
            self.dm_x = self.chm_x.get_directory_manager()
            self.ch_x = self.chm_x.get_checkpoint()
            self.type_algorithm_x = self.chm_x.get_type_algorithm()
            self.list_ca_x = self.chm_x.get_list_ca()
            self.dict_ca_filter_x = self.chm_x.get_dict_ca_filter()
            self.cda_x = self.chm_x.get_cda()
            self.type_algorithm_x = self.dm_x.get_type_algorithm()

        if chm_y is not None:
            self.chm_y = chm_y
            self.dm_y = self.chm_y.get_directory_manager()
            self.ch_y = self.chm_y.get_checkpoint()
            self.type_algorithm_y = self.chm_y.get_type_algorithm()
            self.list_ca_y = self.chm_y.get_list_ca()
            self.dict_ca_filter_y = self.chm_y.get_dict_ca_filter()
            self.cda_y = self.chm_y.get_cda()
            self.type_algorithm_y = self.dm_y.get_type_algorithm()

    def __overlapping_measures(self, set_x, set_y):
        overlapping_dict = {}
        num_decimal = 3

        intersection = set_x & set_y
        union = set_x | set_y

        len_c_x = len(set_x)
        len_c_y = len(set_y)
        len_intersection = len(intersection)
        # print(len(set_x), len(set_y))
        # print(len_intersection)
        len_union = len(union)
        len_min = min(len_c_x, len_c_y)

        overlapping_dict['absolute'] = len(intersection)
        if len_c_x != 0:
            overlapping_dict['intersect_x'] = round(len(intersection) / len_c_x, num_decimal)
        else:
            overlapping_dict['intersect_x'] = 0

        if len_c_y != 0:
            overlapping_dict['intersect_y'] = round(len(intersection) / len_c_y, num_decimal)
        else:
            overlapping_dict['intersect_y'] = 0

        if len_min != 0:
            overlapping_dict['minimum'] = round(len(intersection) / len_min, num_decimal)
        else:
            overlapping_dict['minimum'] = 0

        if len_union != 0:
            overlapping_dict['jaccard'] = round(len(intersection) / len_union, num_decimal)
        else:
            overlapping_dict['jaccard'] = 0

        numerator = (2 * overlapping_dict['intersect_x'] * overlapping_dict['intersect_y'])
        denominator = (overlapping_dict['intersect_x'] + overlapping_dict['intersect_y'])

        if denominator != 0:
            overlapping_dict['harmonicMean'] = round(numerator / denominator, num_decimal)
        else:
            overlapping_dict['harmonicMean'] = 0

        return overlapping_dict, intersection

    def __get_labels_reordered(self, df_x=None, df_y=None):

        if self.type_algorithm_x == 'one-layer':
            x_label = self.list_ca_x[0].get_co_action()
        else:
            if self.cda_x.get_algorithm_name() in custom_flatten_algorithm:
                x_label = self.cda_x.get_algorithm_name()
            else:
                x_label = 'multimodal'

        if self.type_algorithm_y == 'one-layer':
            y_label = self.list_ca_y[0].get_co_action()
        else:
            if self.cda_y.get_algorithm_name() in custom_flatten_algorithm:
                y_label = self.cda_y.get_algorithm_name()
            else:
                y_label = 'multimodal'
        if df_x is not None and df_y is not None:
            if x_label == 'multimodal':
                df_x.rename(columns={'cid': 'group', 'actor': 'userId'}, inplace=True)
                df_x['layer'] = df_x['layer'].map(action_map_inverse)
                return x_label, y_label, df_x, df_y
            elif y_label == 'multimodal':  # if y is multimodal, I swap the dataframes and the labels
                df_y.rename(columns={'cid': 'group', 'actor': 'userId'}, inplace=True)
                # layer name is, e.g., retweet, reply, URL, mention, hashtag. instead of the co-action name which is
                # co-retweet, co-reply, co-url-domain, co-mention, co-hashtag. So it is necessary to map the layer name
                df_y['layer'] = df_y['layer'].map(action_map_inverse)
                return y_label, x_label, df_y, df_x
            else:
                return x_label, y_label, df_x, df_y
        else:
            if x_label == 'multimodal':
                return x_label, y_label
            elif y_label == 'multimodal':
                return y_label, x_label
            else:
                return x_label, y_label

    def __filter_community_size(self, df):
        self.lm.printl(f"{file_name}. __filter_community_size start.")
        agg_df = df.groupby(['group']).size().reset_index(name='nUsers')

        left_com_df = pd.DataFrame(agg_df[agg_df['nUsers'] >= self.community_size_th]['group'])
        filter_df = pd.merge(df, left_com_df, on='group', how='inner')
        self.lm.printl(f"{file_name}. __filter_community_size completed.")
        return filter_df

    def __extract_user_set(self, x_label, y_label, df_x, df_y):
        self.lm.printl(f"{file_name}. __extract_user_set completed.")
        group_list_x = df_x['group'].unique()
        group_list_y = df_y['group'].unique()
        if x_label == 'multimodal':
            user_set_df_x = pd.DataFrame(df_x.groupby(['group', 'layer'])['userId'].apply(set)).reset_index()

            user_set_df_x = user_set_df_x[user_set_df_x['layer'] == y_label][['group', 'userId']]

            excluded_communities_x = list(set(group_list_x) - set(user_set_df_x['group'].unique()))
            new_rows_x = pd.DataFrame(
                {'group': excluded_communities_x, 'userId': [set() for i in range(len(excluded_communities_x))]})
            user_set_df_x = pd.concat([user_set_df_x, new_rows_x], ignore_index=True)
        else:
            user_set_df_x = df_x.groupby('group')['userId'].apply(set).reset_index()

        if y_label == 'multimodal':
            user_set_df_y = pd.DataFrame(df_y.groupby(['group', 'layer'])['userId'].apply(set)).reset_index()
            user_set_df_y = user_set_df_y[user_set_df_y['layer'] == x_label][['group', 'userId']]

            excluded_communities_y = list(set(group_list_y) - set(user_set_df_y['group'].unique()))
            new_rows_y = pd.DataFrame(
                {'group': excluded_communities_y, 'userId': [set() for x in range(len(excluded_communities_y))]})
            user_set_df_y = pd.concat([user_set_df_y, new_rows_y], ignore_index=True)
        else:
            user_set_df_y = df_y.groupby('group')['userId'].apply(set).reset_index()

        user_set_df_x = user_set_df_x.sort_values(by="group")
        user_set_df_y = user_set_df_y.sort_values(by="group")

        user_set_df_x.rename(columns={'userId': 'userSet'}, inplace=True)
        user_set_df_y.rename(columns={'userId': 'userSet'}, inplace=True)

        self.lm.printl(f"{file_name}. __extract_user_set completed.")
        return user_set_df_x, user_set_df_y

    def __annotate_format(self, val):
        # Custom function to format the annotation based on whether the value is an integer or float
        if np.issubdtype(type(val), np.integer):  # Check if it's an integer type
            return f"{int(val)}"  # No decimals for integers
        elif np.issubdtype(type(val), np.floating):  # Check if it's a float
            if val.is_integer():  # If the float is mathematically an integer
                return f"{int(val)}"  # Display as an integer
            else:
                return f"{val:.3f}"  # Otherwise, display with 3 decimals
        return str(val)  # Fallback to default string conversion

    def __get_filter_height_matrix_list(self, height_matrix_list, indices_y):
        # Compute cumulative row start positions for each matrix
        cumulative_rows = np.cumsum(height_matrix_list)

        # Initialize filter_height_matrix_list
        filter_height_matrix_list = np.zeros_like(height_matrix_list)

        # For each matrix, count how many selected rows fall within its row range
        start_idy = 0
        for i, height in enumerate(height_matrix_list):
            end_idy = cumulative_rows[i]
            # Count how many indices_y fall within the range [start_idx, end_idx)
            filter_height_matrix_list[i] = np.sum((indices_y >= start_idy) & (indices_y < end_idy))

            # Move to the next matrix range
            start_idy = end_idy

        # Remove zeroes from filter_height_matrix_list
        no_zero_indices = np.where(filter_height_matrix_list > 0)[0]
        filter_height_matrix_list = filter_height_matrix_list[no_zero_indices]
        return no_zero_indices, filter_height_matrix_list

    def __filter_communities(self, matrix, community_size_x, community_size_y, communities_label_x,
                             communities_label_y, intersection_matrix=None, x_set=None, y_set=None):
        # if threshold on the community size is set, I consider all the communities with size >= threshold
        if self.community_size_th is not None:
            indices_x = np.where(community_size_x >= self.community_size_th)[0]
            indices_y = np.where(community_size_y >= self.community_size_th)[0]
            filter_community_size_x = community_size_x[indices_x]
            filter_community_size_y = community_size_y[indices_y]

            filter_matrix = matrix[indices_y[:, None], indices_x]

            filter_communities_label_x = communities_label_x[indices_x]
            filter_communities_label_y = communities_label_y[indices_y]

            if intersection_matrix is not None:
                filter_intersection_matrix = intersection_matrix[indices_y[:, None], indices_x]
                filter_x_set = x_set[indices_x]
                filter_y_set = y_set[indices_y]
        else:
            filter_community_size_x = community_size_x
            filter_community_size_y = community_size_y
            filter_matrix = matrix

            filter_communities_label_x = communities_label_x
            filter_communities_label_y = communities_label_y

            filter_intersection_matrix = intersection_matrix
            filter_x_set = x_set
            filter_y_set = y_set

        if intersection_matrix is None:
            return filter_matrix, filter_community_size_x, filter_community_size_y, filter_communities_label_x, filter_communities_label_y
        return (filter_matrix, filter_community_size_x, filter_community_size_y, filter_communities_label_x,
                filter_communities_label_y,
                filter_intersection_matrix, filter_x_set, filter_y_set)

    def __filter_communities_and_height_matrix_list(self, matrix, community_size_x,
                                                    community_size_y, communities_label_x, communities_label_y,
                                                    height_matrix_list, y_label_list):
        if self.community_size_th is not None:
            indices_y = np.where(community_size_y >= self.community_size_th)[0]

            # filter the list which contains the length of vertical single layer matrices
            no_zero_indices, filter_height_matrix_list = self.__get_filter_height_matrix_list(height_matrix_list,
                                                                                              indices_y)
            filter_y_label_list = y_label_list[no_zero_indices]
        else:
            filter_height_matrix_list = height_matrix_list
            filter_y_label_list = y_label_list
        filter_matrix, filter_community_size_x, filter_community_size_y, filter_communities_label_x, filter_communities_label_y, = self.__filter_communities(
            matrix, community_size_x, community_size_y, communities_label_x, communities_label_y)

        return filter_matrix, filter_community_size_x, filter_community_size_y, filter_communities_label_x, filter_communities_label_y, filter_height_matrix_list, filter_y_label_list

    def __concat_single_layer_matrix(self, overlapping_tensor, concat_matrix_bool, plot_heatmap_list):
        self.lm.printl(f"{file_name}. __concat_single_layer_matrix started.")
        # INITIALIZE VARIABLES
        # concat_matrix: store the concatenated single layer matrices, for each metric
        concat_matrix = defaultdict(lambda: defaultdict(dict))
        separate_matrix = {}

        # for each concatenated matrix, I save the number of rows, useful in plotting the heatmap,
        # in order to get the correct height of each single layer matrix.
        concat_matrix_n_rows = defaultdict(lambda: defaultdict(list))

        single_layer_co_actions = defaultdict(lambda: defaultdict(list))

        # I save the name of each community in the columns (y_labels) and rows (x_labels)
        community_labels = defaultdict(lambda: defaultdict(lambda: {'x_label': [], 'y_label': []}))

        community_sizes = defaultdict(lambda: defaultdict(lambda: {'x_size_community': [], 'y_size_community': []}))

        for (x_label, y_label), overlapping_matrix_collection in overlapping_tensor.items():
            # in the y-axis must be the concatenated matrices of single layer
            # in the x-axis (generic_label) must be multimodal, flattened ecc.
            if x_label in self.available_list_ca and y_label in self.available_list_ca:
                type_ca, generic_label = y_label, x_label
            elif x_label in self.available_list_ca:
                type_ca, generic_label = x_label, y_label
            elif y_label in self.available_list_ca:
                type_ca, generic_label = y_label, x_label
            if plot_heatmap_list is None or (plot_heatmap_list is not None and generic_label in plot_heatmap_list):
                if concat_matrix_bool:
                    for metric, matrix_object in overlapping_matrix_collection.items():
                        matrix = matrix_object['matrix']
                        # self.lm.printl(f"{file_name}. x_label, y_label, metric, matrix.shape")
                        # Initialize if this is the first time seeing this generic_label + metric
                        if metric not in concat_matrix[generic_label]:
                            concat_matrix[generic_label][metric] = matrix
                        else:
                            self.lm.printl(
                                f"{x_label}, {y_label}, {metric}, {str(concat_matrix[generic_label][metric].shape)}, {matrix.shape}")
                            # Concatenate matrices along the 0 axis (vertically, on the y-axis)
                            concat_matrix[generic_label][metric] = np.concatenate(
                                [concat_matrix[generic_label][metric], matrix], axis=0)

                        # Keep track of heights and labels
                        concat_matrix_n_rows[generic_label][metric].append(matrix.shape[0])
                        single_layer_co_actions[generic_label][metric].append(type_ca)

                        # Extend label information
                        community_labels[generic_label][metric]['y_label'].extend(matrix_object['y_label'])
                        community_labels[generic_label][metric]['x_label'] = matrix_object['x_label']
                        community_sizes[generic_label][metric]['x_size_community'] = matrix_object['x_size_community']
                        community_sizes[generic_label][metric]['y_size_community'].extend(
                            matrix_object['y_size_community'])
                else:
                    separate_matrix[(x_label, y_label)] = overlapping_matrix_collection

        self.lm.printl(f"{file_name}. __concat_single_layer_matrix completed.")
        return concat_matrix, concat_matrix_n_rows, single_layer_co_actions, community_labels, community_sizes, separate_matrix

    def __create_plot_heatmap(self, concat_heatmap, filter_matrix, filter_communities_label_x,
                              filter_communities_label_y, x_label, metric, scale_factor_height=1, n_matrix=None,
                              filter_height_matrix_list=None, filter_y_label_list=None, y_label=None):
        plt.figure(figsize=(len(filter_communities_label_x),
                            len(filter_communities_label_y) * scale_factor_height))  # Adjust height based on number of matrices

        #         num_float = ".3f" if metric!='absolute' else ".0f"
        # Reverse the colormap
        # reversed_cmap = sns.color_palette("viridis", as_cmap=True).reversed()
        reversed_cmap = sns.color_palette("Blues", as_cmap=True)

        ax = sns.heatmap(filter_matrix,
                         cmap=reversed_cmap, cbar=True,
                         xticklabels=filter_communities_label_x, yticklabels=filter_communities_label_y,
                         linewidths=0.5, linecolor='white')

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

        max_value = np.max(filter_matrix)
        min_value = np.min(filter_matrix)
        mid_value = (max_value - min_value) / 2  # 0.5 for harmonicMean
        # plotting the value in each cell. It can be done directly in heatmap, but it is not possible to control
        # the number of decimals according to the type of value. Here I manage integer and float differently
        for i in range(filter_matrix.shape[0]):
            for j in range(filter_matrix.shape[1]):
                # lower values are not plotted
                if filter_matrix[i, j] > 0.035:
                    if filter_matrix[i, j] > mid_value:
                        color = 'white'
                    else:
                        color = 'black'
                    ax.text(j + 0.5, i + 0.5, self.__annotate_format(filter_matrix[i, j]),
                            color=color, ha="center", va="center", fontsize=12)
        if x_label in self.available_list_ca:
            x_label_print = co_action_column_print[x_label]
        else:
            x_label_print = multimodal_print[x_label]
        ax.set_xlabel(x_label_print)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)  # Adjust rotation and font size

        if concat_heatmap:
            # Add horizontal lines to separate the matrices
            cumulative_height = 0
            for i, height in enumerate(filter_height_matrix_list):
                cumulative_height += height
                if i < n_matrix - 1:
                    plt.hlines(cumulative_height, *plt.xlim(), colors='black', linewidth=2)

            # Add y-labels (action names)
            cumulative_height = 0
            for i, y_label in enumerate(filter_y_label_list):
                # replace for print formatting
                y_label_print = co_action_column_print[y_label]
                matrix_height = filter_height_matrix_list[i]
                midpoint = cumulative_height + matrix_height / 2
                plt.text(-0.5, midpoint, y_label_print, va='center', ha='right', fontsize=16, rotation=90)
                cumulative_height += matrix_height

            title = f'{x_label} vs single layer\nmetric: {metric}'
            filename = f"{self.dm.path_overlapping_heatmap}{self.file_prefix}_{x_label}_single_layer_{metric}_th_size_{str(self.community_size_th)}.png"
        else:
            y_label_print = co_action_column_print[y_label]
            # set the label on the y-axis
            ax.set_ylabel(y_label_print)

            title = f'{x_label} vs {y_label}\nmetric: {metric}',
            filename = f"{self.dm.path_overlapping_heatmap}{self.file_prefix}_{x_label}_{y_label}_{metric}_th_size_{str(self.community_size_th)}.png"

        # plt.title(title, fontsize=10)

        # Remove white space
        plt.gca().set_aspect('auto')  # Adjust aspect ratio
        plt.tight_layout(pad=0)  # Tight layout with no padding

        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.show()

    def __plot_concat_matrix(self, concat_matrix, concat_matrix_n_rows, single_layer_co_actions, community_labels,
                             community_sizes, scale_factor_height):
        self.lm.printl(f"{file_name}. __plot_concat_matrix started.")
        for x_label, overlapping_matrix_collection in concat_matrix.items():
            for metric, matrix in overlapping_matrix_collection.items():

                self.lm.printl(
                    f"{file_name}. {x_label} vs single layer, metric:{metric}, filtered community_size_th>={str(self.community_size_th)}.")

                # equal to the number of co-actions the number of vertical single layer matrices
                height_matrix_list = np.array(concat_matrix_n_rows[x_label][metric])
                n_matrix = len(height_matrix_list)
                y_label_list = np.array(single_layer_co_actions[x_label][metric])

                communities_label_x = np.array(community_labels[x_label][metric]['x_label'])
                communities_label_y = np.array(community_labels[x_label][metric]['y_label'])
                community_size_x = np.array(community_sizes[x_label][metric]['x_size_community'])
                community_size_y = np.array(community_sizes[x_label][metric]['y_size_community'])

                filter_matrix, filter_community_size_x, filter_community_size_y, filter_communities_label_x, filter_communities_label_y, filter_height_matrix_list, filter_y_label_list = (
                    self.__filter_communities_and_height_matrix_list(matrix, community_size_x,
                                                                     community_size_y, communities_label_x,
                                                                     communities_label_y, height_matrix_list,
                                                                     y_label_list))
                if filter_matrix.size != 0:
                    self.__create_plot_heatmap(True, filter_matrix, filter_communities_label_x,
                                               filter_communities_label_y, x_label, metric,
                                               scale_factor_height=scale_factor_height,
                                               n_matrix=n_matrix,
                                               filter_height_matrix_list=filter_height_matrix_list,
                                               filter_y_label_list=filter_y_label_list)
                else:
                    self.lm.printl(
                        f"{file_name}. {x_label} vs single layer, filtered community_size_th>={str(self.community_size_th)} implies empty matrix.")
        self.lm.printl(f"{file_name}. __plot_concat_matrix completed.")

    def __plot_separate_matrix(self, separate_matrix, scale_factor_height):
        self.lm.printl(f"{file_name}. __plot_separate_matrix started.")
        for (x_label, y_label), overlapping_matrix_collection in separate_matrix.items():
            for metric, matrix_object in overlapping_matrix_collection.items():
                community_size_x = np.array(matrix_object['x_size_community'])
                community_size_y = np.array(matrix_object['y_size_community'])
                matrix = matrix_object['matrix']
                communities_label_x = np.array(matrix_object['x_label'])
                communities_label_y = np.array(matrix_object['y_label'])

                filter_matrix, filter_community_size_x, filter_community_size_y, filter_communities_label_x, filter_communities_label_y = self.__filter_communities(
                    matrix, community_size_x, community_size_y, communities_label_x, communities_label_y)

                self.__create_plot_heatmap(False, filter_matrix, filter_communities_label_x, filter_communities_label_y,
                                           x_label, metric,
                                           scale_factor_height=scale_factor_height,
                                           y_label=y_label)
        self.lm.printl(f"{file_name}. __plot_separate_matrix completed.")

    def __max_similarity_match(self, similarity_matrix):
        # Convert similarity matrix to a cost matrix by negating it
        cost_matrix = -similarity_matrix

        # Solve the assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Compute the total similarity for the optimal assignment
        total_similarity = similarity_matrix[row_indices, col_indices].sum()

        # Prepare the list of matched pairs
        matches = [(i, j) for i, j in zip(row_indices, col_indices)]

        # Identify unmatched communities in A and B
        unmatched_rows = list(set(range(similarity_matrix.shape[0])) - set(row_indices))
        unmatched_cols = list(set(range(similarity_matrix.shape[1])) - set(col_indices))
        return matches, unmatched_rows, unmatched_cols, total_similarity

    # def __get_max_y_axis_limit(self, type_aggregation, flux_df):
    #     if type_aggregation == 'communities':
    #         # Step 1: Find the maximum y-axis limit across all "generic" values
    #         max_y = 0
    #         for generic_value in flux_df['generic'].unique():
    #             # Filter the dataframe for the current "generic" value
    #             flux_df_generic = flux_df[flux_df['generic'] == generic_value]
    #
    #             # Calculate the frequency of each 'label' for each 'layer' within this subset
    #             frequency_df = flux_df_generic.groupby(['layer', 'label']).size().unstack(fill_value=0)
    #
    #             # Update max_y with the highest sum of values in any layer
    #             max_y = max(max_y, frequency_df.sum(axis=1).max())
    #         max_y = max_y + 1  # Add 1 to the maximum value for padding
    #     elif type_aggregation == 'users':
    #         # Find the maximum y-axis limit across all generic values
    #         max_y = flux_df.groupby(['layer', 'generic'])[['lost_nodes', 'common_nodes', 'gained_nodes']].sum().sum(
    #             axis=1).max()
    #         max_y = max_y + 5000  # Add 5000 to the maximum value for padding. In this case the number of nodes is very high
    #
    #     return max_y

    def __get_max_y_axis_limit(self, type_aggregation, flux_df, generic_label):
        # if generic_label is co-retweet, co-hahstag ecc. I plot choosing the maximum value across the specific single layers,
        # since I will analyze the plot on its own, while if the generic_label is multimodal, I plot the maximum value
        # between multimodal and flat_weighted_sum_louvain
        if generic_label in co_action_column_print.values():
            if type_aggregation == 'communities':
                # Step 1: Find the maximum y-axis limit across all "generic" values
                max_y = 0
                flux_df_generic = flux_df[flux_df['generic'] == generic_label]

                # Calculate the frequency of each 'label' for each 'layer' within this subset
                frequency_df = flux_df_generic.groupby(['layer', 'label']).size().unstack(fill_value=0)

                # Update max_y with the highest sum of values in any layer
                max_y = max(max_y, frequency_df.sum(axis=1).max())
                max_y = max_y + 1  # Add 1 to the maximum value for padding
            elif type_aggregation == 'users':
                flux_df_generic = flux_df[flux_df['generic'] == generic_label]
                # Find the maximum y-axis limit across all generic values
                max_y = flux_df_generic.groupby(['layer', 'generic'])[['lost_nodes', 'common_nodes', 'gained_nodes']].sum().sum(axis=1).max()
                if self.dataset_name == 'uk':
                    y_offset = 5000
                elif self.dataset_name == 'IORussia':
                    y_offset = 500
                max_y = max_y + y_offset  # Add y_offset to the maximum value for padding. In this case the number of nodes is very high
        else:
            if type_aggregation == 'communities':
                # Step 1: Find the maximum y-axis limit across all "generic" values
                max_y = 0
                for generic_value in ['multimodal', 'flat_weighted_sum_louvain']:
                    # Filter the dataframe for the current "generic" value
                    flux_df_generic = flux_df[flux_df['generic'] == generic_value]

                    # Calculate the frequency of each 'label' for each 'layer' within this subset
                    frequency_df = flux_df_generic.groupby(['layer', 'label']).size().unstack(fill_value=0)

                    # Update max_y with the highest sum of values in any layer
                    max_y = max(max_y, frequency_df.sum(axis=1).max())
                max_y = max_y + 1  # Add 1 to the maximum value for padding
            elif type_aggregation == 'users':
                flux_df_generic = flux_df[flux_df['generic'].isin(['multimodal', 'flat_weighted_sum_louvain'])]
                # Find the maximum y-axis limit across all generic values
                max_y = flux_df_generic.groupby(['layer', 'generic'])[['lost_nodes', 'common_nodes', 'gained_nodes']].sum().sum(axis=1).max()
                if self.dataset_name == 'uk':
                    y_offset = 5000
                elif self.dataset_name == 'IORussia':
                    y_offset = 1000
                max_y = max_y + y_offset  # Add y_offset to the maximum value for padding. In this case the number of nodes is very high

        return max_y


    def __plot_flux_df(self, type_aggregation, flux_df, mid_th, metric):
        self.lm.printl(f"{file_name}. __plot_flux_df started.")
        flux_df['generic'] = flux_df['generic'].replace(co_action_column_print)
        flux_df['layer'] = flux_df['layer'].replace(co_action_column_print)


        # Loop through each unique value in the "generic" column
        for generic_label in flux_df['generic'].unique():
            self.lm.printl(f"{file_name}. generic_label: {generic_label}")
            # Filter the dataframe for the current "generic" value
            flux_df_generic = flux_df[flux_df['generic'] == generic_label]
            # if the generic label is a co-action, i am plotting the flux from the single layer to the other single layers
            # in this case the combination layer-generic with itself must be removed
            if generic_label in color_dict.keys():
                flux_df_generic = flux_df_generic[flux_df_generic['layer'] != generic_label]

            if type_aggregation == 'communities':
                # Calculate the frequency of each 'label' for each 'layer'
                frequency_df = flux_df_generic.groupby(['layer', 'label']).size().unstack(fill_value=0)
                # Specify the desired order of labels
                label_order = ['lost', 'common', 'gained']  # Replace with the actual label values in the order you want
                # it is not said that the all the labels are present in the dataframe columns
                reordered_columns = [col for col in label_order if col in frequency_df.columns]
                # colors = {'lost': 'red', 'common': '#1f77b4', 'gained': 'green'}  # Replace with your desired colors

                title = f'Communities flux, from single layer to {generic_label}\nmid_th={mid_th}, metric={metric}'
                legend = 'Communities label'
                filename = f"{self.dm.path_overlapping_stacked_plot}{self.file_prefix}_{generic_label}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}.png"
            elif type_aggregation == 'users':
                # Sum up the values for each layer
                frequency_df = flux_df_generic.groupby('layer')[['lost_nodes', 'common_nodes', 'gained_nodes']].sum()

                # Specify the desired order of labels
                label_order = ['lost_nodes', 'common_nodes', 'gained_nodes']
                # it is not said that the all the labels are present in the dataframe columns
                reordered_columns = [col for col in label_order if col in frequency_df.columns]
                # colors = {'lost_nodes': 'red', 'common_nodes': '#1f77b4',
                #           'gained_nodes': 'green'}  # Replace with your desired colors

                title = f'User nodes flux, from single layer to {generic_label}'
                legend = 'Users label'

                filename = f"{self.dm.path_overlapping_stacked_plot}{self.file_prefix}_{generic_label}_{metric}_th_size_{str(self.community_size_th)}.png"


            # Reorder the columns in frequency_df
            frequency_df = frequency_df[reordered_columns]

            # Set the order of the layers on the x-axis
            layer_order = list(co_action_column_print.values())
            if generic_label in co_action_column_print.values():
                layer_order.remove(generic_label)
            frequency_df = frequency_df.loc[layer_order]  # Reorder rows to match the desired x-axis order

            # Ensure 'layer' is categorical with the specified order
            frequency_df.index = pd.CategoricalIndex(frequency_df.index, categories=layer_order, ordered=True)

            # Plot stacked bar chart
            ax = frequency_df.plot(kind='bar', stacked=True, figsize=(8, 6), color=palette.values(), legend=False, width=0.9)
            # plt.xlabel('Co-action')
            # plt.ylabel('Frequency')
            # plt.title(title)
            # plt.legend(title=legend)
            max_y = self.__get_max_y_axis_limit(type_aggregation, flux_df, generic_label)
            plt.ylim(0, max_y)  # Set y-axis limit

            # Remove axis names
            plt.xlabel("")  # Remove x-axis name
            plt.ylabel("")  # Remove y-axis name

            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=16)  # Rotate and align to the right

            # Add gridlines
            ax.set_axisbelow(True) # Ensure gridlines are behind the bars
            ax.grid(axis='y', linestyle='--', linewidth=0.5, color='gray')  # Add gridlines for the y-axis

            # Add annotations for each segment
            for container in ax.containers:
                if type_aggregation == 'communities':
                    labels = [int(v) if v > 0 else '' for v in container.datavalues]
                elif type_aggregation == 'users':
                    # Only label values >= 1000 because the segment is too small for smaller values
                    if self.dataset_name == 'uk':
                        th_plot_y = 1000
                    elif self.dataset_name == 'IORussia':
                        th_plot_y = 100
                    labels = [int(v) if v >= th_plot_y else '' for v in container.datavalues]  # Only label values >= th_plot_y
                if type_aggregation == 'communities':
                    fontsize = 16
                elif type_aggregation == 'users':
                    fontsize = 14
                ax.bar_label(container, labels=labels, label_type='center', fontsize=fontsize)

            plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.show()
        self.lm.printl(f"{file_name}. __plot_flux_df completed.")

    def __instantiation_data_flux(self, metric, type_aggregation):
        if type_aggregation == 'users' and metric != 'absolute':
            m = f"'{type_aggregation}' are compatible only with 'absolute' metric."
            raise ValueError(m)
        if type_aggregation == 'communities':

            data_flux = {'layer': [], 'generic': [], 'com_layer': [], 'com_generic': [], 'communities': [], 'label': []}
        elif type_aggregation == 'users':
            data_flux = {'layer': [], 'generic': [], 'com_layer': [], 'com_generic': [], 'communities': [],
                         'lost_nodes': [], 'common_nodes': [], 'gained_nodes': []}

        return data_flux

    def __crete_flux_df_communities(self, data_flux, filter_matrix,
                                    filter_communities_label_x, filter_communities_label_y,
                                    ca_abbr, abbr_generic_label, type_ca, generic_label, mid_th, matches,
                                    unmatched_rows, unmatched_cols):
        # for each tuple of indices, I check if the value is greater than the mid_th, in this case it is common
        # otherwise, it is lost (the community from the single layer) and gained the one from the multimodal
        for i, j in matches:
            com_x = filter_communities_label_x[j]
            com_y = filter_communities_label_y[i]
            if filter_matrix[i, j] > mid_th:
                data_flux['com_layer'].append(com_y)
                data_flux['com_generic'].append(com_x)
                data_flux['communities'].append(f'{com_x}_{com_y}')
                data_flux['label'].append('common')
                data_flux['layer'].append(type_ca)
                data_flux['generic'].append(generic_label)
            else:
                data_flux['com_layer'].append(com_y)
                data_flux['com_generic'].append(np.nan)
                data_flux['communities'].append(f'{ca_abbr}_{com_y}')
                data_flux['label'].append('lost')
                data_flux['layer'].append(type_ca)
                data_flux['generic'].append(generic_label)

                data_flux['com_layer'].append(np.nan)
                data_flux['com_generic'].append(com_x)
                data_flux['communities'].append(f'{abbr_generic_label}_{com_x}')
                data_flux['label'].append('gained')
                data_flux['layer'].append(type_ca)
                data_flux['generic'].append(generic_label)

        # If the number of communities in the single layer is greater than the multimodal, I add the lost communities
        # otherwise, if the number of communities in the multimodal is greater than the single layer, I add the gained communities
        for i in unmatched_rows:
            com_y = filter_communities_label_y[i]
            data_flux['com_layer'].append(com_y)
            data_flux['com_generic'].append(np.nan)
            data_flux['communities'].append(f'{ca_abbr}_{com_y}')
            data_flux['label'].append('lost')
            data_flux['layer'].append(type_ca)
            data_flux['generic'].append(generic_label)

        for j in unmatched_cols:
            com_x = filter_communities_label_x[j]
            data_flux['com_layer'].append(np.nan)
            data_flux['com_generic'].append(com_x)
            data_flux['communities'].append(f'{abbr_generic_label}_{com_x}')
            data_flux['label'].append('gained')
            data_flux['layer'].append(type_ca)
            data_flux['generic'].append(generic_label)
        return data_flux

    def __add_set_to_dict(self, input_set, label, com_y, com_x, com, type_ca, generic_label, node_label_dict):
        node_label_dict["userId"].extend(input_set)
        node_label_dict["label"].extend([label] * len(input_set))
        node_label_dict["com_layer"].extend([com_y] * len(input_set))
        node_label_dict["com_generic"].extend([com_x] * len(input_set))
        node_label_dict["communities"].extend([com] * len(input_set))
        node_label_dict["layer"].extend([type_ca] * len(input_set))
        node_label_dict["generic"].extend([generic_label] * len(input_set))

    def __create_flux_df_users(self, data_flux, filter_matrix, filter_community_size_x, filter_community_size_y,
                               filter_communities_label_x, filter_communities_label_y, ca_abbr, abbr_generic_label,
                               type_ca, generic_label, matches, unmatched_rows, unmatched_cols,
                               filter_intersection_matrix, filter_x_set, filter_y_set, node_label_dict):
        # absolute, I save the number of lost, common and gained nodes. Nodes-based analysis, instead of community-based
        for i, j in matches:
            com_x = filter_communities_label_x[j]
            com_y = filter_communities_label_y[i]
            data_flux['com_layer'].append(com_y)
            data_flux['com_generic'].append(com_x)
            data_flux['communities'].append(f'{com_x}_{com_y}')
            data_flux['layer'].append(type_ca)
            data_flux['generic'].append(generic_label)

            size_x = filter_community_size_x[j]
            size_y = filter_community_size_y[i]
            data_flux['lost_nodes'].append(size_y - filter_matrix[i, j])
            data_flux['common_nodes'].append(filter_matrix[i, j])
            data_flux['gained_nodes'].append(size_x - filter_matrix[i, j])

            self.__add_set_to_dict(filter_y_set[i] - filter_intersection_matrix[i, j],
                                   'lost', com_y, np.nan, f'{com_x}_{com_y}', type_ca, generic_label,
                                   node_label_dict)
            self.__add_set_to_dict(filter_intersection_matrix[i, j],
                                   'common', com_y, com_x, f'{com_x}_{com_y}', type_ca, generic_label,
                                   node_label_dict)
            self.__add_set_to_dict(filter_x_set[j] - filter_intersection_matrix[i, j],
                                   'gained', np.nan, com_x, f'{com_x}_{com_y}', type_ca, generic_label,
                                   node_label_dict)

        for i in unmatched_rows:
            com_y = filter_communities_label_y[i]
            data_flux['com_layer'].append(com_y)
            data_flux['com_generic'].append(np.nan)
            data_flux['communities'].append(f'{ca_abbr}_{com_y}')
            data_flux['layer'].append(type_ca)
            data_flux['generic'].append(generic_label)

            size_y = filter_community_size_y[i]
            data_flux['lost_nodes'].append(filter_community_size_y[i])
            data_flux['common_nodes'].append(0)
            data_flux['gained_nodes'].append(0)

            self.__add_set_to_dict(filter_y_set[i],
                                   'lost', com_y, np.nan, f'{ca_abbr}_{com_y}', type_ca, generic_label,
                                   node_label_dict)

        for j in unmatched_cols:
            com_x = filter_communities_label_x[j]
            data_flux['com_layer'].append(np.nan)
            data_flux['com_generic'].append(com_x)
            data_flux['communities'].append(f'{abbr_generic_label}_{com_x}')
            data_flux['layer'].append(type_ca)
            data_flux['generic'].append(generic_label)

            size_x = filter_community_size_x[j]
            data_flux['lost_nodes'].append(0)
            data_flux['common_nodes'].append(0)
            data_flux['gained_nodes'].append(size_x)

            self.__add_set_to_dict(filter_x_set[j],
                                   'gained', np.nan, com_x, f'{abbr_generic_label}_{com_x}', type_ca, generic_label,
                                   node_label_dict)

        return data_flux, node_label_dict

    def __create_flux_df(self, data_flux, filter_matrix, filter_community_size_x, filter_community_size_y,
                         filter_communities_label_x, filter_communities_label_y, ca_abbr, abbr_generic_label,
                         type_ca, generic_label, mid_th, type_aggregation, matches, unmatched_rows, unmatched_cols,
                         filter_intersection_matrix, filter_x_set, filter_y_set, node_label_dict):
        if type_aggregation == 'communities':
            data_flux = self.__crete_flux_df_communities(data_flux, filter_matrix,
                                                         filter_communities_label_x, filter_communities_label_y,
                                                         ca_abbr, abbr_generic_label, type_ca, generic_label,
                                                         mid_th, matches, unmatched_rows, unmatched_cols)
            node_label_dict = None
        elif type_aggregation == 'users':
            data_flux, node_label_dict = self.__create_flux_df_users(data_flux, filter_matrix,
                                                                     filter_community_size_x, filter_community_size_y,
                                                                     filter_communities_label_x,
                                                                     filter_communities_label_y,
                                                                     ca_abbr, abbr_generic_label, type_ca,
                                                                     generic_label, matches,
                                                                     unmatched_rows, unmatched_cols,
                                                                     filter_intersection_matrix, filter_x_set,
                                                                     filter_y_set, node_label_dict)
        return data_flux, node_label_dict


    def __get_offsets_iorussia(self):
        # Define the conditions and corresponding offsets
        if self.file_prefix == 'louvain_resolution_1':
            # offset condition with abbreviations
            offset_conditions = {
            ("co-mention", 0, 1): (15, 0),
            ("co-mention", 1, 2): (0, -10),
            ("co-mention", 3, 0): (0, -10),
            ("co-mention", 5, 4): (15, 0),
            ("co-mention", 6, 5): (15, 0),
            ("co-hashtag", 0, 2): (10, 5),
            ("co-hashtag", 1, 1): (0, -10),
            ("co-hashtag", 3, 4): (15, 0),
            ("co-hashtag", 4, 0): (-15, 0),
            ("co-URL", 0, 1): (10, 0),
            ("co-URL", 1, 0): (0, 8),
        }
        elif self.file_prefix == 'infomap':
            offset_conditions = {
            ("co-mention", 0, 0): (0, -2),
            ("co-mention", 2, 3): (-0.5, -2),
            ("co-hashtag", 0, 0): (0, -2),
            ("co-hashtag", 2, 1): (100, 0),
            ("co-URL", 0, 1): (100, 0),
            ("co-URL", 2, 4): (0, -2),
            
        }
        return offset_conditions


    def __get_offsets_uk(self):
        if self.file_prefix == 'louvain_resolution_1':
            # offset condition with abbreviations
            offset_conditions = {
                ("co-mention", 0, 0): (0, 25),
                #         ("co-retweet", 0, 0): (0, 10),
                ("co-mention", 1, 1): (0, 25),
                #         ("co-retweet", 1, 1): (5, 15),
                ("co-mention", 2, 2): (500, 0),
                #         ("co-retweet", 2, 2): (-5, -10),
                ("co-mention", 3, 3): (-25, 25),
                #         ("co-retweet", 3, 3): (10, 5),
                ("co-mention", 4, 4): (500, 0),
                #         ("co-retweet", 4, 4): (-10, -5),
                ("co-hashtag", 0, 0): (0, 25),
                #         ("co-retweet", 0, 0): (15, 20),
                ("co-hashtag", 2, 1): (500, -20),
                #         ("co-retweet", 1, 1): (20, -15),
                ("co-hashtag", 3, 2): (-400, 28),
                #         ("co-retweet", 2, 2): (-20, 10),
                ("co-hashtag", 5, 4): (50, 30),
                #         ("co-retweet", 4, 4): (10, -20),
                ("co-URL", 4, 1): (400, -20),
                #         ("co-retweet", 1, 1): (-15, 25),
            }
        elif self.file_prefix == 'infomap':
            offset_conditions = {
                ("co-mention", 0, 0): (-1000, 0),
                ("co-mention", 1, 1): (0, 20),
                ("co-mention", 5, 2): (0, 20),
                ("co-hashtag", 0, 0): (-1000, 0),
                ("co-URL", 0, 0): (400, -20),
            }
        return offset_conditions

    def __get_offsets(self, point):
        if self.dataset_name == 'uk':
            offset_conditions = self.__get_offsets_uk()
        elif self.dataset_name == 'IORussia':
            offset_conditions = self.__get_offsets_iorussia()
        
        # Map the condition to the keys used in the printing format
        updated_offset_conditions = {
            (co_action_column_print3.get(key[0], key[0]), key[1], key[2]): value
            for key, value in offset_conditions.items()
        }

        # Match against conditions
        key = (point['layer'], point['com_layer'], point['com_generic'])

        return updated_offset_conditions.get(key, (0, 0))  # Default to (0, 0) if no match

    def __scatterplot(self, features, plot_df, x, y, generic, title, filename):
        self.lm.printl(f"{file_name}. __scatterplot Size: {str(plot_df.shape)}.")
        # if generic == 'co-retweet':
        #     print(plot_df[features+[x, y]])

        # Create a color palette based on unique com_generic values
        unique_com_generic = plot_df['com_generic'].dropna().unique()
        palette = sns.color_palette("pastel", len(unique_com_generic))
        color_map = dict(zip(unique_com_generic, palette))
        unique_layers = plot_df["layer"].unique()
        n_current_layer = len(unique_layers)
        markers = ["o", "s", "d", "^", "*"][0:n_current_layer]

        # Initialize the plot
        plt.figure(figsize=(8, 6))

        # Create scatter plot with seaborn
        scatter = sns.scatterplot(
            data=plot_df,
            x=x, y=y,
            hue="com_generic",
            style="layer",
            palette=color_map,
            s=150,
            markers=markers,
            legend=False,
            edgecolor="white",  # Add a white border around the markers
            linewidth=1.2,  # Set the thickness of the border
        )


        # Add the legend for 'layer' based on the markers
        handles = [Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='gray', markersize=10)
                   for i in range(len(unique_layers))]
        # Create the legend for marker styles
        if self.dataset_name == 'IORussia' and self.file_prefix == 'louvain_resolution_1':
            legend_layer = plt.legend(handles, unique_layers, title="Layer", loc="best", borderaxespad=0.5)
        else:
            legend_layer = plt.legend(handles, unique_layers, title="Layer", loc="best",  bbox_to_anchor=(1, 1), borderaxespad=0.5)

        # Add the legend for 'com_generic' based on the palette
        labels_legend = ["group " + str(item) for item in list(color_map.keys())]
        handles_generic = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cat], markersize=10)
                           for cat in color_map]
        legend_generic = plt.legend(handles=handles_generic, labels=labels_legend, title="Community RTW", loc="best",
                   bbox_to_anchor=(1, 0.7), borderaxespad=0.5)

        plt.gca().add_artist(legend_layer)
        plt.gca().add_artist(legend_generic)

        # Add grid to the plot
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Customize grid appearance

        # Draw lines between points A and B with color based on com_generic, and add layer text near point A
        for _, A in plot_df[plot_df['layer'] != generic].iterrows():
            # Find corresponding point B
            B = plot_df[(plot_df['com_generic'] == A['com_generic']) & (plot_df['layer'] == generic)]

            if not B.empty:
                B = B.iloc[0]  # Get the first match

                # Get the color associated with com_generic for point A
                line_color = color_map[A['com_generic']]

                # Draw a line between point A and point B without a marker
                plt.plot([A[x], B[x]], [A[y], B[y]], color=line_color, linewidth=2)

                # Add a small offset to the position of text near point A
                offset_x = 0  # Adjust this value as needed for spacing
                offset_y = 0  # Adjust this value as needed for spacing
                offset_x, offset_y = self.__get_offsets(A)

                text_x = A[x] + offset_x
                text_y = A[y] + offset_y

                # Add text label with the layer of point A near point A
                plt.text(text_x, text_y, str(A['layer']), color=line_color,
                         ha='center', va='center', fontsize=10, fontweight='bold')

        # Add title
        # plt.title(title)
        # Remove axis labels
        plt.xlabel("")
        plt.ylabel("")

        # To remove the axis labels and also hide the tick labels and ticks from both x and y axes in your Seaborn/Matplotlib scatter plot, replace with
        # plt.gca().tick_params(labelbottom=False, labelleft=False)
        ax = plt.gca()

        # Hide tick marks but keep grid
        ax.tick_params(
            bottom=False,  # remove x-axis ticks
            left=False,  # remove y-axis ticks
            labelbottom=False,  # remove x-axis tick labels
            labelleft=False  # remove y-axis tick labels
        )

        try:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
        except Exception as e:
            self.lm.printl(e)
            plt.savefig(filename, dpi=dpi)
        # Show the plot
        plt.show()
        self.lm.printl(f"{file_name}. __scatterplot completed.")

    def __plot_starplot(self, starplot_df, features, normalized, metric=None, mid_th=None, ax=None, show_legend=False):
        if normalized:
            features_norm = ["norm_" + f for f in features]
        else:
            features_norm = features

        generic = starplot_df.iloc[0]['generic']
        layer = starplot_df.iloc[0]['layer']

        # Extract values for both rows (no need to normalize since it's already done)
        values1 = starplot_df.iloc[0][features_norm].values.tolist()  # values for the first row (always the layer)
        values2 = starplot_df.iloc[1][features_norm].values.tolist()  # values for the second row (always the generic)

        # Extract layer and community values for title and legend
        com_layer, com_generic = str(int(starplot_df.iloc[0]['com_layer'])), str(
            int(starplot_df.iloc[0]['com_generic']))

        # Determine the number of variables we're plotting
        num_vars = len(features_norm)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Close the plot loop
        values1 += values1[:1]
        values2 += values2[:1]

        # Create figure and axis if no axis is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            show_plot = True  # Flag to control whether to display the plot at the end
        else:
            show_plot = False

        color_layer = color_dict2[layer]
        color_generic = color_dict2[generic]
        # Plot filled areas and lines for both sets of values
        ax.fill(angles, values1, color=color_layer, alpha=0.5, label=f'{layer}')
        ax.fill(angles, values2, color=color_generic, alpha=0.5, label=f'{generic}')
        ax.plot(angles, values1, color=color_layer, linewidth=2)
        ax.plot(angles, values2, color=color_generic, linewidth=2)

        # Set the labels for each axis
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=16)

        # Customize the grid lines to be gray and dashed
        ax.grid(color='gray', linestyle='--', linewidth=1.5)
        ax.spines['polar'].set_visible(False)

        # Add radial text labels for inner circles
        for r in np.linspace(0.2, 1, 5):
            # i move slightly the x and y position of the text, the x is moved inversely proportional to the radius
            # the inner circles must be moved more than the outer ones
            ax.text(np.pi / 2-(1/r)*0.22, r+0.09, f"{r:.2f}", color='gray', ha='center', va='center', fontsize=12)
        ax.set_ylim(0, 1)

        # Set title with community info
        title = f"{generic}_{com_generic} and {layer}_{com_layer}"
        ax.set_title(title, fontsize=18, ha='center', va='bottom', fontweight='bold')

        # if not show_plot and show_legend:
        # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Layers')

        # Add legend if this is a standalone plot
        if show_plot:
            # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Layers')
            plt.tight_layout()
            filename = f"{self.dm.path_overlapping_starplot}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_{generic}_{layer}_{com_generic}_{com_layer}_starplot.png"
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.show()

    def __plot_legend_starplot(self, plot_df, generic):
        self.lm.printl(f"{file_name}. __plot_legend_starplot started.")
        if plot_df.shape[0] > 0:

            # plot legend
            set_generic = set(plot_df['generic'].unique())
            set_layer = set(plot_df['layer'].unique())
            list_show_layer = list(set_layer.union(set_generic))

            # Filter the dictionary
            filtered_color_dict = {k: v for k, v in color_dict2.items() if k in list_show_layer}

            # Create a figure for the legend
            fig, ax = plt.subplots(figsize=(6, 2))  # Adjust the size as needed
            ax.axis("off")  # Turn off the axis

            # Create handles and labels for the legend
            handles = [plt.Line2D([0], [0], color=color, lw=6) for color in filtered_color_dict.values()]
            labels = list(filtered_color_dict.keys())

            # Add the legend to the plot
            legend = ax.legend(
                handles,
                labels,
                # title="Layers",
                loc="center",
                frameon=False,  # Remove the box around the legend
                ncol=len(filtered_color_dict),  # Arrange items in a single row
                bbox_to_anchor=(0.5, 0.5),  # Center the legend
                bbox_transform=fig.transFigure,
            )

            # Save the legend figure with no padding and no space
            plt.savefig(f"{self.dm.path_overlapping_starplot}{generic}_starplot_legend.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()
        self.lm.printl(f"{file_name}. __plot_legend_starplot completed.")

    def __compute_weight_stats_summary(self,df):
        """
        Compute mean of average_weight, median_weight, std_weight, and mad_weight
        per (generic, label), following rules:
        - label == 'common': combine layer + generic
        - label == 'gained': use only *_generic
        - label == 'lost': use only *_layer

        Returns
        -------
        pd.DataFrame with columns:
        ['generic', 'label', 'avg_average_weight', 'avg_median_weight',
        'avg_std_weight', 'avg_mad_weight']
        """

        metrics = ['avg_weight', 'median_weight', 'std_weight', 'mad_weight']
        results = []

        for (generic, label), group in df.groupby(['generic', 'label']):
            row = {'generic': generic, 'label': label}

            for metric in metrics:
                col_layer = f"{metric}_layer"
                col_generic = f"{metric}_generic"

                if label == 'common':
                    values = pd.concat([group[col_layer], group[col_generic]])
                elif label == 'gained':
                    values = group[col_generic]
                elif label == 'lost':
                    values = group[col_layer]
                else:
                    values = pd.Series(dtype=float)

                row[f"avg_{metric}"] = values.mean() if len(values) > 0 else np.nan

            results.append(row)

        return pd.DataFrame(results)

    def __add_coordination_label(self, df, col="percCoord", thre_co=0.68, thre_not=0.10):
        """
        Add a labelCoordination column based on thresholds applied to percCoord.
        
        - percCoord >= 0.68 → 'coordinated'
        - percCoord <= 0.10 → 'notCoordinated'
        - otherwise → 'mixed'  (or 'split' if preferred)
        """
        def classify(p):
            if p >= thre_co:
                return "coordinated"
            elif p <= thre_not:
                return "notCoordinated"
            else:
                return "mixed"   # or "split"
        
        df["labelCoordination"] = df[col].apply(classify)
        return df


    def __validation_by_label(self, df):
        """
        Count labelCoordination values per (generic, label) using rules:
        - common: count both *_layer and *_generic
        - gained: count only *_layer
        - lost: count only *_generic
        """

        results = []
        for (generic, layer, label), group in df.groupby(["generic", "layer", "label"]):
            row = {"generic": generic, "layer":layer, "label": label}
            if label == "common" or label == "gained":
                dict_value_counts = dict(group["labelCoordination_generic"].value_counts())
            elif label == "lost":
                dict_value_counts = dict(group["labelCoordination_layer"].value_counts())
            row.update(dict_value_counts)

            results.append(row)

        results_df = pd.DataFrame(results)
        results_df = results_df.fillna(0)
        
        return results_df

    def __get_shift_coordination_mean(self,generic, label):
        shift = 0
        if self.dataset_name == 'uk' and self.file_prefix == 'infomap':
            if generic == 'flat_weighted_sum_infomap' and label == 'common':
                shift = -2.5
            elif generic == 'multimodal' and label == 'gained':
                shift = -1
        elif self.dataset_name == 'uk' and self.file_prefix == 'louvain_resolution_1':
            if generic == 'flat_weighted_sum_louvain' and (label == 'lost' or label == 'common'):
                shift = -3
            elif generic == 'multimodal' and label == 'gained':
                shift = -4
        elif self.dataset_name == 'IORussia' and self.file_prefix == 'infomap':
            if generic == 'flat_weighted_sum_infomap' and (label == 'common' or label == 'gained'):
                shift = -2.5
        elif self.dataset_name == 'IORussia' and self.file_prefix == 'louvain_resolution_1':
            if generic == 'flat_weighted_sum_louvain' and label == 'lost':
                shift = -2

        return shift

    def __darken(self, color, factor=0.7):
        """Darken a color by multiplying its RGB by a factor (0–1)."""
        return tuple([c * factor for c in color])
    
    def __get_offset_cosine_similarity(self, generic, layer):
        offset = 0
        if generic == 'RTW':
            if self.dataset_name == 'uk' and self.file_prefix == 'louvain_resolution_1':
                if layer == 'HST':
                    offset = -0.4
            elif self.dataset_name == 'IORussia' and self.file_prefix == 'louvain_resolution_1':
                if layer == 'MEN':
                    offset = -1
                if layer == 'HST':
                    offset = -0.4
            elif self.dataset_name == 'IORussia' and self.file_prefix == 'infomap':
                if layer == 'URL' or layer == 'HST':
                    offset = -0.2
        return offset
    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------

    def compute_overlapping(self, save_overlapping_tensor=True, save_intersections=False):
        df_x = self.ch_x.read_dataframe(self.dm_x.path_user_dataframe + "com_df.csv", dtype=dtype)
        df_y = self.ch_y.read_dataframe(self.dm_y.path_user_dataframe + "com_df.csv", dtype=dtype)
        # print(len(df_x['cid'].unique()))
        # print(len(df_y['group'].unique()))
        # print(df_x.shape)
        # print(df_y.shape)
        # print(df_x['cid'].unique())
        # print(df_y['userId'].dtype)
        # print(df_x['actor'].dtype)

        x_label, y_label, df_x, df_y = self.__get_labels_reordered(df_x, df_y)
        # FROM NOW ON x_label is the multimodal label, y_label is the one-layer label
        self.lm.printl(f"{file_name}. unique layer df_x: {df_x['layer'].unique()}.")
        if self.community_size_th is not None:
            df_x = self.__filter_community_size(df_x)
            df_y = self.__filter_community_size(df_y)
            self.file_prefix = f"{self.file_prefix}_th_{str(self.community_size_th)}"

        self.lm.printl(f"{file_name}. compute_overlapping start ({x_label}, {y_label}).")
        tensor_path = self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_tensor.p"
        if os.path.exists(tensor_path):
            overlapping_tensor = self.ch.load_object(tensor_path)
        else:
            overlapping_tensor = {}

        tensor_set_path = self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_set.p"
        if os.path.exists(tensor_set_path):
            overlapping_set_tensor = self.ch.load_object(tensor_set_path)
        else:
            overlapping_set_tensor = {}

        n_com_x = len(df_x['group'].unique())
        n_com_y = len(df_y['group'].unique())
        total_combination = n_com_x * n_com_y
        self.lm.printl(f"{file_name}. nCommunities: {n_com_x} x {n_com_y}={total_combination} combinations.")

        user_set_df_x, user_set_df_y = self.__extract_user_set(x_label, y_label, df_x, df_y)
        # user_set_df_x['userSet'].apply(len).tolist() user_set_df_y['userSet'].apply(len).tolist()
        # overlapping_matrix_dict = {}
        # Initialize the overlapping matrix numpy format, composed by the matrix and the community labels, which are used
        # as x and y labels in the heatmap

        # x_size_community and y_size_community are the number of users in each community. I exploit df_x,
        # user_set_df_x contains empty set for some communities, the ones that do not have any actors in this specific single layer
        # So the number of users in each community is the number of users, without considering the layer.
        overlapping_matrix_numpy = {
            metric: {
                'matrix': [],
                'x_label': list(user_set_df_x['group'].unique()),  # community labels
                'y_label': list(user_set_df_y['group'].unique()),  # community labels
                'x_size_community': df_x.groupby('group').size().values,  # number of users in each community
                'y_size_community': df_y.groupby('group').size().values,  # number of users in each community
            }
            for metric in available_overlapping_metrics
        }

        overlapping_set = {
            'x_set': [],
            'y_set': [],
            'overlapping_set': [],
            'x_label': list(user_set_df_x['group'].unique()),  # community labels
            'y_label': list(user_set_df_y['group'].unique()),  # community labels
        }

        count = 0
        for index_y, row_y in user_set_df_y.iterrows():
            group_y, c_y = row_y['group'], row_y['userSet']
            overlapping_set['y_set'].append(c_y)

            # overlapping_matrix_dict[group_y] = {}
            row_dict = {metric: [] for metric in available_overlapping_metrics}
            row_dict2 = []  # it is just one list, since it is equal for all the metrics
            for index_x, row_x in user_set_df_x.iterrows():  # if there is multimodal, x is multimodal, so it goes on the rows of the matrix
                group_x, c_x = row_x['group'], row_x['userSet']

                overlapping_dict, intersection = self.__overlapping_measures(c_x, c_y)
                
                overlapping_set['x_set'].append(c_x)
                row_dict2.append(intersection)

                # overlapping_matrix_dict[group_y][group_x] = overlapping_dict # dictionary format
                # save the values of the row, for each metric
                for metric, value in overlapping_dict.items():
                    row_dict[metric].append(value)
                count += 1
                self.lm.printK(count, 100, f"{file_name}. compute_overlapping {count}/{total_combination} completed.")

            # save the row in the matrix for each metric
            for metric, row in row_dict.items():
                overlapping_matrix_numpy[metric]['matrix'].append(row)

            # save the row in the matrix for the overlapping set
            overlapping_set['overlapping_set'].append(row_dict2)

        # convert to numpy array the matrix
        for metric in overlapping_matrix_numpy.keys():
            overlapping_matrix_numpy[metric]['matrix'] = np.array(overlapping_matrix_numpy[metric]['matrix'])

        # overlapping_tensor[(x_label, y_label)] = overlapping_matrix_dict
        overlapping_tensor[(x_label, y_label)] = overlapping_matrix_numpy
        overlapping_set_tensor[(x_label, y_label)] = overlapping_set

        if save_overlapping_tensor:
            self.ch.save_object(overlapping_tensor,
                                self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_tensor.p")

        if save_intersections:
            self.ch.save_object(overlapping_set_tensor,
                                self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_set.p")

        self.lm.printl(f"{file_name}. compute_overlapping completed.")

    def plot_heatmap_overlapping_matrix(self, concat_matrix_bool=True, plot_heatmap_list=None):
        self.lm.printl(f"{file_name}. plot_heatmap_overlapping_matrix start.")
        overlapping_tensor = self.ch.load_object(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_tensor.p")

        # concat_matrix_bool=True: merge single layer matrices in a unique view if True
        concat_matrix, concat_matrix_n_rows, single_layer_co_actions, community_labels, community_sizes, separate_matrix = self.__concat_single_layer_matrix(
            overlapping_tensor, concat_matrix_bool, plot_heatmap_list)
        self.__plot_concat_matrix(concat_matrix, concat_matrix_n_rows, single_layer_co_actions, community_labels,
                                  community_sizes, 0.7)
        self.__plot_separate_matrix(separate_matrix, 0.7)

        self.lm.printl(f"{file_name}. plot_heatmap_overlapping_matrix completed.")

    def plot_stacked_flux(self, type_aggregation, mid_th=0.5, metric='harmonicMean', plot_heatmap_list=None):
        self.lm.printl(f"{file_name}. plot_stacked_flux {type_aggregation} start.")
        overlapping_tensor = self.ch.load_object(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_tensor.p")

        overlapping_set_tensor = self.ch.load_object(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_set.p")

        try:
            data_flux = self.__instantiation_data_flux(metric, type_aggregation)
        except ValueError as e:
            self.lm.printl(e)
            return
        node_label_dict = {"userId": [], "label": [],
                           'layer': [], 'generic': [], 'com_layer': [], 'com_generic': [], 'communities': [],
                           }
        for (x_label, y_label), overlapping_matrix_collection in overlapping_tensor.items():
            # in the y-axis must be the concatenated matrices of single layer
            # in the x-axis (generic_label) must be multimodal, flattened ecc.
            if x_label in self.available_list_ca and y_label not in self.available_list_ca:  # e.g. co-retweet, multimodal
                type_ca, generic_label = x_label, y_label
            elif x_label not in self.available_list_ca and y_label in self.available_list_ca:  # e.g. multimodal, co-retweet
                type_ca, generic_label = y_label, x_label
            elif x_label in self.available_list_ca and y_label in self.available_list_ca:  # e.g., co-retweet, co-hashtag
                type_ca, generic_label = y_label, x_label
            if plot_heatmap_list is None or (plot_heatmap_list is not None and generic_label in plot_heatmap_list):
                self.lm.printl(f"{file_name}. plot_stacked_flux {type_aggregation} ({x_label}, {y_label}).")
                # select the info on the overlapping from overlapping_set_tensor, which contains the set of users
                # overlapping_tensor contains the matrix of overlapping values
                overlapping_set = overlapping_set_tensor[(x_label, y_label)] # TODO the sets are computed only for multimodal and flat_weighted_sum_louvain

                ca_abbr = co_action_abbreviation_map[type_ca]
                if generic_label in self.available_list_ca:
                    abbr_generic_label = co_action_abbreviation_map[generic_label]
                else:
                    abbr_generic_label = generic_label

                matrix_object = overlapping_matrix_collection[metric]
                matrix = matrix_object['matrix']

                communities_label_x = np.array(matrix_object['x_label'])
                communities_label_y = np.array(matrix_object['y_label'])
                community_size_x = np.array(matrix_object['x_size_community'])
                community_size_y = np.array(matrix_object['y_size_community'])

                intersection_matrix = np.array(overlapping_set['overlapping_set'])
                x_set = np.array(overlapping_set['x_set'])
                y_set = np.array(overlapping_set['y_set'])

                (filter_matrix, filter_community_size_x, filter_community_size_y, filter_communities_label_x,
                 filter_communities_label_y, filter_intersection_matrix, filter_x_set, filter_y_set) = (
                    self.__filter_communities(
                        matrix, community_size_x, community_size_y, communities_label_x, communities_label_y,
                        intersection_matrix, x_set, y_set))

                # HUNGARIAN ALGORITHM MATCHING: find the best couple of communities between the single layer and the multimodal (flattening)
                matches, unmatched_rows, unmatched_cols, total_similarity = self.__max_similarity_match(filter_matrix)

                data_flux, node_label_dict = self.__create_flux_df(data_flux, filter_matrix, filter_community_size_x,
                                                                   filter_community_size_y,
                                                                   filter_communities_label_x,
                                                                   filter_communities_label_y,
                                                                   ca_abbr, abbr_generic_label, type_ca, generic_label,
                                                                   mid_th,
                                                                   type_aggregation, matches, unmatched_rows,
                                                                   unmatched_cols,
                                                                   filter_intersection_matrix, filter_x_set, filter_y_set,
                                                                   node_label_dict)

        data_flux_df = pd.DataFrame(data_flux)
        node_label_df = pd.DataFrame(node_label_dict)

        if type_aggregation == 'communities':
            filename = f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv"
        elif type_aggregation == 'users':
            filename = f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_flux_df.csv"
            self.ch.save_dataframe(node_label_df, f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_th_size_{str(self.community_size_th)}_node_labelling.csv")

        self.ch.save_dataframe(data_flux_df, filename)
        data_flux_df = self.ch.read_dataframe(filename, dtype=dtype)
        self.__plot_flux_df(type_aggregation, data_flux_df, mid_th, metric)

        self.lm.printl(f"{file_name}. plot_stacked_flux {type_aggregation} completed.")

    def combine_single_layer_metrics_communities(self, cda):
        self.lm.printl(f"{file_name}. combine_metrics_communities start.")
        df_list = []
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            # from here I don't have access to the community discovery results of the single layer algorithms. I have
            # access only to the single layer co-actions, until the communities directory
            df = self.ch.read_dataframe(
                f"{dict_path["path_filter_community"]}{repr(cda)}{os.sep}analysis{os.sep}{type_ca}_th_size_{str(self.community_size_th)}_metrics_communities.csv",
                dtype=dtype)
            df['layer'] = type_ca
            df = df.rename(columns={"community": "com_layer"})  # Renaming for consistency
            df_list.append(df)
        # Concatenate all data into a single DataFrame
        combined_df = pd.concat(df_list, ignore_index=True)
        self.ch.save_dataframe(combined_df,
                               f"{self.dm.path_overlapping_analysis}{self.file_prefix}_single_layer_metrics_communities.csv")
        self.lm.printl(f"{file_name}. combine_metrics_communities completed.")

    def combine_coordination_communities(self, cda):
        self.lm.printl(f"{file_name}. combine_coordination_communities start.")
        if cda.get_algorithm_name() in one_layer_algorithm:
            df_list = []
            if self.community_size_th is None:
                th_str = ""
            else:
                th_str = f"th_size_{str(self.community_size_th)}_"
            for type_ca, dict_path in self.dm.dict_path_ca.items():
                # from here I don't have access to the community discovery results of the single layer algorithms. I have
                # access only to the single layer co-actions, until the communities directory
                df = self.ch.read_dataframe(
                    f"{dict_path["path_filter_community"]}{repr(cda)}{os.sep}analysis{os.sep}{type_ca}_{th_str}coordination_communities.csv", dtype=dtype)
                df['layer'] = type_ca
                
                df_list.append(df)
            # Concatenate all data into a single DataFrame
            combined_df = pd.concat(df_list, ignore_index=True)
        else:
            if cda.get_algorithm_name() == 'flat_nw_infomap' or cda.get_algorithm_name() == 'flat_nw_louvain':
                self.lm.printl(f"{file_name}. combine_coordination_communities skipped, algorithm: {cda.get_algorithm_name()}.")
                return
            else:
                combined_df = self.ch.read_dataframe(f"{self.dm.path_community}{repr(cda)}{os.sep}analysis{os.sep}{cda.get_algorithm_name()}_coordination_communities.csv", dtype=dtype)
                combined_df['layer'] = cda.get_algorithm_name()
                
        self.ch.update_dataframe(combined_df, f"{self.dm.path_validation}{self.file_prefix}_coordination_communities.csv", dtype=dtype)
        self.lm.printl(f"{file_name}. combine_coordination_communities completed.")


    def combine_node_metrics(self, cda):
        self.lm.printl(f"{file_name}. combine_node_metrics start, algorithm: {cda.get_algorithm_name()}.")
        # if self.cda.get_algorithm_name() in flatten_algorithm:
        # else:
        if cda.get_algorithm_name() in one_layer_algorithm:
            df_list = []
            for type_ca, dict_path in self.dm.dict_path_ca.items():
                # from here I don't have access to the community discovery results of the single layer algorithms. I have
                # access only to the single layer co-actions, until the communities directory
                df = self.ch.read_dataframe(
                    f"{dict_path["path_filter_community"]}{repr(cda)}{os.sep}analysis{os.sep}{type_ca}_node_metrics.csv",
                    dtype=dtype)
                df = df.rename(columns={"community": "com_layer"})  # Renaming for consistency
                df_list.append(df)

            # Concatenate all data into a single DataFrame
            combined_df = pd.concat(df_list, ignore_index=True)
        
        # if it is a flattened algorithm, I just read and merge the data
        elif cda.get_algorithm_name() in flatten_algorithm:
            combined_df = self.ch.read_dataframe(
                f"{self.dm.path_community}{repr(cda)}{os.sep}analysis{os.sep}{cda.get_algorithm_name()}_node_metrics.csv",
                dtype=dtype)
            combined_df = combined_df.rename(columns={"community": "com_layer"})  # Renaming for consistency
        else:
            self.lm.printl(f"{file_name}. combine_node_metrics skipped, algorithm: {cda.get_algorithm_name()}.")
            return

        self.ch.update_dataframe(combined_df, f"{self.dm.path_overlapping_analysis}{self.file_prefix}_node_metrics.csv",
                                 dtype=dtype)
        self.lm.printl(f"{file_name}. combine_node_metrics completed.")

    # Function to organize starplots by layer and display them in a grid
    def __plot_grid_starplots(self, plot_df, generic, features, normalized, mid_th, metric):
        self.lm.printl(f"{file_name}. __plot_grid_starplots start {generic}.")
        # Get unique layers
        unique_layers = plot_df[plot_df['layer'] != generic]['layer'].unique()

        # Loop through each unique layer and plot the star plots in separate grids
        for layer in unique_layers:
            # Filter rows by layer
            layer_rows = plot_df[plot_df['layer'] == layer]

            # Calculate the number of star plots for the layer
            num_starplots = len(layer_rows)

            # Set the number of columns: maximum 3 or the number of star plots if fewer
            ncols = min(num_starplots, 3)

            # Calculate the number of rows needed for the grid (ceil division)
            nrows = (num_starplots + ncols - 1) // ncols  # This ensures enough rows for all star plots

            # Prepare the figure and axes (adjust the size based on the number of plots)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows),
                                     subplot_kw=dict(polar=True))

            # Check if axes is a 2D array or a single axis
            if nrows == 1 and ncols == 1:
                axes = np.array([axes])  # Convert to a 1D array if it's a single plot
            else:
                axes = axes.flatten()

            # Add central title for the layer group at the top of the figure
            plt.suptitle(f"Generic: {generic} - Layer {layer}", fontsize=15, fontweight='bold', y=1.05)

            # Plot the star plots for this layer
            plot_index = 0
            len_layer_rows = len(layer_rows)
            for j, row in layer_rows.iterrows():
                if plot_index < len(axes):
                    ax = axes[plot_index]
                    row2 = plot_df.loc[(plot_df['layer'] == generic) & (plot_df['com_generic'] == row['com_generic'])]
                    # Concatenate the two selected rows (current row and matching rows with null 'generic')
                    starplot_df = pd.concat([pd.DataFrame([row]), row2], ignore_index=True)
                    if j == len_layer_rows - 1:
                        show_legend = True
                    else:
                        show_legend = True
                    # Plot the starplot on the corresponding axis
                    self.__plot_starplot(starplot_df, features, normalized, ax=ax, show_legend=show_legend)
                    plot_index += 1

            # Remove any unused subplots (axes) if there are fewer plots than grid spaces
            for i in range(plot_index, len(axes)):
                fig.delaxes(axes[i])  # Delete unused axes

            # Adjust layout and spacing
            plt.tight_layout()
            filename = f"{self.dm.path_overlapping_starplot}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_{generic}_{layer}_grid_starplot.png"
            plt.savefig(filename, dpi=800, bbox_inches='tight', pad_inches=0)
            plt.show()

        self.lm.printl(f"{file_name}. __plot_grid_starplots completed {generic}.")

    def __merge_common_metrics(self, metrics_df, common_df, generic):
        # plot_df1 selects all rows where 'layer' is not equal to 'generic' and merges with metrics_df
        plot_df1 = common_df[common_df['layer'] != generic].merge(metrics_df, on=["layer", "com_layer"], how="inner")

        # plot_df2 selects all rows where 'layer' is equal to 'generic' and merges with plot_df1
        # with respect to 'generic' and 'com_generic'.
        plot_df2 = common_df[common_df['layer'] == generic].merge(
            plot_df1[["generic", "com_generic"]].drop_duplicates(),
            on=["generic", "com_generic"], how="inner")
        # Merge plot_df with common_df to add matching pairs
        plot_df2 = plot_df2.merge(metrics_df, on=["layer", "com_layer"], how="inner")
        plot_df = pd.concat([plot_df1, plot_df2], ignore_index=True)

        return plot_df

    def __get_perplexity(self, n_rows):
        if n_rows > 10:
            perplexity = 10
        else:
            perplexity = n_rows - 1
        return perplexity

    def __get_n_neighbors(self, n_rows):
        if n_rows > 10:
            n_neighbors = 10
        else:
            n_neighbors = n_rows - 1
        return n_neighbors

    def __get_labelled_metrics_df(self, flux_selected_df, node_metrics_generic_df, node_metrics_layer_df, generic,
                                  layer):
        label_lost_df = flux_selected_df[flux_selected_df['label'] == 'lost'][['com_layer']]
        lost_communities_metrics = node_metrics_layer_df.merge(label_lost_df, how='inner', left_on='community',
                                                               right_on='com_layer')
        lost_communities_metrics.drop(columns=['com_layer'], inplace=True)
        lost_communities_metrics['label'] = 'lost'

        label_gained_df = flux_selected_df[flux_selected_df['label'] == 'gained'][['com_generic']]
        # if generic == 'multimodal', i must filter the node_metrics_generic_df by layer, since you can have the same
        # community in different layers. So, we consider, e.g. co-retweet vs the subgraph composed by the nodes in the
        # communities of the multimodal in the layer co-retweet
        if generic == 'multimodal':
            node_metrics_generic_df = node_metrics_generic_df[node_metrics_generic_df['layer'] == layer]
        elif generic in flatten_algorithm:
            pass
        gained_communities_metrics = node_metrics_generic_df.merge(label_gained_df, how='inner', left_on='community',
                                                                   right_on='com_generic')
        gained_communities_metrics.drop(columns=['com_generic'], inplace=True)
        gained_communities_metrics['label'] = 'gained'

        labelled_metrics_df = pd.concat([lost_communities_metrics, gained_communities_metrics])

        return labelled_metrics_df

    def __plot_KDE(self, metrics_node_to_compute, labelled_metrics_df, generic, layer, metric):
        self.lm.printl(f"{file_name}. __plot_KDE start ({generic}, {layer}).")
        # Kernel Density Plot
        plt.figure(figsize=(8, 6))
        # select the default palette of seaborn and take the second and the third color (red and green)
        default_palette = sns.color_palette()
        palette = [default_palette[1], default_palette[2]]  # red: lost, green: gained
        labels = labelled_metrics_df['label'].unique()  # ['lost', 'gained']

        # for each couple of node metrics, plot the KDE (except role, which is a categorical variable)
        for col1, col2 in combinations(metrics_node_to_compute, 2):
            if col1 != 'role' and col2 != 'role':
                self.lm.printl(f"Start KDE plot: {col1}, {col2}")
                for l, color in zip(labels, palette):
                    subset = labelled_metrics_df[labelled_metrics_df['label'] == l]
                    try:
                        sns.kdeplot(data=subset, x=col1, y=col2, fill=True, alpha=0.5, label=l, color=color)
                    except Exception as e:
                        try:
                            self.lm.printl(
                                f"{file_name}. __plot_KDE error {col1}, {col2} label={l} {str(e)}. set levels=7")
                            sns.kdeplot(
                                data=subset, x=col1, y=col2,
                                fill=True, alpha=0.5, label=l, color=color, levels=7,
                                clip=((0, 0.002), (0, 0.002)),
                                bw_adjust=1.2
                            )
                        except Exception as e:
                            self.lm.printl(
                                f"{file_name}. __plot_KDE error {col1}, {col2} label={l} {str(e)}. set levels=7, clip_percentiles=(1, 91)")
                            # x_min, x_max = labelled_metrics_df[col1].min(), labelled_metrics_df[col1].max()
                            # y_min, y_max = labelled_metrics_df[col2].min(), labelled_metrics_df[col2].max()
                            clip_percentiles = (1, 91)
                            # Compute clipping ranges based on percentiles
                            x_min, x_max = np.percentile(subset[col1], clip_percentiles)
                            y_min, y_max = np.percentile(subset[col2], clip_percentiles)

                            clip = ((x_min, x_max), (y_min, y_max))
                            sns.kdeplot(
                                data=subset, x=col1, y=col2,
                                fill=True, alpha=0.5, label=l, color=color, levels=7,
                                clip=clip,
                                bw_adjust=1.2
                            )
                # Plot the legend lines and labels, it doesn't work simply with legend
                handles = [Line2D([0], [0], color=color) for color in palette]
                labels = [l for l in labels]
                plt.legend(handles, labels, title='label')

                plt.title(f'Kernel Density Plot {generic}/{layer}\n{col1} vs. {col2}')
                plt.savefig(
                    f"{self.dm.path_overlapping_KDE_plot}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_{generic}_{layer}_{col1}_{col2}_KDE.png")
                plt.show()

        self.lm.printl(f"{file_name}. __plot_KDE completed ({generic}, {layer}).")

    def __plot_distribution(self, metrics_node_to_compute, labelled_metrics_df, generic, layer, metric):
        self.lm.printl(f"{file_name}. __plot_distribution start ({generic}, {layer}).")
        # Distribution Plot for each node metrics
        for col in metrics_node_to_compute:
            if col != 'role':
                plt.figure(figsize=(8, 6))
                for l in labelled_metrics_df['label'].unique():
                    subset = labelled_metrics_df[labelled_metrics_df['label'] == l]
                    sns.kdeplot(data=subset, x=col, fill=True, alpha=0.5, label=l)
                plt.title(f'Distribution of {col}')
                plt.legend(title='Label', loc='best')
                plt.xlabel(col)
                plt.ylabel('Density')
                plt.savefig(
                    f"{self.dm.path_overlapping_distribution_plot}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_{generic}_{layer}__{col}_distribution.png")
                plt.show()
        self.lm.printl(f"{file_name}. __plot_distribution completed ({generic}, {layer}).")


    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    def plot_single_layer_metrics(self, visualization, type_visualization_starplot='single', normalized=True,
                                  features=["size", "density", "avg_weight", "conductance", "avg_degree",
                                            "avg_clustering",
                                            "assortativity"],
                                  mid_th=0.5, metric='harmonicMean'):
        self.lm.printl(f"{file_name}. plot_single_layer_metrics ({visualization}) start.")

        flux_df = self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv",
            dtype=dtype)
        metrics_df = self.ch.read_dataframe(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_single_layer_metrics_communities.csv",
            dtype=dtype)

        # Create a mapping from generic values to URLs for nice formatting
        # flux_df['layer'] = flux_df['layer'].replace(url_map)
        # flux_df['generic'] = flux_df['generic'].replace(url_map)
        # metrics_df['layer'] = metrics_df['layer'].replace(url_map)

        flux_df['layer'] = flux_df['layer'].replace(co_action_column_print)
        flux_df['generic'] = flux_df['generic'].replace(co_action_column_print)
        metrics_df['layer'] = metrics_df['layer'].replace(co_action_column_print)


        if visualization == 't_sne':
            perplexity_all = self.__get_perplexity(metrics_df.shape[0])
            self.lm.printl(f"All dataset. Perplexity: {str(perplexity_all)}")
            # Extract features from combined_df for t-SNE
            X = metrics_df[features]
            # Apply t-SNE on combined_df
            tsne = TSNE(n_components=2, perplexity=perplexity_all, random_state=0, n_jobs=1)
            X_tsne = tsne.fit_transform(X)
            x_all = 'x_tsne_all'
            y_all = 'y_tsne_all'
            metrics_df[[x_all, y_all]] = X_tsne
        elif visualization == 'umap':
            n_neighbors_all = self.__get_n_neighbors(metrics_df.shape[0])
            X = metrics_df[features]
            # Initialize and fit UMAP
            umap_model = umap.UMAP(n_neighbors=n_neighbors_all, n_components=2, metric='manhattan', random_state=42)
            X_umap = umap_model.fit_transform(X)
            x_all = 'x_umap_all'
            y_all = 'y_umap_all'
            metrics_df[[x_all, y_all]] = X_umap
        elif visualization == 'pca':
            pca = PCA(n_components=2)
            X = metrics_df[features]
            X_pca = pca.fit_transform(X)
            x_all = 'x_pca_all'
            y_all = 'y_pca_all'
            metrics_df[[x_all, y_all]] = X_pca

        cosine_results = []
        for generic in metrics_df['layer'].unique():
            metrics_generic_df = metrics_df[metrics_df['layer'] == generic]
            common_df = flux_df.loc[(flux_df['label'] == 'common') & (flux_df['generic'] == generic) & (
                    flux_df['layer'] != 'generic')].copy()
            common_df['com_layer'] = common_df['com_layer'].astype('int')
            common_df['com_generic'] = common_df['com_generic'].astype('int')

            if visualization == "t_sne" or visualization == "umap" or visualization == "pca":
                plot_df = self.__merge_common_metrics(metrics_df, common_df, generic)
                if plot_df.shape[0] > 0:
                    if visualization == "t_sne":
                        perplexity_generic = self.__get_perplexity(plot_df.shape[0])
                        self.lm.printl(f"Generic: {generic}. T-SNE(perplexity: {str(perplexity_generic)})")
                        # Extract features from plot_df for t-SNE
                        X = plot_df[features]
                        # Apply t-SNE on combined_df
                        tsne = TSNE(n_components=2, perplexity=perplexity_generic, random_state=0, n_jobs=1)
                        X_tsne = tsne.fit_transform(X)
                        x_generic = 'x_tsne_generic'
                        y_generic = 'y_tsne_generic'
                        plot_df[[x_generic, y_generic]] = X_tsne
                        suffix = f"{self.dm.path_overlapping_t_sne_plot}{self.file_prefix}_{generic}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}"
                        filename_all = f"{suffix}_tsne_embedding_all_perplexity_{str(perplexity_all)}.png"
                        filename_generic = f"{suffix}_tsne_embedding_generic_perplexity_{str(perplexity_generic)}.png"
                        title = f"t-SNE Visualization of common communities of {generic}"
                    elif visualization == "umap":
                        n_neighbors_generic = self.__get_n_neighbors(plot_df.shape[0])
                        self.lm.printl(f"Generic: {generic}. UMAP(n_neighbors: {str(n_neighbors_generic)})")
                        suffix = f"{self.dm.path_overlapping_umap_plot}{self.file_prefix}_{generic}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}"
                        filename_all = f"{suffix}_umap_embedding_all_neighbors_{str(n_neighbors_all)}.png"
                        title = f"UMAP Visualization of common communities of {generic}"
                        if plot_df.shape[0] > 2:
                            X = plot_df[features]
                            # Initialize and fit UMAP
                            umap_model = umap.UMAP(n_neighbors=n_neighbors_generic, n_components=2, metric='manhattan',
                                                   random_state=42)
                            X_umap = umap_model.fit_transform(X)
                            x_generic = 'x_umap_generic'
                            y_generic = 'y_umap_generic'
                            plot_df[[x_generic, y_generic]] = X_umap
                            filename_generic = f"{suffix}_umap_embedding_generic_neighbors_{str(n_neighbors_generic)}.png"
                        else:
                            self.lm.printl(f"Generic: {generic}. Not enough data points for UMAP.")
                    elif visualization == "pca":
                        self.lm.printl(f"Generic: {generic}. PCA.")
                        pca = PCA(n_components=2)
                        X = plot_df[features]
                        X_pca = pca.fit_transform(X)
                        x_generic = 'x_pca_generic'
                        y_generic = 'y_pca_generic'
                        plot_df[[x_generic, y_generic]] = X_pca
                        suffix = f"{self.dm.path_overlapping_pca_plot}{self.file_prefix}_{generic}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}"
                        filename_all = f"{suffix}_pca_embedding_all.png"
                        filename_generic = f"{suffix}_pca_embedding_generic.png"
                        title = f"PCA Visualization of common communities of {generic}"

                    self.__scatterplot(features, plot_df, x_all, y_all, generic, title, filename_all)
                    # if there is only 2 points in plot_df, n_neighbors=1, so umap will not work for the local case
                    if visualization != 'umap' or (visualization == 'umap' and plot_df.shape[0] > 2):
                        self.__scatterplot(features, plot_df, x_generic, y_generic, generic, title, filename_generic)
                else:
                    self.lm.printl(f"Empty dataframe for {generic}. No plot will be generated.")
            elif visualization == "starplot":
                if normalized:
                    # Normalize the data (min-max scaling) for the relevant features
                    scaler = MinMaxScaler()
                    features_norm = ['norm_' + f for f in features]
                    # Apply scaling to the features columns
                    metrics_df[features_norm] = scaler.fit_transform(metrics_df[features])

                plot_df = self.__merge_common_metrics(metrics_df, common_df, generic)
                if plot_df.shape[0] > 0:
                    self.__plot_legend_starplot(plot_df, generic)
                    if type_visualization_starplot == 'single':
                        for _, row in plot_df[plot_df['layer'] != generic].iterrows():
                            row2 = plot_df.loc[
                                (plot_df['layer'] == generic) & (plot_df['com_generic'] == row['com_generic'])]
                            # Concatenate the two selected rows (current row and matching rows with null 'generic')
                            starplot_df = pd.concat([pd.DataFrame([row]), row2], ignore_index=True)

                            self.__plot_starplot(starplot_df, features, normalized, metric=metric, mid_th=mid_th)
                    elif type_visualization_starplot == 'grid':
                        self.__plot_grid_starplots(plot_df, generic, features, normalized, mid_th, metric)
                else:
                    self.lm.printl(f"Empty dataframe for {generic}. No plot will be generated.")
            elif visualization == 'cosine_similarity':
                if normalized:
                    # Normalize the data (min-max scaling) for the relevant features
                    scaler = MinMaxScaler()
                    features_norm = ['norm_' + f for f in features]
                    # Apply scaling to the features columns
                    metrics_df[features_norm] = scaler.fit_transform(metrics_df[features])
                    selected_features = features_norm
                else:
                    selected_features = features
                plot_df = self.__merge_common_metrics(metrics_df, common_df, generic)
                for _, row in plot_df[plot_df['layer'] != generic].iterrows():
                    row2 = plot_df.loc[
                        (plot_df['layer'] == generic) & (plot_df['com_generic'] == row['com_generic'])]
                    # Concatenate the two selected rows (current row and matching rows with null 'generic')
                    starplot_df = pd.concat([pd.DataFrame([row]), row2], ignore_index=True)
                    vector1 = starplot_df[selected_features].values[0].reshape(1, -1)
                    vector2 = starplot_df[selected_features].values[1].reshape(1, -1)

                    # Compute cosine similarity
                    cosine_sim = cosine_similarity(vector1, vector2)[0][0]

                    com_layer = str(int(starplot_df.iloc[0]['com_layer']))
                    com_generic = str(int(starplot_df.iloc[0]['com_generic']))
                    layer = starplot_df.iloc[0]['layer']
                    # Append result to the results dataframe
                    cosine_results.append(
                        {'com_layer': com_layer,
                         'com_generic': com_generic,
                         'generic': generic,
                         'layer': layer,
                         'cosine_similarity': cosine_sim}
                    )
        if visualization == 'cosine_similarity':
            cosine_results_df = pd.DataFrame(cosine_results)
            self.ch.save_dataframe(cosine_results_df,
                                   self.dm.path_cosine_similarity + f"{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_normalized_{str(normalized)}_single_layer_metrics_cosine_similarity.csv")
        self.lm.printl(f"{file_name}. plot_single_layer_metrics ({visualization})completed.")

    def plot_barchart_cosine_similarity(self, generic='co-retweet', normalized=True, metric='harmonicMean', mid_th=0.5):
        self.lm.printl(f"{file_name}. plot_barchart_cosine_similarity start.")
        cosine_df = self.ch.read_dataframe(self.dm.path_cosine_similarity + f"{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_normalized_{str(normalized)}_single_layer_metrics_cosine_similarity.csv", dtype=dtype)
        
        generic = 'RTW'
        df = cosine_df[cosine_df['generic']==generic]
        
        # --- darker version of each color ---
        dark_palette = {k: self.__darken(v, 0.7) for k, v in color_dict2.items()}

        plt.figure(figsize=(9, 6), dpi=300)

        xpos = []
        colors = []
        values = []
        tick_labels = []
        counter = 0

        block_centers = {}
        layer_order = ["RPL", "URL", "MEN", "HST"]

        ordered_layers = sorted(df["layer"].unique())
        # keep only layers that appear in the dataframe, preserving the desired order
        ordered_layers = [l for l in layer_order if l in df["layer"].unique()]

        # --------------------------------------------------------------
        # Build bar-by-bar structure (like your generic/label plot)
        # --------------------------------------------------------------
        for layer in ordered_layers:

            gdf = df[df["layer"] == layer]
            vals = np.sort(gdf["cosine_similarity"].values)

            start = counter
            # Add bars
            for v in vals:
                xpos.append(counter)
                colors.append(color_dict2[layer])
                values.append(v)
                tick_labels.append(layer)
                counter += 1

            # Compute center of block
            if len(vals) > 0:
                block_centers[layer] = np.mean(range(start, start + len(vals)))
            else:
                block_centers[layer] = counter

            counter += 1   # spacing between blocks

        # --------------------------------------------------------------
        # Plot bars
        # --------------------------------------------------------------
        plt.bar(x=xpos, height=values, color=colors, width=0.8, zorder=2)

        # --------------------------------------------------------------
        # Horizontal mean line + annotation (on top)
        # --------------------------------------------------------------
        for layer in ordered_layers:

            gdf = df[df["layer"] == layer]
            vals = gdf["cosine_similarity"].values
            if len(vals) == 0:
                continue

            mean_val = np.mean(vals)
            center = block_centers[layer]

            # Line
            plt.axhline(
                y=mean_val,
                color=dark_palette[layer],
                linestyle='--',
                linewidth=1,
                zorder=999,
            )
            offset = self.__get_offset_cosine_similarity(generic, layer)
            center_offset = center + offset
            # Annotation
            plt.text(
                center_offset,
                mean_val,
                f"{mean_val:.2f}",
                ha='center', va='bottom',
                fontsize=20,
                color=dark_palette[layer],
                zorder=1000
            )

        # --------------------------------------------------------------
        # Formatting
        # --------------------------------------------------------------
        plt.grid(axis='y', linestyle='--', linewidth=0.6, color='lightgray', zorder=0)

        plt.xticks(
            ticks=[block_centers[layer] for layer in ordered_layers],
            labels=ordered_layers,
            fontsize=16
        )

        plt.xlabel("")
        plt.ylabel("")
        plt.title("")

        plt.tight_layout()
        plt.savefig(f"{self.dm.path_cosine_similarity}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_normalized_{str(normalized)}_single_layer_metrics_cosine_similarity_barchart_{generic}.png", dpi=800)
        plt.show()

        self.lm.printl(f"{file_name}. plot_barchart_cosine_similarity completed.")

    def compute_single_layer_NMI(self):
        self.lm.printl(f"{file_name}. compute_single_layer_NMI start.")
        df_x = self.ch_x.read_dataframe(self.dm_x.path_user_dataframe + "com_df.csv", dtype=dtype)
        df_y = self.ch_y.read_dataframe(self.dm_y.path_user_dataframe + "com_df.csv", dtype=dtype)

        x_label, y_label, df_x, df_y = self.__get_labels_reordered(df_x, df_y)
        if x_label not in self.available_list_ca and y_label not in self.available_list_ca:
            self.lm.printl(
                f"{file_name}. compute_single_layer_NMI. Both layers must be single layers. No multimodal layer.")
            return
        else:
            self.lm.printl(f"{file_name}. compute_single_layer_NMI start ({x_label}, {y_label}).")
            # FROM NOW ON x_label is the multimodal label, y_label is the one-layer label

            if self.community_size_th is not None:
                df_x = self.__filter_community_size(df_x)
                df_y = self.__filter_community_size(df_y)
                self.file_prefix = f"{self.file_prefix}_th_{str(self.community_size_th)}"

            # Check if df_x or df_y are empty after filtering
            if len(df_x) == 0:
                self.lm.printl(
                    f"{file_name}. compute_single_layer_NMI. {x_label} dataframe is empty after filtering by community size threshold={self.community_size_th}.")
                return
            if len(df_y) == 0:
                self.lm.printl(
                    f"{file_name}. compute_single_layer_NMI. {y_label} dataframe is empty after filtering by community size threshold={self.community_size_th}.")
                return

            common_nodes = set(df_x['userId']).intersection(set(df_y['userId']))
            exclusive_x = set(df_x['userId']) - set(df_y['userId'])
            exclusive_y = set(df_y['userId']) - set(df_x['userId'])

            # Step 2: Add unique singleton communities for exclusive nodes
            # Determine the maximum group label already assigned in both com_x and com_y
            max_label_x = df_x['group'].max()
            max_label_y = df_y['group'].max()

            # Start assigning unique labels from the maximum of both
            max_label = max(max_label_x, max_label_y)
            print("max_label: ", max_label)
            # Create a new group for each exclusive node in com_x and com_y
            # Assign unique group labels for exclusive nodes in com_x
            singleton_x = pd.DataFrame(
                {'userId': list(exclusive_x), 'group': range(max_label + 1, max_label + 1 + len(exclusive_x))})
            df_y = pd.concat([df_y, singleton_x], ignore_index=True)

            # Update the max_label to account for newly assigned singleton labels in com_x
            max_label += len(exclusive_x)

            # Assign unique group labels for exclusive nodes in com_y
            singleton_y = pd.DataFrame(
                {'userId': list(exclusive_y), 'group': range(max_label + 1, max_label + 1 + len(exclusive_y))})
            df_x = pd.concat([df_x, singleton_y], ignore_index=True)

            # Step 3: Prepare the labels for NMI calculation
            # We now have both dataframes with all nodes, including the singleton nodes.
            # Create a mapping from userId to community group for both dataframes
            labels_x = df_x.set_index('userId')['group'].to_dict()
            labels_y = df_y.set_index('userId')['group'].to_dict()

            # Ensure both partitions have labels for all nodes (even the singleton nodes)
            all_nodes = list(set(labels_x.keys()).union(labels_y.keys()))

            # Generate the community labels for each node
            # In scikit learn implementation of NMI, you pass the list of community labels for each node, but the order of
            # the labels must be the same for both partitions. So we need to ensure that the order of the labels is the same.
            # Loop through all nodes and get the community label for each node in both partitions (none of them should be -1),
            # otherwise the node is missing in one of the partitions, but I added the singleton nodes, so it should not happen.
            community_labels_x = [labels_x.get(node, -1) for node in all_nodes]  # -1 for missing nodes in com_x
            community_labels_y = [labels_y.get(node, -1) for node in all_nodes]  # -1 for missing nodes in com_y

            # Step 4: Calculate NMI
            nmi_score = normalized_mutual_info_score(community_labels_x, community_labels_y)

            data = {
                'layer_x': [x_label],
                'layer_y': [y_label],
                'NMI_score': [nmi_score]
            }
            results_df = pd.DataFrame(data)
            self.ch.update_dataframe(results_df,
                                     self.dm.path_overlapping_NMI + f"{self.file_prefix}_single_layer_NMI.csv",
                                     dtype=dtype)

            self.lm.printl(f"{file_name}. compute_single_layer_NMI completed.")

    def plot_heatmap_single_layer_NMI(self):
        self.lm.printl(f"{file_name}. plot_heatmap_single_layer_NMI start.")

        if self.community_size_th is not None:
            prefix = f"{self.file_prefix}_th_{str(self.community_size_th)}"
        else:
            prefix = self.file_prefix

        df = self.ch.read_dataframe(
            self.dm.path_overlapping_NMI + f"{prefix}_single_layer_NMI.csv",
            dtype=dtype)
        df['layer_x'] = df['layer_x'].replace(co_action_column_print)
        df['layer_y'] = df['layer_y'].replace(co_action_column_print)

        layer_order = list(co_action_column_print.values())
        # Set the order for the layer_x and layer_y columns
        df['layer_x'] = pd.Categorical(df['layer_x'], categories=layer_order, ordered=True)
        df['layer_y'] = pd.Categorical(df['layer_y'], categories=layer_order, ordered=True)

        # Pivot the dataframe to create a matrix format
        heatmap_data = df.pivot(index='layer_x', columns='layer_y', values='NMI_score')

        # Create a mask for the upper triangular part of the matrix excluding the diagonal
        mask = np.triu(np.ones_like(heatmap_data, dtype=bool), k=1)
        
        # Reverse the colormap
        reversed_cmap = sns.color_palette("viridis", as_cmap=True).reversed()

        # Plot the heatmap
        plt.figure(figsize=(8, 6.5))  # Adjust the size of the heatmap
        ax = sns.heatmap(
            heatmap_data,
            mask=mask,
            annot=True,
            cmap=reversed_cmap, cbar=True,
            linewidths=0.5,
            linecolor='white',
            annot_kws={'size': 16},
            cbar_kws={'label': 'NMI score'},
            vmin=0,  # Set the minimum value of the color scale
            vmax=0.4  # Set the maximum value of the color scale
        )

        # Overlay black rectangles with labels for the diagonal cells
        for i in range(len(heatmap_data)):
            # Extract the diagonal value
            diagonal_value = heatmap_data.iloc[i, i]

            # Ensure the value is numeric and properly formatted
            if pd.notna(diagonal_value):  # Check if the value is not NaN
                diagonal_value = int(diagonal_value)  # Convert to float if necessary

                # Draw a black rectangle with a white border
                ax.add_patch(plt.Rectangle((i+0.01, i+0.01), 0.978, 0.978, fill=True, color='black', edgecolor='white', linewidth=0.8))

                # Add the diagonal label
                ax.text(
                    i + 0.5, i + 0.5,  # Center the text in the cell
                    f"{diagonal_value}",  # Format the value as one decimal point
                    color='white',  # White text for contrast
                    ha='center', va='center', fontsize=16
                )

        # plt.title('Heatmap of NMI Scores between Layers')
        # Remove axis names
        ax.set_xlabel("")  # Remove x-axis label
        ax.set_ylabel("")  # Remove y-axis label
        # Set font size for the color bar label
        ax.figure.axes[-1].yaxis.label.set_size(16)  # Adjust the label font size
        ax.figure.axes[-1].tick_params(labelsize=12)  # Adjust the tick font size
        # Optionally, adjust the tick label size
        ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick labels
        ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick labels

        plt.savefig(self.dm.path_overlapping_NMI + f"{prefix}_single_layer_NMI_heatmap.png", dpi=dpi,
                    bbox_inches='tight', pad_inches=0)
        plt.show()
        self.lm.printl(f"{file_name}" + f". plot_heatmap_single_layer_NMI completed")

    def plot_node_metrics_gained_lost(self, metrics_node_to_compute, mid_th=0.5, metric='harmonicMean',
                                      th_size_metrics=None):
        """
        Plot the distribution of the node metrics for the gained and lost communities between the multimodal and the single layer.
        :param metrics_node_to_compute: List of node metrics to compute.
        :param mid_th: Threshold to define the gained and lost communities.
        :param metric: metric to consider for the flux_df, e.g., harmonicMean, Jaccard, etc.
        :param th_size_metrics: Threshold to filter the communities in the node metrics files.
        The default is None, which means that no filtering is applied. Indeed, we computed all the metrics on the whole set of communities.
        However, since flux_df takes into account only the communities with size>=self.community_size_th, the plot are made only for these communities.
        :return:
        """
        generic, layer = self.__get_labels_reordered()
        self.lm.printl(f"{file_name}. plot_node_metrics_gained_lost start ({generic}, {layer}).")

        # flux_df contains the information about the communities (common, gained, lost) between the generic and the layer
        flux_df = self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv",
            dtype=dtype)

        # Read the node metrics for the multimodal and the single layer
        file_multimodal = self.dm_x.path_community_analysis + f"{self.chm_x.get_cda().get_algorithm_name()}_th_size_{str(th_size_metrics)}_node_metrics_communities.csv"
        file_layer = self.dm_y.path_community_analysis + f"{layer}_th_size_{str(th_size_metrics)}_node_metrics_communities.csv"
        node_metrics_generic_df = self.ch_x.read_dataframe(file_multimodal, dtype=dtype)
        node_metrics_layer_df = self.ch_y.read_dataframe(file_layer, dtype=dtype)

        # select the communities info for the couple, e.g, multimodal/co-retweet
        flux_selected_df = flux_df.loc[(flux_df['generic'] == generic) & (flux_df['layer'] == layer)].copy()

        labelled_metrics_df = self.__get_labelled_metrics_df(flux_selected_df, node_metrics_generic_df,
                                                             node_metrics_layer_df, generic, layer)

        # KDE plots
        self.__plot_KDE(metrics_node_to_compute, labelled_metrics_df, generic, layer, metric)
        # distribution plots
        self.__plot_distribution(metrics_node_to_compute, labelled_metrics_df, generic, layer, metric)

        self.lm.printl(f"{file_name}. plot_node_metrics_gained_lost. completed ({generic}, {layer}).")



    def plot_boxplot_metrics_gained_lost_nodes(self):
        self.lm.printl(f"{file_name}. plot_boxplot_metrics_gained_lost_nodes start.")
        flux_df = self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_th_size_{str(self.community_size_th)}_node_labelling.csv",
            dtype=dtype)
        node_metrics_df = self.ch.read_dataframe(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_node_metrics.csv",
            dtype=dtype)
        node_metrics_df = node_metrics_df.rename(columns={'nodeId': 'userId'})  # Rename nodeId to userId for consistency
        self.lm.printl(f"Node metrics shape: {str(node_metrics_df.shape)}")
        self.lm.printl(f"Flux df shape: {str(flux_df.shape)}")

        # Initialize results DataFrame
        results_pvalue = []
        
        for generic in flux_df['generic'].unique():
            generic_flux_df = flux_df[flux_df['generic'] == generic]
            for layer in generic_flux_df['layer'].unique():
                self.lm.printl(f"Plot boxplot for generic: {generic}, layer: {layer}. prefix: {self.file_prefix}")
                
                generic_layer_flux_df = generic_flux_df[generic_flux_df['layer'] == layer]
                node_metrics_layer_df = node_metrics_df[node_metrics_df['layer'] == layer]
                merge_df = generic_layer_flux_df.merge(node_metrics_layer_df, on=["userId", "layer"], how="inner")
                merge_df.drop(columns=['com_layer_x', 'com_layer_y', 'communities', 'com_generic'], inplace=True)

                # Create a single figure for all metrics related to this layer
                plt.figure(figsize=(7, 3.5))  # Adjusted figure size to be smaller

                # Define specific scientific notation preferences for each metric
                scientific_notation_limits = {
                    'degree_centrality': (-2, -2),  # Always show as 10^-2
                    'eigenvector_centrality': (-2, -2),  # Always show as 10^-2
                    'local_clustering_coefficient': None,  # No scientific notation, show as is (10^0)
                    'page_rank': (-4, -4)  # Always show as 10^-4
                }
                
                # Add statistical annotations
                # Define the 4-level threshold map
                pvalue_thresholds = [
                    (1e-3, "***"),  # p ≤ 0.001
                    (1e-2, "**"),  # p ≤ 0.01
                    (5e-2, "*"),  # p ≤ 0.05
                    (1, "ns")  # p > 0.05
                ]

                # DEFINE ORDER OF THE PAIRS
                # Check that pairs order is the same used in Annotator. This is crucial for the dataframe results.
                pairs = [
                        ("common", "gained"),
                        ("lost", "common"),
                        ("lost", "gained"),
                    ]
                # A label can miss in data, so we need to get the unique labels from the data and
                # remove the missing labels 
                label_order = ['lost', 'common', 'gained']  # Define the order of the labels for consistency
                unique_labels = merge_df['label'].unique().tolist()
                label_order = [label for label in label_order if label in unique_labels]
                pairs = [(l1, l2) for l1, l2 in pairs if l1 in unique_labels and l2 in unique_labels]
                self.lm.printl(f"Labels in data: {unique_labels}, pairs: {pairs}")
                
                if len(pairs) == 0:
                    self.lm.printl(f"Warning: No valid pairs for statistical test in generic: {generic}, layer: {layer}, prefix: {self.file_prefix}. Skipping.")
                    continue
                
                for i, metric in enumerate(
                        ['degree_centrality', 'eigenvector_centrality', 'local_clustering_coefficient', 'page_rank']):
                    # Plot each boxplot in a grid layout
                    ax = plt.subplot(1, 4, i + 1)  # Create a subplot for each metric
                    sns.boxplot(
                        x='label',
                        y=metric,
                        data=merge_df,
                        palette=palette,
                        order=label_order,  # Order the boxes for consistency
                        hue='label',  # Assign label to hue for proper coloring
                        dodge=False,  # Ensure boxes don't overlap since we're using hue
                        showfliers=False,  # Show outliers,
                        medianprops={'color': 'black', 'linewidth': 1}  # Median line color and thickness
                    )

                    # Adjust opacity of the box colors
                    #             for patch in ax.patches:  # Loop through each box
                    #                 patch.set_alpha(0.5)  # Set the opacity (0.0 = transparent, 1.0 = opaque)
                    # Thicken the median lines
                    # median_lines = [line for line in ax.lines if
                    #                 line.get_linestyle() == '-']  # Median lines have solid line style '-'
                    # for line in median_lines:
                    #     line.set_linewidth(1.5)  # Adjust thickness as desired

                    # plt.title(f'{metric}', fontsize=10)

                    if metric != 'local_clustering_coefficient':
                        # Set scientific notation limits for the y-axis based on the metric
                        sci_limits = scientific_notation_limits[metric]
                        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=sci_limits)

                    # Adjust the size of y-axis tick labels
                    ax.tick_params(axis='y', labelsize=14)  # Increase the font size of y-axis tick labels

                    # Remove labels from axes
                    ax.set_xlabel('')  # Remove x-axis label
                    ax.set_ylabel(metric.replace('_', ' '), fontsize=14)  # Remove y-axis label
                    plt.xticks('')  # Remove x-axis label


                    self.lm.printl("Warning: Check that pairs order is the same used in Annotator. This is crucial for the dataframe results.")

                    
                    annotator = Annotator(ax, pairs, data=merge_df, x='label', y=metric, order=label_order)
                    #             annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
                    annotator.configure(test='Brunner-Munzel', text_format='star', pvalue_thresholds=pvalue_thresholds, loc='inside')
                    # annotator.apply_and_annotate()
                    # Apply test and retrieve p-values & statistics
                    _, test_results = annotator.apply_test().annotate()
                    stat_results = [(res.data.stat_value, res.data.pvalue) for res in test_results]

                    # Extract p-values from `annotator` correctly
                    for (label1, label2), test_result in zip(pairs, stat_results):
                        p_value = test_result[1]  # Second element in the tuple is the p-value

                        # # Assign significance level
                        # if p_value <= 0.001:
                        #     significance = "***"
                        # elif p_value <= 0.01:
                        #     significance = "**"
                        # elif p_value <= 0.05:
                        #     significance = "*"
                        # else:
                        #     significance = "ns"

                        # 4-level significance assignment
                        if p_value <= 0.001:
                            significance = "***"
                        elif p_value <= 0.01:
                            significance = "**"
                        elif p_value <= 0.05:
                            significance = "*"
                        else:
                            significance = "ns"

                        results_pvalue.append({
                            'generic': generic,
                            'layer': layer,
                            'metric': metric,
                            'label1': label1,
                            'label2': label2,
                            'p_value': p_value,
                            'significance': significance
                        })
                # Adjust spacing between subplots
                plt.subplots_adjust(wspace=1)  # Decrease the horizontal space between subplots (default is 0.4)

                # Add a title for the whole figure and adjust layout
                #         plt.suptitle(f'{generic}\nLayer: {layer}', fontsize=10)
                # plt.tight_layout(pad=1)  # Make sure subplots fit into the figure area
                plt.savefig(f"{self.dm.path_node_metrics_boxplot}{self.file_prefix}_boxplot_node_metrics_{generic}_{layer}.png",
                            dpi=dpi, bbox_inches='tight',
                            pad_inches=0.05)
                self.lm.printl(f"{file_name}. {self.dm.path_node_metrics_boxplot}{self.file_prefix}_boxplot_node_metrics_{generic}_{layer}.png saved.")
                plt.show()

        # Create a separate figure for the legend
        legend_fig = plt.figure(figsize=(6, 2))  # Adjust size as needed
        legend_ax = legend_fig.add_subplot(111)

        # Manually create legend handles and labels
        handles = [plt.Line2D([0], [0], color=palette[key], lw=6) for key in palette.keys()]
        labels = list(palette.keys())

        # Add legend to the separate figure
        legend_ax.legend(
            handles, labels,
            # title="Labels",
            loc='center',
            frameon=False,  # Remove the legend box
            ncol=len(labels)  # Arrange in a single row
        )
        legend_ax.axis('off')  # Turn off the axis for the legend figure
        # Save or show the legend figure
        legend_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove extra space

        # Save or show the legend figure
        plt.tight_layout()
        plt.savefig(f"{self.dm.path_node_metrics_boxplot}{self.file_prefix}_legend_horizontal.png",
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()

        # Convert results to DataFrame
        df_results_pvalue = pd.DataFrame(results_pvalue)
        self.ch.save_dataframe(df_results_pvalue, f"{self.dm.path_node_metrics_boxplot}{self.file_prefix}_brunner_munzel_results.csv")
        self.lm.printl(f"{file_name}. plot_boxplot_metrics_gained_lost_nodes completed.")


    def compute_coordination_by_label(self, mid_th=0.5, metric='harmonicMean'):
        self.lm.printl(f"{file_name}. compute_coordination_by_label start.")
        flux_df = self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv",
            dtype=dtype)
        coord_df = self.ch.read_dataframe(
            self.dm.path_validation + f"{self.file_prefix}_coordination_communities.csv",
            dtype=dtype)

        coord_df['layer'].replace({'glouvain': 'multimodal', 'ginfomap': 'multimodal'}, inplace=True) # Only one of the two replacements is done, depending on the algorithm used
        coord_df.rename(columns={'community': 'com_layer'}, inplace=True)
        merged_df = pd.merge(flux_df, coord_df, on=['layer', 'com_layer'], how='left')
        merged_df = merged_df.rename(columns={'size': 'size_layer', 'avg_weight': 'avg_weight_layer', 
                                            'std_weight': 'std_weight_layer', 'median_weight': 'median_weight_layer',
                                            'mad_weight': 'mad_weight_layer'})

        coord_df.rename(columns={'com_layer': 'com_generic', 'layer': 'generic'}, inplace=True)
        merged2_df = pd.merge(merged_df, coord_df, on=['generic', 'com_generic'], how='left')
        merged2_df = merged2_df.rename(columns={'size': 'size_generic', 'avg_weight': 'avg_weight_generic', 
                                            'std_weight': 'std_weight_generic', 'median_weight': 'median_weight_generic',
                                            'mad_weight': 'mad_weight_generic'})
        coordination_label_df = self.__compute_weight_stats_summary(merged2_df)

        self.ch.save_dataframe(coordination_label_df, f"{self.dm.path_validation}{self.file_prefix}_coordination_label.csv")
        self.lm.printl(f"{file_name}. compute_coordination_by_label completed.")
    

    def combine_validation_communities(self, cda):
        self.lm.printl(f"{file_name}. combine_validation_communities start.")
        if cda.get_algorithm_name() in one_layer_algorithm:
            df_list = []
            if self.community_size_th is None:
                th_str = ""
            else:
                th_str = f"th_size_{str(self.community_size_th)}_"
            for type_ca, dict_path in self.dm.dict_path_ca.items():
                # from here I don't have access to the community discovery results of the single layer algorithms. I have
                # access only to the single layer co-actions, until the communities directory
                df = self.ch.read_dataframe(f"{dict_path["path_filter_community"]}{repr(cda)}{os.sep}analysis{os.sep}{type_ca}_validation_communities.csv", dtype=dtype)
                df['layer'] = type_ca
                
                df_list.append(df)
            # Concatenate all data into a single DataFrame
            combined_df = pd.concat(df_list, ignore_index=True)
        else:
            if cda.get_algorithm_name() in flatten_algorithm:
                combined_df = self.ch.read_dataframe(f"{self.dm.path_community}{repr(cda)}{os.sep}analysis{os.sep}{cda.get_algorithm_name()}_validation_communities.csv", dtype=dtype)
                combined_df['layer'] = cda.get_algorithm_name()
            else:
                combined_df = self.ch.read_dataframe(f"{self.dm.path_community}{repr(cda)}{os.sep}analysis{os.sep}{cda.get_algorithm_name()}_group_isControl_validation_communities.csv", dtype=dtype)
                combined_df['layer'] = cda.get_algorithm_name()
                
        self.ch.update_dataframe(combined_df, f"{self.dm.path_validation}{self.file_prefix}_validation_communities.csv", dtype=dtype)
        self.lm.printl(f"{file_name}. combine_validation_communities completed.")


    def compute_validation_by_label(self, mid_th=0.5, metric='harmonicMean'):
        self.lm.printl(f"{file_name}. compute_validation_by_label start.")
        flux_df = self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv",
            dtype=dtype)
        validation_df = self.ch.read_dataframe(
            self.dm.path_validation + f"{self.file_prefix}_validation_communities.csv",
            dtype=dtype)

        validation_df = self.__add_coordination_label(validation_df)
        filter_df = validation_df[['group', 'layer', 'labelCoordination']].copy()

        filter_df['layer'].replace({'glouvain': 'multimodal','ginfomap': 'multimodal'}, inplace=True)
        filter_df.rename(columns={'group': 'com_layer'}, inplace=True)
        merged_df = pd.merge(flux_df, filter_df, on=['layer', 'com_layer'], how='left')
        merged_df = merged_df.rename(columns={'labelCoordination': 'labelCoordination_layer'})

        filter_df.rename(columns={'com_layer': 'com_generic', 'layer': 'generic'}, inplace=True)
        merged2_df = pd.merge(merged_df, filter_df, on=['generic', 'com_generic'], how='left')
        merged2_df = merged2_df.rename(columns={'labelCoordination': 'labelCoordination_generic'})

        result_df = self.__validation_by_label(merged2_df)

        self.ch.save_dataframe(result_df, f"{self.dm.path_validation}{self.file_prefix}_validation_label.csv")
        self.lm.printl(f"{file_name}. compute_validation_by_label completed.")

    def plot_validation_multimodal(self, cda):
        self.lm.printl(f"{file_name}. plot_validation_multimodal start.")
        validation_label_df = self.ch.read_dataframe(f"{self.dm.path_validation}{self.file_prefix}_validation_label.csv", dtype=dtype)

        aggr_generic = validation_label_df.groupby(['generic', 'label'])[['notCoordinated', 'coordinated']].sum().reset_index()
        aggr_generic['percCoord'] = aggr_generic['coordinated']/(aggr_generic['coordinated']+aggr_generic['notCoordinated'])
        # Interest only in multimodal and flat_weighted_sum_louvain (infomap)
        mul_df = aggr_generic[(aggr_generic['generic']=='multimodal') | (aggr_generic['generic']==cda.get_algorithm_name())].copy()
        # Apply pretty names
        mul_df['generic'] = mul_df['generic'].replace(multimodal_print)

        value2plot_dict = {
            'notCoordinated': 'not-coordinated count', 
            'coordinated': 'coordinated count', 
            'percCoord': 'percentage coordinated'
        }
        for value2plot, y_label in value2plot_dict.items():
            
            # Desired label order
            label_order = ["lost", "common", "gained"]

            df_plot = mul_df.copy()
            df_plot["label"] = pd.Categorical(df_plot["label"], categories=label_order, ordered=True)

            plt.figure(figsize=(8, 6))

            ax = sns.barplot(
                data=df_plot,
                x="generic",
                y=value2plot,
                hue="label",
                palette=palette,
                order=sorted(df_plot["generic"].unique(), reverse=True),  # Reverse the order of the generic labels
                legend=False,
                width=0.9
            )

            # ----------------------
            # ★ ADD BAR ANNOTATIONS
            # ----------------------
            for container in ax.containers:
                if self.file_prefix == 'louvain_resolution_1':
                    if value2plot == 'percCoord':
                        th_plot_y = 0.04
                        labels = [round(v, 2) if v > th_plot_y else '' for v in container.datavalues]  # Only label values >= th_plot_y
                    else:
                        th_plot_y = 4
                        labels = [int(v) if v > th_plot_y else '' for v in container.datavalues]  # Only label values >= th_plot_y
                elif self.file_prefix == 'infomap':
                    if value2plot == 'percCoord':
                        th_plot_y = 0.05
                        labels = [round(v, 2) if v > th_plot_y else '' for v in container.datavalues]  # Only label values >= th_plot_y
                    else:
                        th_plot_y = 0
                        labels = [int(v) if v > th_plot_y else '' for v in container.datavalues]  # Only label values >= th_plot_y
        
                ax.bar_label(
                    container,
                    labels = labels,
                    padding=3,
                    fontsize=16,
                    label_type = 'center'
                )

            # Formatting
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
            plt.xlabel("", fontsize=16)
            plt.ylabel("", fontsize=16)

            plt.tight_layout()
            plt.savefig(f"{self.dm.path_validation}{self.file_prefix}_multimodal_{cda.get_algorithm_name()}_validation_{value2plot}.png",
                        dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.show()

        self.lm.printl(f"{file_name}. plot_validation_multimodal completed.")

    def plot_coordination_by_label(self, cda, metric='harmonicMean', mid_th=0.5, metric_to_plot = 'avg_weight'):
        self.lm.printl(f"{file_name}. plot_coordination_by_label start.")
        flux_df = self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv",
            dtype=dtype)
        coord_df = self.ch.read_dataframe(
            self.dm.path_validation + f"{self.file_prefix}_coordination_communities.csv",
            dtype=dtype)

        coord_df['layer'].replace({'glouvain': 'multimodal', 'ginfomap': 'multimodal'}, inplace=True) # Only one of the two replacements is done, depending on the algorithm used

        coord_df.rename(columns={'community': 'com_layer'}, inplace=True)
        merged_df = pd.merge(flux_df, coord_df, on=['layer', 'com_layer'], how='left')
        merged_df = merged_df.rename(columns={'size': 'size_layer', 'avg_weight': 'avg_weight_layer', 
                                            'std_weight': 'std_weight_layer', 'median_weight': 'median_weight_layer',
                                            'mad_weight': 'mad_weight_layer'})

        coord_df.rename(columns={'com_layer': 'com_generic', 'layer': 'generic'}, inplace=True)
        merged2_df = pd.merge(merged_df, coord_df, on=['generic', 'com_generic'], how='left')
        merged2_df = merged2_df.rename(columns={'size': 'size_generic', 'avg_weight': 'avg_weight_generic', 
                                            'std_weight': 'std_weight_generic', 'median_weight': 'median_weight_generic',
                                            'mad_weight': 'mad_weight_generic'})


        mul_df = merged2_df[(merged2_df['generic']=='multimodal') | (merged2_df['generic']==cda.get_algorithm_name())].copy()
        # Apply pretty names
        # mul_df['generic'] = mul_df['generic'].replace(multimodal_print)
        label_order = ["lost", "common", "gained"]
        # Darker version for mean lines + text
        dark_palette = {lbl: self.__darken(col, factor=0.7) for lbl, col in palette.items()}

        # Build plotting dataframe
        rows = []

        for (generic, label), group in mul_df.groupby(['generic', 'label']):
            if label == 'common':
                values = list(group[metric_to_plot + '_layer']) + list(group[metric_to_plot + '_generic'])
            elif label == 'gained':
                values = list(group[metric_to_plot + '_generic'])
            elif label == 'lost':
                values = list(group[metric_to_plot + '_layer'])
            values.sort()
            for v in values:
                rows.append({'generic': generic, 'label': label, 'value': v})

        plot_df = pd.DataFrame(rows)
        plot_df['label'] = pd.Categorical(plot_df['label'], categories=label_order, ordered=True)

        # -------------------------
        # Plot: one figure per generic
        # -------------------------
        for generic, gdf in plot_df.groupby("generic"):
            plt.figure(figsize=(12, 5), dpi=300)

            xpos = []
            colors = []
            values = []
            tick_labels = []
            counter = 0

            # Compute block centers for xticks
            block_centers = {}
            
            for lbl in label_order:
                vals = gdf[gdf['label'] == lbl]['value'].values

                # Save start index (before adding bars)
                start = counter

                for v in vals:
                    xpos.append(counter)
                    colors.append(palette[lbl])
                    values.append(v)
                    tick_labels.append(lbl)
                    counter += 1

                # Save block center
                if len(vals) > 0:
                    block_centers[lbl] = np.mean(range(start, start + len(vals)))
                else:
                    block_centers[lbl] = counter

                counter += 1  # block spacing

            # ---- Plot bars ----
            plt.bar(x=xpos, height=values, color=colors, width=0.8, zorder=2)

            # -----------------------
            # Add mean horizontal lines per label (ABOVE EVERYTHING)
            # -----------------------
            for lbl in label_order:
                vals = gdf[gdf['label'] == lbl]['value'].values
                if len(vals) == 0:
                    continue

                mean_val = np.mean(vals)

                # Full-width horizontal line
                plt.axhline(
                    y=mean_val,
                    color=dark_palette[lbl],
                    linestyle='--',
                    linewidth=1,
                    zorder=999  # <<< ensures it is on top
                )
                
                shift = self.__get_shift_coordination_mean(generic, lbl)
                center_lbl_shifted = block_centers[lbl] + shift
        #         print(block_centers[lbl], center_lbl_shifted)
                # Add centered annotation
                plt.text(
                    center_lbl_shifted,       # x center of the block
                    mean_val,                 # y at mean
                    f"{mean_val:.2f}",
                    ha='center', va='bottom',
                    fontsize=20,
                    color=dark_palette[lbl],
        #             fontweight='bold',
                    zorder=1000
                )

            # -----------------------
            # Formatting
            # -----------------------
            plt.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgray', zorder=0)

            plt.xticks(
                ticks=[block_centers[lbl] for lbl in label_order],
                labels=label_order,
                fontsize=16
            )
            
            plt.xlabel("", fontsize=16)
            plt.ylabel("", fontsize=16)

            plt.tight_layout()
            plt.savefig(f"{self.dm.path_validation}{self.file_prefix}_{generic}_{metric_to_plot}_coordination.png",
                        dpi=dpi, bbox_inches='tight', pad_inches=0)
            self.lm.printl(f"{file_name}. {self.dm.path_validation}{self.file_prefix}_{generic}_{metric_to_plot}_coordination.png saved.")
            plt.show()
       
        self.lm.printl(f"{file_name}. plot_coordination_by_label completed.")