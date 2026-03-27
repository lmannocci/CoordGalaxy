from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.ConversionManager.ConversionManager import *
from utils.Checkpoint.Checkpoint import *
from FilterGraphManager.FilterModels.BackboneNetwork.BackboneNetwork import *
from MergeNetworkManager import MergeNetworkManager

import os
import pandas as pd

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class FilterGraphManager:
    def __init__(self, dataset_name, user_fraction, type_filter, tw, ca, filter_instance):
        """
            FilterGraphManager constructor.
            :param ch: [Checkpoint] Checkpoint instance to save object.
            :param type_time_window: [str] The type of time window can be: ATW (Adjacent Time Window), OTW (Overlapping Time Window),
            ANY (no time window. The ATW exploits only tw_str, since tw_slide_interval_str is equal to tw_str.
            :param tw_str: [str] Length of the window, e.g., 1d, 1h, 30s.
            :param tw_slide_interval_str: [str] Size of the slide of the window. How much the window scrolls each time.
            :param threshold: [double] Threshold for removing edges with weight less than threshold.
        """
        self.lm = LogManager('main')
        self.ch = Checkpoint()

        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter

        self.tw = tw
        self.ca = ca

        self.dm = DirectoryManager(file_name, dataset_name, results=results, user_fraction=self.user_fraction,
                                   type_filter=self.type_filter, tw=self.tw, ca=self.ca, filter_instance=filter_instance)
        self.filter_instance = self.dm.get_filter()
        self.cm = ConversionManager()

        self.mm = MergeNetworkManager(self.dm, dataset_name, user_fraction, type_filter, tw, ca)

    def __filter_edges_threshold(self, tuple_element):
        """
            Remove edges with weight, nAction, twCount less than threshold, according to tuple_element parameter.
        """
        threshold = self.filter_instance.get_threshold()
        # if this is the first filter, I read from the original edge_list directory, otherwise from the previous filter edge list directory
        if self.filter_instance.get_previous_filter() is None:
            path_read = self.dm.path_edge_list
        else:
            path_read = self.dm.path_previous_filter_edge_list
        self.lm.printl(f"{file_name}. filter_edges_threshold start, threshold: {threshold}.")
        edge_list_files = [pos_csv for pos_csv in os.listdir(path_read) if pos_csv.endswith('.p')]
        for elf in edge_list_files:
            edge_list = self.ch.load_object(path_read + elf)
            count_edge = 0
            # edge: tuple(userId1, userId2, weight)

            # Solution1
            # filtered_edge_list = [(user1, user2, weight, nAction, twCount) for user1, user2, weight, nAction, twCount in edge_list if weight >= threshold]
            filtered_edge_list = [edge for edge in edge_list if edge[tuple_index[tuple_element]] >= threshold]

            # Solution2
            # filtered_edge_list = []
            # for edge in edge_list:
            #     # save edges with weight greater than threshold
            #     if edge[tuple_index['weight']] >= self.threshold:
            #         filtered_edge_list.append(edge)
            #         count_edge += 1

            # Solution 3 NOT WORK
            # edge_list_array = np.array(edge_list)
            # weights = edge_list_array[:, 2]
            # filtered_array = edge_list_array[weights >= threshold]
            # filtered_edge_list = filtered_array.tolist()

            self.lm.printl(f"{file_name}. filter_edges_threshold {elf}, Number of edges: {str(len(edge_list))}, Filtered number of edges: {str(len(filtered_edge_list))}.")
            # save the filtered edge list with same name of the original file, but in a different folder
            self.ch.save_object(filtered_edge_list, self.dm.path_filter_edge_list + elf)

            # self.cm.from_edge_list_to_df(filtered_edge_list, self.dm.path_filter_edge_list_df + elf.split('.')[0] + '.csv')

        self.lm.printl(f"{file_name}. filter_edges_threshold completed.")

    # def __filter_action(self, threshold):
    #     self.lm.printl(f"{file_name}. __filter_action start.")
    #     edge_list_files = [pos_csv for pos_csv in os.listdir(self.dm.path_edge_list_temporal) if pos_csv.endswith('.p')]
        # info_edge_list_files = [pos_csv for pos_csv in os.listdir(self.dm.path_overlapping_info_edge_list_temporal) if
        #                         pos_csv.endswith('.csv')]
        # sort the file in alphabetical order, which corresponds to the chronological order
        # edge_list_files = sorted(edge_list_files)
        # info_edge_list_files = sorted(info_edge_list_files)

        # if len(edge_list_files) == 0:
        #     m = f"No edge_list files to be merged."
        #     self.lm.printl(m)
        #     raise Exception(m)
        #
        # if len(info_edge_list_files) == 0:
        #     m = f"No info_edge_list files to be merged."
        #     self.lm.printl(m)
        #     raise Exception(m)

        # f1 = edge_list_files[0]
        # f2 = edge_list_files[-1]
        # # I take the files of the first and last time window. the name of each file is in format {start_date}_{end_date}.p
        # # For the first time window, I take the first element, that is the start_date and for the last time window, I take the end_date.
        # start_date = f1.split('_')[0]
        # end_date = f2.split('_')[1]
        # filename = start_date + '_' + end_date

        # for elf, ielf in zip(edge_list_files, info_edge_list_files):
        #     edge_list = self.ch.load_object(self.dm.path_edge_list_temporal + elf)
        #     info_edge_list = self.ch.read_dataframe(self.dm.path_overlapping_info_edge_list_temporal + ielf,
        #                                             dtype=dtype)
        #     edge_list_df = pd.DataFrame(edge_list, columns=['userId1', 'userId2', 'weight'])
        #
        #     g_info = info_edge_list.groupby(["userId1", "userId2"]).size().reset_index().rename(columns={0: 'count'})
        #     g_info_filter = g_info[g_info['count'] > threshold]
        #
        #     final_df = edge_list_df.merge(g_info_filter, on=['userId1', 'userId2'], how='inner')
        #     # drop count column, leaving only userId1, userId2, weight
        #     final_df = final_df.drop(columns=['count'])
        #
        #     final_edge_list = self.cm.from_dataframe_to_edge_list(final_df)
        #     self.ch.save_object(final_edge_list, self.dm.path_filter_edge_list_temporal + elf)
        #
        #     final_info_df = info_edge_list.merge(g_info_filter, on=['userId1', 'userId2'], how='inner')
        #     final_info_df.to_csv(self.dm.path_filter_info_edge_list_temporal + ielf)
        # self.lm.printl(f"{file_name}. __filter_action completed.")

    # def __merge_action(self):
    #     self.lm.printl(f"{file_name}. __merge_action start.")
    #     edge_list_files = [pos_csv for pos_csv in os.listdir(self.dm.path_filter_edge_list_temporal) if
    #                        pos_csv.endswith('.p')]
    #     # sort the file in alphabetical order, which corresponds to the chronological order
    #     edge_list_files = sorted(edge_list_files)
    #
    #     if len(edge_list_files) == 0:
    #         m = f"No edge_list files to be merged."
    #         self.lm.printl(m)
    #         raise Exception(m)
    #
    #     f1 = edge_list_files[0]
    #     f2 = edge_list_files[-1]
    #     # I take the files of the first and last time window. the name of each file is in format {start_date}_{end_date}.p
    #     # For the first time window, I take the first element, that is the start_date and for the last time window, I take the end_date.
    #     start_date = f1.split('_')[0]
    #     end_date = f2.split('_')[1]
    #     filename = start_date + '_' + end_date
    #     combined_df = pd.DataFrame()
    #     for elf in edge_list_files:
    #         edge_list = self.ch.load_object(self.dm.path_filter_edge_list_temporal + elf)
    #         temp_df = pd.DataFrame(edge_list, columns=['userId1', 'userId2', 'weight'])
    #         # df_list.append(temp_df)
    #         group_df = temp_df.groupby(['userId1', 'userId2'], as_index=False)['weight'].agg(['sum', 'count'])
    #         combined_df = pd.concat([combined_df, group_df], ignore_index=True)
    #
    #         combined_df = combined_df.groupby(['userId1', 'userId2'], as_index=False).agg(
    #             {'sum': 'sum', 'count': 'sum'})
    #
    #     del group_df
    #     del edge_list
    #     combined_df['average'] = combined_df['sum'] / combined_df['count']
    #     # for each edge in the edge_list, format (userid1, userId2, weight)
    #     self.lm.printl(f"{file_name}. All files have been read and processed.")
    #     if self.tw.get_type_merge() == "sum":
    #         combined_df = combined_df.drop(columns=['count', 'average'])
    #     elif self.tw.get_type_merge() == "average":
    #         combined_df = combined_df.drop(columns=['count', 'sum'])
    #     self.lm.printl(f"{file_name}. columns dropped.")
    #
    #     # the dataframe is too big, so I can't transform it in an edge list and then save it
    #     # I first save it as dataframe, then I read it in chunk to free memory
    #     # combined_df.to_csv(f"{self.dm.path_filter_edge_list_df}temporary_df.csv", index=False)
    #     merged_edge_list = self.cm.from_dataframe_to_edge_list(combined_df)
    #     self.lm.printl(f"{file_name}. merge_edge_list has been created.")
    #
    #     self.ch.save_object(merged_edge_list, f"{self.dm.path_filter_edge_list}{filename}")
    #
    #     self.lm.printl(
    #         f"{file_name}. __merge_action completed merging edge lists for window {start_date}_{end_date}. Number of edges: {str(len(merged_edge_list))}.")
    #


    def __filter_backbone(self):
        """
        This method filters the edges according to the method of Backbone extraction. It constructs a G_alpha graph,
        where alpha is assigned to each edge. Edges with alpha less than threshold are maintained,
        otherwise edges with alpha greater than threshold are removed.
        """
        self.lm.printl(f"{file_name}. __filter_backbone start.")
        if self.filter_instance.get_previous_filter() is None:
            path_read = self.dm.path_edge_list
        else:
            path_read = self.dm.path_previous_filter_edge_list
        threshold = self.filter_instance.get_threshold()
        bn = BackboneNetwork()

        edge_list_files = [pos_csv for pos_csv in os.listdir(path_read) if pos_csv.endswith('.p')]
        for elf in edge_list_files:
            edge_list = self.ch.load_object(path_read + elf)
            # edge: tuple(userId1, userId2, weight)
            G = nx.Graph()
            G.add_weighted_edges_from(edge_list, weight='weight')
            G_alpha = bn.disparity_filter(G, weight='weight')
            # I save the G_alpha in processed directory of the filter, which can be useful in the future, if I don't want
            # to compute again the disparity filter
            self.ch.save_object(G_alpha, self.dm.path_filter_processed + elf)
            # Solution 3
            # Create a list of edges to keep and remove those above the threshold in one iteration
            filtered_edge_list = []
            for u, v, data in G_alpha.edges(data=True):
                # in this case threshold is a sort of p-value, so we maintain only the edges with weight_alpha less than
                # threshold, which is usually low, such as 0.01 o 0.05
                if data['alpha'] < threshold:
                    filtered_edge_list.append((u, v, data['alpha']))
                else:
                    G.remove_edge(u, v)
            # save the filtered edge list with same name of the original file, but in a different folder
            self.ch.save_object(filtered_edge_list, self.dm.path_filter_edge_list + elf)

            self.lm.printl(f"{file_name}. __filter_backbone {elf}, Number of edges: {str(len(edge_list))}.")
        self.lm.printl(f"{file_name}. __filter_backbone completed.")

    def __filter_node_topEdge(self):
        """
        Filter the edge list acoording the method node_topEdge. It sorts the edges by edge weight, from the highest to
        the lowest. Given a maximum number of nodes, it continues adding edges, until the number of nodes is lower
        or equal to the maximum number of nodes.
        """
        threshold = self.filter_instance.get_threshold()
        # if this is the first filter, I read from the original edge_list directory, otherwise from the previous filter edge list directory
        if self.filter_instance.get_previous_filter() == None:
            path_read = self.dm.path_edge_list
        else:
            path_read = self.dm.path_previous_filter_edge_list
        self.lm.printl(f"{file_name}. __filter_node_topEdge start, maximum number of nodes: {threshold}.")
        edge_list_files = [pos_csv for pos_csv in os.listdir(path_read) if pos_csv.endswith('.p')]
        for elf in edge_list_files:
            filtered_edge_list = []
            set_nodes = set()
            edge_list = self.ch.load_object(path_read + elf)
            # edge: tuple(userId1, userId2, weight)

            # Sorting the list by the third value (float) in descending order
            sorted_edge_list = sorted(edge_list, key=lambda e: e[tuple_index[2]], reverse=True)

            for user1, user2, weight in sorted_edge_list:
                cond1 = (user1 not in set_nodes and user2 not in set_nodes and len(set_nodes) <= threshold - 2)  # there must be space for both users in the set
                cond2 = (((user1 not in set_nodes and user2 in set_nodes) or (user1 in set_nodes and user2 not in set_nodes)) and len(set_nodes) <= threshold - 1) # or space for one user
                cond3 = (user1 in set_nodes and user2 in set_nodes)  # or both are already in the set, so I can add the edge, because the number of nodes does not increase
                if cond1 or cond2 or cond3:
                    set_nodes.add(user1)
                    set_nodes.add(user2)
                    filtered_edge_list.append((user1, user2, weight))

            self.lm.printl(f"{file_name}. __filter_node_topEdge {elf}, Number of edges: {str(len(edge_list))}, Filtered number of edges: {str(len(filtered_edge_list))}.")
            # save the filtered edge list with same name of the original file, but in a different folder
            self.ch.save_object(filtered_edge_list, self.dm.path_filter_edge_list + elf)

            # convert the edge lists to dataframe
            # dict_edge_list = {"userId1": [], "userId2": [], "weight": []}
            # for e in filtered_edge_list:
            #     dict_edge_list["userId1"].append(e[0])
            #     dict_edge_list["userId2"].append(e[1])
            #     dict_edge_list["weight"].append(e[2])
            # df = pd.DataFrame(dict_edge_list)
            # Convert to DataFrame with specified column names
            max_index = len(edge_list[0])
            columns = list(tuple_index.keys())[0:max_index]
            df = pd.DataFrame(edge_list, columns=columns)
            self.ch.save_dataframe(df, self.dm.path_filter_edge_list_df + elf.split('.')[0] + '.csv')

        self.lm.printl(f"{file_name}. __filter_node_topEdge completed.")

    # def filter_edges_topK(self, G, K, filename):
    #     def compare_edge(e):
    #         return e[2]['weight']
    #
    #     edges = list(G.edges(data=True))
    #     if K > len(edges):
    #         raise Exception("Error - filter_graph_topK: K must be less or equal to the number of edges.")
    #
    #     sorted_edge = sorted(edges, key=compare_edge, reverse=True)
    #     filtered_edge_list = sorted_edge[0:K]
    #
    #     self.ch.save_object(filtered_edge_list, f"{filename}.p")
    #     self.ch.save_edge_list_csv(filtered_edge_list, f"{filename}.csv")
    #     self.lm.printl("FilterGraphManager: filter_graph_threshold completed.")
    #     return filtered_edge_list

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    def filter_graph(self):
        type_filter = self.filter_instance.get_type_filter()
        self.lm.printl(f"{file_name}. filter_graph {type_filter} start.")

        if type_filter in ["low_std", "mean", 'median', "high_std", 'th']:
            self.__filter_edges_threshold("w_")
        elif type_filter == "backbone":
            self.__filter_backbone()
        elif type_filter == "filter_merge_action":
            self.__filter_edges_threshold("nAction")
            self.mm.merge_edge_list(self.dm.path_filter_edge_list_temporal, self.dm.path_filter_edge_list)
        elif type_filter == "merge_filter_action":
            self.__filter_edges_threshold("nAction")
        elif type_filter == "node_topEdge":
            self.__filter_node_topEdge()

        self.lm.printl(f"{file_name}. filter_graph {type_filter} completed.")
