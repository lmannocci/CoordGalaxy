import pandas as pd

from SimilarityFunctionManager.methods.similarityFunction import *
from utils.PlotManager.PlotManager import PlotManager
from utils.common_variables import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *
from DirectoryManager import DirectoryManager

import uunet.multinet as ml
import os
import matplotlib.pyplot as plt
import networkx as nx
import statistics
import numpy as np
from collections import defaultdict
import seaborn as sns
from itertools import combinations

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class NetworkMeasure:
    def __init__(self, list_ca, dict_ca_filter, icm, dm:DirectoryManager, type_algorithm):

        self.lm = LogManager('main')
        self.ch = Checkpoint()
        self.cm = ConversionManager()

        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter
        self.icm = icm
        self.dm = dm
        self.list_ca_str = '_'.join(list(self.dm.dict_path_ca.keys()))
        self.type_algorithm = type_algorithm

        self.pm = PlotManager()

    def __get_files_path_info(self, filter_instance, dict_path):
        if filter_instance is None:  # not filtered network must be read
            # read edge list
            path = dict_path["path_NF_edge_list"]
            path_analysis = dict_path["path_NF_analysis"]
            edge_list_files = [pos_csv for pos_csv in os.listdir(path) if pos_csv.endswith('.p')]
            edge_list_files = sorted(edge_list_files)
            is_graph = False
            return is_graph, path, path_analysis, edge_list_files
        else:  # filtered network
            path = dict_path["path_filter_graph"]
            path_analysis = dict_path["path_filter_analysis"]
            graph_files = [pos_csv for pos_csv in os.listdir(path) if pos_csv.endswith('.p')]
            graph_files = sorted(graph_files)
            is_graph = True
            if len(graph_files) == 0:
                path = dict_path["path_filter_edge_list"]
                edge_list_files = [pos_csv for pos_csv in os.listdir(path) if pos_csv.endswith('.p')]
                edge_list_files = sorted(edge_list_files)
                is_graph = False
                return is_graph, path, path_analysis, edge_list_files
            return is_graph, path, path_analysis, graph_files

    def __compute_network_measure(self, network, is_graph, type_ca, path_analysis, metrics_to_compute, metrics_dict):
        if 'weight_statistics' in metrics_to_compute:
            stats_dict = self.__compute_edge_weight_statistics(is_graph, network)
            metrics_dict.update(stats_dict)
            self.__plot_edge_weight_distribution(path_analysis, type_ca, is_graph, network)
            self.lm.printl(f"{file_name}. weight_statistics computed for co-action {type_ca}")

        if 'nNodes' in metrics_to_compute:
            metrics_dict['nNodes'] = self.__nNodes(is_graph, network)
            self.lm.printl(f"{file_name}. nNodes computed for co-action {type_ca}")

        if 'nEdges' in metrics_to_compute:
            metrics_dict['nEdges'] = self.__nEdges(is_graph, network)
            self.lm.printl(f"{file_name}. nEdges computed for co-action {type_ca}")

        if "node_topEdge_trend" in metrics_to_compute:
            self.__plot_node_topEdge_trend(path_analysis, type_ca, is_graph, network)
            self.lm.printl(f"{file_name}. node_topEdge_trend computed for co-action {type_ca}")

        if 'degree_distribution' in metrics_to_compute:
            self.__plot_degree_distribution(path_analysis, type_ca, is_graph, network)
            self.lm.printl(f"{file_name}. degree_distribution computed for co-action {type_ca}")

        if 'nAction_distribution' in metrics_to_compute:
            self.__show_nAction_distribution(path_analysis, type_ca, is_graph, network)

        require_construction = False
        for m in metrics_to_compute:
            if m in require_network_construction_metrics:
                require_construction = True
                break
        # For the following metrics, the construction of the graph is necessary
        if require_construction:
            if is_graph:
                G = network
            else:
                G = self.cm.from_edge_list_to_graph(network)

            # Assortativity
            if "assortativity" in metrics_to_compute:
                assortativity = nx.degree_assortativity_coefficient(G)
                metrics_dict['assortativity'] = assortativity
                self.lm.printl(f"{file_name}. assortativity computed for co-action {type_ca}")

                # Degree centrality
            if 'degree_centrality' in metrics_to_compute:
                degree_centrality = nx.degree_centrality(G)
                metrics_dict['degreeCentrality'] = degree_centrality
                self.lm.printl(f"{file_name}. degreeCentrality computed for co-action {type_ca}")

                # Betweeness centrality
            if 'betweenness_centrality' in metrics_to_compute:
                betweenness_centrality = nx.betweenness_centrality(G)
                metrics_dict['betweennessCentrality'] = betweenness_centrality
                self.lm.printl(f"{file_name}. betweennessCentrality computed for co-action {type_ca}")

                # Closeness centrality
            if 'closeness_centrality' in metrics_to_compute:
                closeness_centrality = nx.closeness_centrality(G)
                metrics_dict['closenessCentrality'] = closeness_centrality
                self.lm.printl(f"{file_name}. closenessCentrality computed for co-action {type_ca}")

                # Shortest path
            if 'shortest_path_lengths' in metrics_to_compute:
                shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
                metrics_dict['shortestPathLengths'] = shortest_path_lengths
                self.lm.printl(f"{file_name}. shortestPathLenghts computed for co-action {type_ca}")

                # Eccentricity
            if 'eccentricity' in metrics_to_compute:
                eccentricity = nx.eccentricity(G)
                metrics_dict['eccentricity'] = eccentricity

            if 'connected_components' in metrics_to_compute:
                connected_components = list(nx.connected_components(G))
                metrics_dict['nConnectedComponents'] = len(connected_components)
                self.__plot_component_sizes_distribution(connected_components, path_analysis, type_ca)
                self.lm.printl(f"{file_name}. nConnectedComponents computed for co-action {type_ca}")

        return metrics_dict

    def __show_nAction_distribution(self, path_analysis, type_ca, is_graph, network):
        if is_graph:
            nAction = [data[NA_VAR] for _, _, data in network.edges(data=True)]
        else:
            nAction = [e[tuple_index[NA_VAR]] for e in network]
        # Plotting the distribution
        self.pm.plot_histogram(path_analysis, type_ca, nAction,
                               f"Number of common actions per edge", 'Frequency',
                               f"Distribution of number of common actions involved to construct an edge, {type_ca}.",
                               "number_co_action_distribution.png")

    def __compute_edge_weight_statistics(self, is_graph, network):
        self.lm.printl(f"{file_name}. __compute_edge_weight_statistics start.")

        if is_graph:
            weights = [data[W_VAR] for _, _, data in network.edges(data=True)]
        else:
            weights = [e[tuple_index[W_VAR]] for e in network]

        # Compute statistics
        mean_weight = statistics.mean(weights)
        median_weight = statistics.median(weights)
        std_dev = statistics.stdev(weights)
        max_weight = max(weights)
        min_weight = min(weights)

        # Create DataFrame with statistics
        stats_dict = {
            'meanWeight': mean_weight,
            'medianWeight': median_weight,
            'stdDevWeight': std_dev,
            'maxWeight': max_weight,
            'minWeight': min_weight
        }
        self.lm.printl(f"{file_name}. __compute_edge_weight_statistics completed.")
        return stats_dict

    def __plot_component_sizes_distribution(self, connected_components, path_analysis, type_ca):
        self.lm.printl(f"{file_name}. __plot_component_sizes_distribution start.")
        component_sizes = [len(component) for component in connected_components]
        df = pd.DataFrame(component_sizes, columns=['componentSize'])
        df_sorted = df.sort_values(by='componentSize', ascending=False).reset_index(drop=True)
        self.ch.save_dataframe(df_sorted, path_analysis + "size_connected_components.csv")
        self.pm.plot_line(path_analysis, type_ca, df_sorted.index+1, df_sorted['componentSize'], 'Index',
                          'Size of connected components', f'Size of connected components {type_ca}',
                          'plot_size_connected_components.png')
        self.lm.printl(f"{file_name}. __plot_component_sizes_distribution completed.")

    def __plot_degree_distribution(self, path_analysis, type_ca, is_graph, network):
        self.lm.printl(f"{file_name}. show_degree_distribution start.")

        if is_graph:
            degrees = [network.degree(node) for node in network.nodes()]
        else:
            # Initialize a defaultdict to keep track of degrees
            degree_dict = defaultdict(int)

            # Iterate through the edge list
            for node1, node2, weight in network:
                degree_dict[node1] += 1
                degree_dict[node2] += 1
            # Extract the degree values
            degrees = list(degree_dict.values())

        self.pm.plot_histogram(path_analysis, type_ca, degrees, 'Degree', 'Frequency',
                               f'{type_ca} Degree Distribution', 'degree_distribution.png')
        self.lm.printl(f"{file_name}. show_degree_distribution finish.")

    def __plot_edge_weight_distribution(self, path_analysis, type_ca, is_graph, network):
        self.lm.printl(f"{file_name}. __show_edge_weight_distribution start.")

        if is_graph:
            weights = [data[W_VAR] for _, _, data in network.edges(data=True)]
        else:
            weights = [e[tuple_index[W_VAR]] for e in network]

        # Plotting the distribution
        self.pm.plot_histogram(path_analysis, type_ca, weights, 'Weights', 'Frequency',
                               f'{type_ca} Distribution of Weights', 'edge_weight_distribution.png')

        self.lm.printl(f"{file_name}. __show_edge_weight_distribution completed.")

    def __plot_node_topEdge_trend(self, path_analysis, type_ca, is_graph, network):
        self.lm.printl(f"{file_name}. __show_edge_weight_distribution start.")

        if is_graph:
            # Convert the graph to a list of tuples (node1, node2, weight)
            edge_list = [(u, v, d[W_VAR]) for u, v, d in network.edges(data=True)]
        else:
            edge_list = network

        filtered_edge_list = []
        set_nodes = set()
        # edge: tuple(userId1, userId2, weight)

        # Sorting the list by the third value (float) in descending order
        sorted_edge_list = sorted(edge_list, key=lambda x: x[2], reverse=True)
        nNodes_values = []
        nEdges_values = []
        for user1, user2, weight in sorted_edge_list:
            set_nodes.add(user1)
            set_nodes.add(user2)
            nNodes_values.append(len(set_nodes))
            filtered_edge_list.append((user1, user2, weight))
            nEdges_values.append(len(filtered_edge_list))

        self.pm.plot_line(path_analysis, type_ca, nEdges_values, nNodes_values, 'Number of edges', 'Number of nodes',
                         f'Filter node_topEdge {type_ca}', 'filter_node_topEdge.png')

        self.lm.printl(f"{file_name}. __show_edge_weight_distribution completed.")

    def __nNodes(self, is_graph, network):
        if is_graph:
            return network.number_of_nodes()
        else:
            # compute nNodes. edge list in format userId1, userId2, weight, nAction, twCount
            user_set1 = set([e[tuple_index[NODE1_VAR]] for e in network])
            user_set2 = set([e[tuple_index[NODE2_VAR]] for e in network])
            user_set = user_set1.union(user_set2)
            return len(user_set)

    def __nEdges(self, is_graph, network):
        if is_graph:
            return network.number_of_edges()
        else:
            return len(network)

    def __setUsers(self, is_graph, network):
        if is_graph:
            return set(network.nodes())
        else:
            user_set1 = set([e[tuple_index[NODE1_VAR]] for e in network])
            user_set2 = set([e[tuple_index[NODE1_VAR]] for e in network])
            user_set = user_set1.union(user_set2)
            return user_set

    def __filter_threshold(self, is_graph, network, threshold, filter_par_type):
        if is_graph:
            # Filter edges based on weight threshold
            filtered_edge_list = [(u, v, data) for u, v, data in network.edges(data=True) if data[filter_par_type] >= threshold]
        else:
            filtered_edge_list = [edge for edge in network if edge[tuple_index[filter_par_type]] >= threshold]
        is_graph_filtered = False
        # Create a new graph with filtered edges
        # filtered_graph = self.__create_weighted_graph(filtered_edge_list)

        return is_graph_filtered, filtered_edge_list

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    def compute_metrics_network(self, metrics_to_compute):
        self.lm.printl(f"{file_name}. compute_metrics_networks start.")

        self.icm.check_metrics_networks(metrics_to_compute)

        metrics_list = []
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            filter_ca = self.dict_ca_filter[type_ca]
            is_graph, path, path_analysis, list_files = self.__get_files_path_info(filter_ca, dict_path)
            for elf in list_files:
                metrics_dict = {}
                metrics_dict['layer'] = type_ca
                if filter_ca is not None:
                    metrics_dict['filter'] = filter_ca.filter_repr_abbr()
                else:
                    metrics_dict['filter'] = 'None'
                metrics_dict['tw'] = elf

                # this can be a NetworkX graph or an edge list, depending on the attribute is_graph.
                # Both are pickle file, so I can read them in the same mode
                network = self.ch.load_object(path + elf)

                metrics_dict = self.__compute_network_measure(network, is_graph, type_ca, path_analysis, metrics_to_compute, metrics_dict)


                # if any(metric in metrics_to_compute for metric in
                #        available_edge_list_network_metrics) and is_graph == False:
                #     self.__compute_edge_list_network_metrics(network, type_ca, path_analysis, metrics_to_compute,
                #                                              metrics_dict)
                # elif any(metric in metrics_to_compute for metric in
                #          available_graph_network_metrics) and is_graph == True:
                #     metrics_dict = self.__compute_graph_network_metrics(network, type_ca, path_analysis,
                #                                                         metrics_to_compute, metrics_dict)
                # else:
                #     # case: is_graph = False and within metrics_to_compute is requested "assortativity", for which graph is required
                #     m = f"{file_name}. Impossible computing requested metrics. NetworkX graph format is required to compute this type of measure."
                #     self.lm.printl(m)
                #     raise Exception(m)

                metrics_list.append(metrics_dict)

            # save in the directory analysis of the single co-action if it is selected only one co-action
            if self.type_algorithm == "one-layer":
                metrics_df = pd.DataFrame(metrics_list)
                self.ch.save_dataframe(metrics_df, path_analysis + "metrics.csv")

        metrics_df = pd.DataFrame(metrics_list)
        # metrics_df = metrics_df.set_index('layer')
        if self.type_algorithm == "multi-layer":
            # save in the directory analysis of multi-co-action
            self.ch.save_dataframe(metrics_df, self.dm.path_analysis + "metrics.csv")

        self.lm.printl(f"{file_name}. compute_metrics_networks completed.")

    #
    # def compute_nAction_statistics(self, min_th, max_th, step):
    #     """
    #     Computes statistics for multiple graphs incrementally.
    #     """
    #
    #     self.lm.printl(f"{file_name}. compute_nAction_statistics start.")
    #
    #     layer_data = []
    #     threshold_data = []
    #     overlapping_data = []
    #     for threshold in range(min_th, max_th + step, step):
    #         set_users_dict = {}
    #         node_sum = 0
    #         edge_sum = 0
    #         for type_ca, dict_path in self.dm.dict_path_ca.items():
    #             filter_ca = self.dict_ca_filter[type_ca]
    #             is_graph, path, path_analysis, list_files = self.__get_files_path_info(filter_ca, dict_path)
    #             elf = list_files[0]
    #             # this can be a NetworkX graph or an edge list, depending on the attribute is_graph.
    #             # Both are pickle file, so I can read them in the same mode
    #             network = self.ch.load_object(path + elf)
    #             nNodes = self.__nNodes(is_graph, network)
    #             nEdges = self.__nEdges(is_graph, network)
    #
    #             filter_G = self.__filter_threshold(is_graph, network, threshold)
    #
    #             # set_users_dict contains the set of users/nodes for each co-action/layer of the filtered graph
    #             set_users_dict[type_ca] = self.__setUsers(filter_G, True)
    #
    #             layer_data.append({
    #                 'layer': type_ca,
    #                 'threshold': threshold,
    #                 'nNodes': nNodes,
    #                 'nEdges': nEdges,
    #                 'nFilterNodes': self.__nNodes(True, filter_G),
    #                 'nFilterEdges': self.__nEdges(True, filter_G)
    #             })
    #
    #             # Total number of nodes and edges of all layer (same nodes in different layers count separate)
    #             node_sum = node_sum + nNodes
    #             edge_sum = edge_sum + nEdges
    #
    #         # Calculate overlap percentages of nodes/users for each couple of co-actions of the filtered graph
    #         for c1, c2 in combinations(self.dm.dict_path_ca.keys(), 2):
    #             user_set1 = set_users_dict[c1]
    #             user_set2 = set_users_dict[c2]
    #
    #             c1_name = co_action_map[c1]
    #             c2_name = co_action_map[c2]
    #
    #             _, absolute_o, o_coefficient = overlapping_coefficient(user_set1, user_set2)
    #             o_perc = round(o_coefficient * 100)
    #
    #             overlapping_data.append({
    #                 'layer1': c1_name,
    #                 'layer2': c2_name,
    #                 'threshold': threshold,
    #                 'overlapping': absolute_o,
    #                 'percOverlapping': o_perc
    #             })
    #
    #         # Union of all set of users/nodes of all layers/co-actions,
    #         # to compute the total number of nodes (unique)  of the filtered graph
    #         merge_user_set = set()
    #         for user_set in set_users_dict.values():
    #             merge_user_set = merge_user_set | user_set
    #         nUniqueNodes = len(merge_user_set)
    #
    #         threshold_data.append({
    #             'threshold': threshold,
    #             'nNodes': node_sum,
    #             'nEdges': edge_sum,
    #             'nUniqueNodes': nUniqueNodes,
    #         })
    #
    #     layer_df = pd.DataFrame(layer_data)
    #     threshold_df = pd.DataFrame(threshold_data)
    #     overlapping_df = pd.DataFrame(overlapping_data)
    #
    #     self.ch.save_dataframe(layer_df, self.dm.path_analysis + 'nAction_layer_df.csv')
    #     self.ch.save_dataframe(threshold_df, self.dm.path_analysis + 'nAction_threshold_df.csv')
    #     self.ch.save_dataframe(overlapping_df, self.dm.path_analysis + 'nAction_overlapping_df.csv')
    #
    #     # Plot for each couple of co-actions the threshold vs percOverlapping
    #     # (how much overlapping of nodes is there between each couple of co-actions)
    #     self.plot_nAction_threshold(overlapping_df)
    #
    #     self.lm.printl(f"{file_name}. compute_nAction_statistics completed.")

    def compute_threshold_statistics(self, min_th, max_th, step, filter_par_type):
        """
        Computes statistics for min_th - max_th threshold (on nAction or weight).
        """

        self.lm.printl(f"{file_name}. compute_threshold_statistics start.")

        layer_data = []
        set_users_dict = {}
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            self.lm.printl(f"{file_name}. {type_ca} start.")
            filter_ca = self.dict_ca_filter[type_ca]
            is_graph, path, path_analysis, list_files = self.__get_files_path_info(filter_ca, dict_path)
            elf = list_files[0]
            # this can be a NetworkX graph or an edge list, depending on the attribute is_graph.
            # Both are pickle file, so I can read them in the same mode
            network = self.ch.load_object(path + elf)

            set_users_dict[type_ca] = {}

            nNodes = self.__nNodes(is_graph, network)
            nEdges = self.__nEdges(is_graph, network)

            threshold = min_th
            while threshold <= max_th:
                self.lm.printl(f"{file_name}. {type_ca}-{str(threshold)} start.")
                # always returns a filtered edge list (is_graph_filtered=False)
                is_graph_filtered, filtered_edge_list = self.__filter_threshold(is_graph, network, threshold, filter_par_type)
                self.lm.printl(f"{file_name}. {type_ca}-{str(threshold)} filtered.")

                # set_users_dict contains the set of users/nodes for each co-action/layer, for each threshold
                # of the filtered graph
                set_users_dict[type_ca][threshold] = self.__setUsers(is_graph_filtered, filtered_edge_list)

                nFilterNodes = self.__nNodes(is_graph_filtered, filtered_edge_list)
                nFilterEdges = self.__nEdges(is_graph_filtered, filtered_edge_list)

                layer_data.append({
                    'layer': type_ca,
                    'threshold': threshold,
                    'nNodes': nNodes,
                    'nEdges': nEdges,
                    'nFilterNodes': nFilterNodes,
                    'nFilterEdges': nFilterEdges
                })
                self.lm.printl(f"{file_name}. {type_ca}-{str(threshold)} completed.")
                threshold += step
                threshold = round(threshold, 2)
            self.lm.printl(f"{file_name}. {type_ca} completed.")

        self.lm.printl(f"{file_name}. Overlapping computation start.")
        threshold_data = []
        overlapping_data = []
        threshold = min_th
        while threshold <= max_th:
            self.lm.printl(f"{file_name}. Overlapping computation {threshold}.")
            # Calculate overlap percentages of nodes/users for each couple of co-actions of the filtered graph
            for c1, c2 in combinations(self.dm.dict_path_ca.keys(), 2):

                user_set1 = set_users_dict[c1][threshold]
                user_set2 = set_users_dict[c2][threshold]

                c1_name = co_action_map[c1]
                c2_name = co_action_map[c2]

                _, absolute_o, o_coefficient = overlapping_coefficient(user_set1, user_set2)
                o_perc = round(o_coefficient * 100)

                overlapping_data.append({
                    'layer1': c1_name,
                    'layer2': c2_name,
                    'threshold': threshold,
                    'overlapping': absolute_o,
                    'percOverlapping': o_perc
                })

            # Union of all set of users/nodes of all layers/co-actions,
            # to compute the total number of nodes (unique)  of the filtered graph
            merge_user_set = set()
            total_layers_nodes = 0
            # Given a threshold, I do the union of all the co-actions user sets
            for user_set_ca_dict in set_users_dict.values():
                merge_user_set = merge_user_set | user_set_ca_dict[threshold]
                # Total number of nodes and edges of all layer (same nodes in different layers count separate)
                total_layers_nodes += len(user_set_ca_dict[threshold])

            nUniqueNodes = len(merge_user_set)
            threshold_data.append({
                'threshold': threshold,
                'nNodes': total_layers_nodes,
                'nUniqueNodes': nUniqueNodes,
            })

            threshold += step
            threshold = round(threshold, 2)
        self.lm.printl(f"{file_name}. Overlapping computation completed.")

        layer_df = pd.DataFrame(layer_data)
        threshold_df = pd.DataFrame(threshold_data)
        overlapping_df = pd.DataFrame(overlapping_data)

        self.ch.save_dataframe(layer_df, self.dm.path_analysis + f'{filter_par_type}_layer_df.csv')
        self.ch.save_dataframe(threshold_df, self.dm.path_analysis + f'{filter_par_type}_threshold_df.csv')
        self.ch.save_dataframe(overlapping_df, self.dm.path_analysis + f'{filter_par_type}_overlapping_df.csv')

        self.lm.printl(f"{file_name}. compute_threshold_statistics completed.")

    def plot_threshold_overlapping(self, filter_par_type, step):
        self.lm.printl(f"{file_name}. plot_threshold_overlapping started.")

        df = self.ch.read_dataframe(self.dm.path_analysis + f'{filter_par_type}_overlapping_df.csv', dtype=dtype)
        self.pm.plot_grid_combinations(df, self.dm.path_analysis, f"plot_{filter_par_type}_threshold_overlapping.png", "layer1",
                                       'layer2', 'threshold', 'percOverlapping',
                                       'threshold', 'percOverlapping', step)
        self.lm.printl(f"{file_name}. plot_threshold_overlapping completed.")

    def plot_nodes_edges_threshold(self, filter_par_type):
        self.lm.printl(f"{file_name}. plot_nodes_edges_threshold started.")

        df = self.ch.read_dataframe(self.dm.path_analysis + f'{filter_par_type}_layer_df.csv', dtype=dtype)

        # plot for each layer/co-action, the number of filtered nodes, for each threshold value
        self.pm.plot_grid_line(self.dm.path_analysis, f"plot_{filter_par_type}_threshold_nNodes.png",df,
                               'layer', 'threshold', 'nFilterNodes',
                               'threshold', 'nFilterNodes', 'nFilterNodes layer')

        # plot for each layer/co-action, the number of filtered edges, for each threshold value
        self.pm.plot_grid_line(self.dm.path_analysis, f"plot_{filter_par_type}_threshold_nEdges.png", df,
                               'layer', 'threshold', 'nFilterEdges',
                               'threshold', 'nFilterEdges', 'nFilterEdges layer')

        self.lm.printl(f"{file_name}. plot_nodes_edges_threshold completed.")

    def select_threshold_statistics(self, min_th, max_th, step, absolute_th_mode, filter_par_type, target_type):
        # filter_par_type nAction - weight
        # target - node/edge
        self.lm.printl(f"{file_name}. select_threshold_statistics started.")

        df = self.ch.read_dataframe(self.dm.path_analysis + f'{filter_par_type}_layer_df.csv', dtype=dtype)

        layer_list = df['layer'].unique()
        n_layers = len(layer_list)

        # for each abs/perc threshold, it includes the corresponding threshold for the target (node/edge)
        # in case of absolute mode, the absolute threshold th coincides with th_layer_target
        th_layer_target = {}
        # for each abs/perc threshold, it includes the corresponding threshold for the filter_par_type (nAction/weight)
        th_layer_parameter = {}
        result_dict = {}
        th = min_th
        while th <= max_th:
            th_layer_target[th] = {}
            th_layer_parameter[th] = {}
            result_dict[th] = {}
            for i, layer in enumerate(layer_list):
                if absolute_th_mode:
                    if target_type == "node":
                        s = "thNode"
                    elif target_type == "edge":
                        s = "thEdge"
                else:
                    if target_type == "node":
                        s = "percNode"
                    elif target_type == "edge":
                        s = "percEdge"

                self.lm.printl(f"{file_name}. {s}: {str(th)}, layer: {layer} computing thresholds.")
                subset = df[df['layer'] == layer]
                # get original not filtered number of node and edges for the current layer
                n_nodes = subset['nNodes'].values[0]
                n_edges = subset['nEdges'].values[0]

                # absolute_th_mode==False if I want to fix the percentage of node in each layer,
                # so I have to compute what is, e.g., the 30% of nodes in a layer.
                # absolute_number_node==True if I want to fix a number of node in each layer, e.g., 10000 nodes.
                # so i have already the th_layer_target.
                if not absolute_th_mode:
                    if target_type == "node":
                        # nFilteredNodes
                        th_layer_target[th][layer] = int(n_nodes * th)
                        filter_col = "nFilterNodes"
                    elif target_type == "edge":
                        # nFilteredEdges
                        th_layer_target[th][layer] = int(n_edges * th)
                        filter_col = "nFilterEdges"
                else:
                    if target_type == "node":
                        th_layer_target[th][layer] = th
                        filter_col = "nFilterNodes"
                    elif target_type == "edge":
                        th_layer_target[th][layer] = th
                        filter_col = "nFilterEdges"

                filter_df = subset[subset[filter_col] <= th_layer_target[th][layer]]

                # this can happen, if the required number of filtered nodes correspond to a threshold greater than 149,
                # which is the maximum threshold tried and computed in compute_nAction_statistics
                # in this case I have to manage the missing case (i do not have the threshold, so i do not compute
                # anything for this layer-th

                if filter_df.shape[0] == 0:
                    th_layer_target[th][layer] = None
                    th_layer_parameter[th][layer] = None
                    final_nNodes = None
                    final_nEdges = None
                else:
                    th_layer_parameter[th][layer] = filter_df.sort_values(by='threshold', ascending=True)['threshold'].values[0]
                    final_nNodes = subset[subset['threshold'] == th_layer_parameter[th][layer]]['nFilterNodes'].values[0]
                    final_nEdges = subset[subset['threshold'] == th_layer_parameter[th][layer]]['nFilterEdges'].values[0]

                if not absolute_th_mode:
                    row = {
                        'layer': layer,
                        f'perc_{target_type}': th,
                        f'th_{target_type}': th_layer_target[th][layer],
                        f'th_{filter_par_type}': th_layer_parameter[th][layer],
                        'nNodes': final_nNodes,
                        'nEdges': final_nEdges
                    }
                else:
                    # the percentage is not present, you directly have the number of nodes
                    row = {
                        'layer': layer,
                        f'th_{target_type}': th_layer_target[th][layer],
                        f'th_{filter_par_type}': th_layer_parameter[th][layer],
                        'nNodes': final_nNodes,
                        'nEdges': final_nEdges
                    }
                result_dict[th][layer] = row
            th += step


        set_users_dict = {}
        weight_stat_dict = {}

        for type_ca, dict_path in self.dm.dict_path_ca.items():
            set_users_dict[type_ca] = {}
            weight_stat_dict[type_ca] = {}

            # read network
            filter_ca = self.dict_ca_filter[type_ca]
            is_graph, path, path_analysis, list_files = self.__get_files_path_info(filter_ca, dict_path)
            elf = list_files[0]
            # this can be a NetworkX graph or an edge list, depending on the attribute is_graph.
            # Both are pickle file, so I can read them in the same mode
            network = self.ch.load_object(path + elf)

            th = min_th
            while th <= max_th:
                if th_layer_parameter[th][type_ca] is None:
                    weight_stat_dict[type_ca][th] = {
                        'meanWeight': None,
                        'medianWeight': None,
                        'stdDevWeight': None,
                        'maxWeight': None,
                        'minWeight': None
                    }
                    result_dict[th][type_ca].update(weight_stat_dict[type_ca][th])

                    set_users_dict[type_ca][th] = None
                else:
                    self.lm.printl(f"{file_name}. {s}: {str(th)}, layer: {type_ca} computing user sets for different node threshold.")

                    # always returns a filtered edge list (is_graph_filtered=False)
                    is_graph_filtered, filtered_edge_list = self.__filter_threshold(is_graph, network, th_layer_parameter[th][type_ca], filter_par_type)

                    # set_users_dict contains the set of users/nodes for each co-action/layer, for each threshold
                    # of the filtered graph
                    set_users_dict[type_ca][th] = self.__setUsers(is_graph_filtered, filtered_edge_list)

                    # compute weight statistics for filtered network
                    weight_stat_dict[type_ca][th] = self.__compute_edge_weight_statistics(is_graph_filtered, filtered_edge_list)
                    result_dict[th][type_ca].update(weight_stat_dict[type_ca][th])
                th += step

        # convert the result_dict which is a dictionary of dictionary in a flat list of dictionaries
        # (which can be transformed in dataframe). The dictionary of dictionary have been chosen so that in a second
        # moment could be updated with the statistics information on the weight filtered network, which can be computed
        # only after the read of the network
        result_filtering = []
        for key, layer_dict in result_dict.items():
            for row in layer_dict.values():
                result_filtering.append(row)

        self.lm.printl(f"{file_name}. Overlapping computation start.")
        overlapping_data = []
        th = min_th
        while th <= max_th:
            # Calculate overlap percentages of nodes/users for each couple of co-actions of the filtered graph
            for c1, c2 in combinations(self.dm.dict_path_ca.keys(), 2):
                self.lm.printl(f"{file_name}. {s}: {str(th)}, layer: {c1}-{c2} computing overlapping.")
                user_set1 = set_users_dict[c1][th]
                user_set2 = set_users_dict[c2][th]

                c1_name = co_action_map[c1]
                c2_name = co_action_map[c2]

                if user_set1 is None or user_set2 is None:
                    absolute_o = None
                    o_perc= None
                else:
                    _, absolute_o, o_coefficient = overlapping_coefficient(user_set1, user_set2)
                    o_perc = round(o_coefficient * 100)

                od = {
                    'layer1': c1_name,
                    'layer2': c2_name,
                    f'{s}': th,
                    'thLayer1': th_layer_parameter[th][c1],
                    'thLayer2': th_layer_parameter[th][c2],
                    'overlapping': absolute_o,
                    'percOverlapping': o_perc
                }

                overlapping_data.append(od)

            th += step

        result_df = pd.DataFrame(result_filtering)
        overlapping_df = pd.DataFrame(overlapping_data)

        if not absolute_th_mode:
            self.ch.save_dataframe(result_df, self.dm.path_analysis + f'{target_type}_{filter_par_type}_info_percNode.csv')
            self.ch.save_dataframe(overlapping_df, self.dm.path_analysis + f'{target_type}_{filter_par_type}_overlapping_percNode.csv')
        else:
            self.ch.save_dataframe(result_df, self.dm.path_analysis + f'{target_type}_{filter_par_type}_info_absolute.csv')
            self.ch.save_dataframe(overlapping_df, self.dm.path_analysis + f'{target_type}_{filter_par_type}_overlapping_absolute.csv')
        self.lm.printl(f"{file_name}. select_threshold_statistics completed.")

    # Multilayer Characterization Network Measures
    # -------------------------------------------
    def get_ML_layer_comparison(self):
        self.lm.printl(f"{file_name}. get_ML_layer_comparison start.")
        self.lm.printl(f"{file_name}. Extracting info multiplex network for co-actions {self.list_ca_str}")

        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt')]
        # TODO temporal multiplex network not implemented. There is only one file and temporal directory does not exist
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        # read multiplex network
        MG = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)

        for comparison in comparison_type:
            self.lm.printl(
                f"{file_name}. Computing {comparison} multiplex network")
            df = self.cm.to_df(ml.layer_comparison(MG, method=comparison))
            df.columns = ml.layers(MG)
            df.index = ml.layers(MG)
            df = df.reset_index(names='layer')
            filename = comparison.replace(".", "_") + "_" + net_filename_no_ext + '.csv'
            self.ch.save_dataframe(df, self.dm.path_analysis + filename)

        self.lm.printl(f"{file_name}. get_ML_layer_comparison finish.")

    def plot_ML_layer_comparison(self):
        self.lm.printl(f"{file_name}. plot_ML_layer_comparison start.")
        # Read all CSV files in the directory into dataframes
        dataframes = []
        file_names = []
        for comparison in comparison_type:
            fn = comparison.replace(".", "_") + "_multiplex_graph.csv"
            file_path = os.path.join(self.dm.path_analysis, fn)
            df = pd.read_csv(file_path)
            dataframes.append(df)
            file_names.append(fn)

        # Set up the figure for subplots
        # fig, axes = plt.subplots(3, 2, figsize=(15, 19), gridspec_kw={'wspace': 0.2, 'hspace': 0.25})
        # axes = axes.flatten()

        # Plot each heatmap in a subplot
        for i, (df, fn, comparison) in enumerate(zip(dataframes, file_names, comparison_type)):
            self.lm.printl(f"{file_name}. {comparison} start.")

            # Replace 'layer' values with mapped values from action_map_inverse_print
            df = df.rename(columns=co_action_column_print2)
            # Replace 'layer' values with mapped values from action_map_inverse_print
            df['layer'] = df['layer'].replace(co_action_column_print2)

            layer_order = list(co_action_column_print2.values())
            col2select = layer_order + ['layer']
            df = df[col2select]

            # Define the custom order for the 'layer' column
            df['layer'] = pd.Categorical(df['layer'], categories=layer_order, ordered=True)
            # Sort the dataframe by the 'layer' column
            df_sorted = df.sort_values('layer')
            aggregated_df = df_sorted.set_index('layer')

            if comparison == 'pearson.degree':
                min_v, max_v = -1, 1
                custom_palette = sns.color_palette("Spectral", as_cmap=True)
            else:
                min_v, max_v = 0, 1
                custom_palette = sns.color_palette("viridis", as_cmap=True).reversed()

            # Reverse the colormap
            # reversed_cmap = sns.color_palette("viridis", as_cmap=True).reversed()

            # Create a new figure for each plot
            fig, ax = plt.subplots(figsize=(8, 6.5))  # Adjust size if necessary

            if comparison == 'coverage.edges' or comparison == 'coverage.actors':
                sns.heatmap(aggregated_df, annot=True, fmt=".3f", cmap=custom_palette, cbar=True, ax=ax,
                            vmin=min_v, vmax=max_v, linewidths=0.5, linecolor='white', annot_kws={'size': 12})
            else:
                # Create a mask for the upper triangular part of the matrix excluding the diagonal
                mask = np.triu(np.ones_like(aggregated_df, dtype=bool), k=1)
                sns.heatmap(aggregated_df, mask=mask, annot=True, fmt=".3f", cmap=custom_palette, cbar=True, ax=ax,
                        vmin=min_v, vmax=max_v, linewidths=0.5, linecolor='white', annot_kws={'size': 12})

            # Customize labels
            ax.set_ylabel('')  # Remove "layer" axis label
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)  # Adjust rotation and font size
            ax.set_yticklabels(ax.get_yticklabels(), rotation=90, fontsize=16)

            # plt.suptitle("Metrics comparison", fontsize=16)
            plt.savefig(f"{self.dm.path_analysis}{comparison.replace('.', '_')}_layer_comparison_heatmaps.png", dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.show()
        self.lm.printl(f"{file_name}. plot_ML_layer_comparison completed.")

    def get_ML_summary(self):
        self.lm.printl(f"{file_name}. get_ML_summary start.")
        self.lm.printl(f"{file_name}. Extracting info summary of multiplex network for co-actions {self.list_ca_str}")

        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt')]

        # TODO temporal multiplex network not implemented. There is only one file and temporal directory does not exist
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        filename = "summary_" + net_filename_no_ext + '.csv'
        # read multiplex network
        MG = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)

        df = self.cm.to_df(ml.summary(MG))

        df = df.rename(
            columns={'n': 'nNodes', 'm': 'nEdges', 'dir': 'directed', "nc": "nConnectedComponents",
                     "slc": "sizeLargestComponent", "dens": "density",
                     "cc": "clusteringCoefficient", "apl": "averagePathLength", "dia": "diameter"})

        self.ch.save_dataframe(df, self.dm.path_analysis + filename)

        self.lm.printl(f"{file_name}. get_ML_summary finish.")
    # Multilayer Characterization Network Measures
    # -------------------------------------------

    def edge_weight_temporal_mean_std(self):
        self.lm.printl(f"{file_name}. edge_weight_temporal_mean_std start")
        result_dict = {}
        if os.path.exists(self.dm.path_processed + "temporal_weights.p"):
            result_dict = self.ch.load_object(self.dm.path_processed + "temporal_weights.p")
        else:
            for type_ca, dict_path in self.dm.dict_path_ca.items():
                self.lm.printl(f"{file_name}. edge_temporal_weight start for co-action {type_ca}")
                result_dict[type_ca] = {}
                edge_list_files_temporal = [pos_csv for pos_csv in os.listdir(dict_path["path_NF_edge_list_temporal"])
                                            if pos_csv.endswith('.p')]
                edge_list_files_temporal.sort()
                result_dict[type_ca]['mean_values'] = []
                result_dict[type_ca]['std_values'] = []
                result_dict[type_ca]['tw_values'] = []
                for elf in edge_list_files_temporal:
                    edge_list = self.ch.load_object(dict_path["path_NF_edge_list_temporal"] + elf)

                    tw_value = elf.split(' ')[0]
                    result_dict[type_ca]['tw_values'].append(tw_value)

                    weights = [e[tuple_index[W_VAR]] for e in edge_list]
                    result_dict[type_ca]['mean_values'].append(np.mean(weights))
                    result_dict[type_ca]['std_values'].append(np.std(weights))

                result_dict[type_ca]['mean_values'] = np.asarray(result_dict[type_ca]['mean_values'])
                result_dict[type_ca]['std_values'] = np.asarray(result_dict[type_ca]['std_values'])

                self.ch.save_object(result_dict, self.dm.path_processed + "temporal_weights.p")

        # Plot a separate visualization for each co-action
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            mean_values = result_dict[type_ca]['mean_values']
            tw_values = result_dict[type_ca]['tw_values']
            std_values = result_dict[type_ca]['std_values']
            plt.figure(figsize=(12, 10))
            # Plot the mean line
            plt.plot(tw_values, mean_values, color=color_dict[type_ca], linestyle='--')
            # Plot the filled area representing one standard deviation above and below the mean
            plt.fill_between(tw_values, mean_values - std_values, mean_values + std_values, color=color_dict[type_ca],
                             alpha=0.2, label=type_ca)

            plt.xlabel('Time Windows')
            plt.ylabel('Weight')
            plt.xticks(rotation=70)
            plt.title(f'{type_ca} Weight average and standard deviation ')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.savefig(dict_path["path_NF_analysis"] + "temporal_weight_distribution.png", dpi=800)

        # Plot a unique visualization of mean for all co-actions
        plt.figure(figsize=(12, 10))
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            mean_values = result_dict[type_ca]['mean_values']
            tw_values = result_dict[type_ca]['tw_values']
            plt.plot(tw_values, mean_values, color=color_dict[type_ca], linestyle='--', label=type_ca)
            plt.xlabel('Time Windows')
            plt.ylabel('Weight Average')
            plt.legend()
            plt.xticks(rotation=70)
            plt.title(f'Weight averages')

            plt.grid(True)

        plt.show()
        plt.savefig(f"{self.dm.path_analysis}temporal_weight_mean_distribution.png", dpi=800)

        self.lm.printl(f"{file_name}. edge_weight_temporal_mean_std completed")

    # def compute_metrics_not_filtered_info_networks(self, metrics_to_compute):
    #     self.lm.printl(f"{file_name}. compute_metrics_not_filtered_info_networks starts")
    #     metrics_list = []
    #     for type_ca, dict_path in self.dm.dict_path_ca.items():
    #         self.lm.printl(f"{file_name}. Reading not filtered info_edge_list for co-action {type_ca}")
    #         info_edge_list_files = [pos_csv for pos_csv in os.listdir(dict_path["info_edge_list_temporal"]) if
    #                                 pos_csv.endswith('.csv')]
    #
    #         if 'NF_edge_contribution' in metrics_to_compute:
    #             self.lm.printl(f"{file_name}. NF_edge_contribution for co-action {type_ca}")
    #             group_df = pd.DataFrame()
    #             for elf in info_edge_list_files:
    #                 self.lm.printl(f"{file_name}. Processing {elf} for co-action {type_ca}")
    #                 info_edge_list = self.ch.read_dataframe(dict_path["info_edge_list_temporal"] + elf, dtype=dtype)
    #
    #                 temp_df = info_edge_list.groupby(["userId1", "userId2"]).size().reset_index().rename(
    #                     columns={0: 'count'})
    #                 group_df = pd.concat([group_df, temp_df], ignore_index=True)
    #
    #             # Plotting the distribution
    #             self.pm.plot_histogram(dict_path["path_NF_analysis"], type_ca, group_df['count'],
    #                                    f"Number {type_ca} per edge", 'Frequency',
    #                                    f"Distribution of number of {type_ca} involved to construct an edge",
    #                                    "number_co_action_distribution.png")
    #
    #     self.lm.printl(f"{file_name}. compute_metrics_not_filtered_info_networks starts")


    # def __compute_edge_list_network_metrics(self, edge_list, type_ca, path_analysis, metrics_to_compute, metrics_dict):
    #     self.lm.printl(f"{file_name}. Reading not filtered single layer network (edge lists) for co-action {type_ca}")

        # if 'weight_statistics' in metrics_to_compute:
        #     stats_dict = self.__compute_edge_weight_statistics(edge_list=edge_list)
        #     metrics_dict.update(stats_dict)
        #     self.__show_edge_weight_distribution(path_analysis, type_ca, edge_list=edge_list)
        #     self.lm.printl(f"{file_name}. weight_statistics computed for co-action {type_ca}")
        #
        # if 'nNodes' in metrics_to_compute:
        #     # compute nNodes. edge list in format userId1, userId2, weight
        #     user_set1 = set([e[0] for e in edge_list])
        #     user_set2 = set([e[1] for e in edge_list])
        #     user_set = user_set1.union(user_set2)
        #     metrics_dict['nNodes'] = len(user_set)
        #     self.lm.printl(f"{file_name}. nNodes computed for co-action {type_ca}")
        #
        # if 'nEdges' in metrics_to_compute:
        #     metrics_dict['nEdges'] = len(edge_list)
        #     self.lm.printl(f"{file_name}. nEdges computed for co-action {type_ca}")
        #
        # if "node_topEdge_trend" in metrics_to_compute:
        #     self.__plot_node_topEdge_trend(path_analysis, type_ca, edge_list=edge_list)
        #     self.lm.printl(f"{file_name}. node_topEdge_trend computed for co-action {type_ca}")

        # return metrics_dict

    # def __compute_graph_network_metrics(self, G, type_ca, path_analyisis, metrics_to_compute, metrics_dict):
    #     self.lm.printl(f"{file_name}. Reading filtered single layer network (graph) for co-action {type_ca}")

        # Basic graph properties
        # if 'nNodes' in metrics_to_compute:
        #     metrics_dict['nNodes'] = G.number_of_nodes()
        #     self.lm.printl(f"{file_name}. nNodes computed for co-action {type_ca}")
        #
        # if 'nEdges' in metrics_to_compute:
        #     metrics_dict['nEdges'] = G.number_of_edges()
        #     self.lm.printl(f"{file_name}. nEdges computed for co-action {type_ca}")


        # if "weight_statistics" in metrics_to_compute:
        #     stats_dict = self.__compute_edge_weight_statistics(G=G)
        #     metrics_dict.update(stats_dict)
        #     self.lm.printl(f"{file_name}. weight_statistics computed for co-action {type_ca}")
        #     self.__show_edge_weight_distribution(path_analyisis, type_ca, G=G)
        #     self.lm.printl(f"{file_name}. weight_edge_distribution computed for co-action {type_ca}")

        # Assortativity
        # if "assortativity" in metrics_to_compute:
        #     assortativity = nx.degree_assortativity_coefficient(G)
        #     metrics_dict['assortativity'] = assortativity
        #     self.lm.printl(f"{file_name}. assortativity computed for co-action {type_ca}")
        #
        # # Degree centrality
        # if 'degree_centrality' in metrics_to_compute:
        #     degree_centrality = nx.degree_centrality(G)
        #     metrics_dict['degreeCentrality'] = degree_centrality
        #     self.lm.printl(f"{file_name}. degreeCentrality computed for co-action {type_ca}")
        #
        # # Betweeness centrality
        # if 'betweenness_centrality' in metrics_to_compute:
        #     betweenness_centrality = nx.betweenness_centrality(G)
        #     metrics_dict['betweennessCentrality'] = betweenness_centrality
        #     self.lm.printl(f"{file_name}. betweennessCentrality computed for co-action {type_ca}")
        #
        # # Closeness centrality
        # if 'closeness_centrality' in metrics_to_compute:
        #     closeness_centrality = nx.closeness_centrality(G)
        #     metrics_dict['closenessCentrality'] = closeness_centrality
        #     self.lm.printl(f"{file_name}. closenessCentrality computed for co-action {type_ca}")
        #
        # # Shortest path
        # if 'shortest_path_lengths' in metrics_to_compute:
        #     shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        #     metrics_dict['shortestPathLengths'] = shortest_path_lengths
        #     self.lm.printl(f"{file_name}. shortestPathLenghts computed for co-action {type_ca}")
        #
        # # Eccentricity
        # if 'eccentricity' in metrics_to_compute:
        #     eccentricity = nx.eccentricity(G)
        #     metrics_dict['eccentricity'] = eccentricity

        # if 'degree_distribution' in metrics_to_compute:
        #     self.__show_degree_distribution(path_analyisis, type_ca, G)
        #     self.lm.printl(f"{file_name}. degree_distribution computed for co-action {type_ca}")


        # return metrics_dict