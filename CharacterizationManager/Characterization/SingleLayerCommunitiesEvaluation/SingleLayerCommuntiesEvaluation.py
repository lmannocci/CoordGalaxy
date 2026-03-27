from CharacterizationManager.Characterization.CommunitiesEvaluation.CommunitiesEvaluation import CommunitiesEvaluation
from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.common_variables import *
from utils.PlotManager.PlotManager import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *

import os
import networkx as nx
import numpy as np
import pandas as pd
import time
from cdlib import evaluation, NodeClustering

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class SingleLayerCommunitiesEvaluation:
    def __init__(self, dataset_name, user_fraction, type_filter, list_ca, dict_ca_filter, icm, dm, type_algorithm, cda):
        self.lm = LogManager('main')
        self.ch = Checkpoint()
        self.cm = ConversionManager()

        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter

        self.list_ca = list_ca
        self.type_ca = self.list_ca[0].get_co_action()
        self.dict_ca_filter = dict_ca_filter
        self.icm = icm
        self.dm = dm
       

        self.type_algorithm = type_algorithm
        self.cda = cda
        self.pm = PlotManager()

        self.ce = CommunitiesEvaluation(self.lm)

    def __plot_size_communities(self, df):
        self.pm.plot_line(self.dm.path_community_analysis, self.type_ca, df['group'], df['nUsers'], "group", 'nUsers',
                          'Size communities distribution', "size_communities_distribution.png",
                          marker='o', markersize=3)

    # Function to get nodes by community
    def __get_community_nodes(self, graph, community_label="group"):
        communities = {}
        for node, data in graph.nodes(data=True):
            community = data.get(community_label)
            if community not in communities:
                communities[community] = []
            communities[community].append(node)
        return communities

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    def compute_statistics_communities(self):
        self.lm.printl(f"{file_name}. compute_statistics_communities start.")

        df = self.ch.read_dataframe(self.dm.path_user_dataframe + "com_df.csv", dtype=dtype)
        agg_df = df.groupby(['group']).size().reset_index(name='nUsers')
        info_com_df = agg_df['nUsers'].describe(
            percentiles=[.5, .6, .7, .75, .90, .91, .92, .93, .94, .95, .97, .98]).reset_index(name=self.type_ca).T
        info_com_df.columns = info_com_df.iloc[0]
        info_com_df = info_com_df.iloc[1:]
        info_com_df = info_com_df.reset_index(names="layer")
        info_com_df = info_com_df.reset_index(drop=True)
        # remove the name of the index
        info_com_df = info_com_df.rename_axis(None, axis=1)
        info_com_df = info_com_df.rename(
            columns={'count': 'nCommunities', 'mean': 'avgUsers', 'std': 'stdUsers', 'min': 'minUsers',
                     'max': 'maxUsers'})

        self.ch.save_dataframe(info_com_df, self.dm.path_community_analysis + f"{repr(self.cda)}_statistics_communities.csv")

        self.__plot_size_communities(agg_df)

        self.lm.printl(f"{file_name}. compute_statistics_communities completed.")

    # Optimized function to calculate community metrics with timing and progress messages
    def compute_metrics_communities(self, community_size_th, community_label="group", weight_label="w_"):
        type_ca = self.list_ca[0].get_co_action()
        self.lm.printl(f"{file_name}. compute_metrics_community co-action: {type_ca} started.")
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_community_graph) if
                       pos_csv.endswith('.p')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        # read single network
        G = self.ch.load_object(self.dm.path_community_graph + net_filename)

        communities = self.__get_community_nodes(G, community_label)
        metrics = []
        total_weight = G.size(weight=weight_label)

        for community, nodes in communities.items():
            num_nodes = len(nodes)
            # Community Size
            if num_nodes >= community_size_th:
                subgraph = G.subgraph(nodes)
                num_edges = subgraph.number_of_edges()
                community_data = {"community": community}

                start_time = time.time()
                size = num_nodes
                community_data["size"] = size
                self.lm.printl(f"Community {community}. Size computed in {time.time() - start_time:.4f} seconds")

                # Internal Density
                #             start_time = time.time()
                #             density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
                #             community_data["density"] = density
                #             print(f"Community {community}. Density computed in {time.time() - start_time:.4f} seconds")

                # Internal Density (using NetworkX built-in function)
                start_time = time.time()
                density = nx.density(subgraph)
                community_data["density"] = density
                self.lm.printl(f"Community {community}. Density computed in {time.time() - start_time:.4f} seconds")

                # Average Weight of Internal Edges
                start_time = time.time()
                weights = [data[weight_label] for u, v, data in subgraph.edges(data=True)]
                avg_weight = np.mean(weights) if weights else 0
                community_data["avg_weight"] = avg_weight
                self.lm.printl(f"Community {community}. Avg weight computed in {time.time() - start_time:.4f} seconds")
                
                # Standard Deviation of Weights of Internal Edges
                start_time = time.time()
                std_weight = np.std(weights) if weights else 0
                community_data["std_weight"] = std_weight
                self.lm.printl(f"Community {community}. Std weight computed in {time.time() - start_time:.4f} seconds")

                # Median Weight of Internal Edges
                start_time = time.time()
                median_weight = np.median(weights) if weights else 0
                community_data["median_weight"] = median_weight
                self.lm.printl(f"Community {community}. Median weight computed in {time.time() - start_time:.4f} seconds")

                # MAD of Weights of Internal Edges
                start_time = time.time()
                mad_weight = np.median(np.abs(weights - np.median(weights))) if weights else 0
                community_data["mad_weight"] = mad_weight
                self.lm.printl(f"Community {community}. MAD weight computed in {time.time() - start_time:.4f} seconds")

                # Conductance (Using CDlib)
                start_time = time.time()
                cdlib_community = NodeClustering([nodes], G, method_name="custom")
                conductance_score = evaluation.conductance(G, cdlib_community).score
                community_data["conductance"] = conductance_score
                self.lm.printl(f"Community {community}. Conductance computed in {time.time() - start_time:.4f} seconds")

                # Average Degree (Internal)
                start_time = time.time()
                degrees = [subgraph.degree(n) for n in nodes]
                avg_degree = np.mean(degrees) if degrees else 0
                community_data["avg_degree"] = avg_degree
                self.lm.printl(f"Community {community}. Avg degree computed in {time.time() - start_time:.4f} seconds")

                # Clustering Coefficient (Internal)
                start_time = time.time()
                clustering_coeffs = nx.clustering(subgraph, weight=weight_label).values()
                avg_clustering = np.mean(list(clustering_coeffs)) if clustering_coeffs else 0
                community_data["avg_clustering"] = avg_clustering
                self.lm.printl(f"Community {community}. Avg clustering computed in {time.time() - start_time:.4f} seconds")

                # Assortativity (Degree Assortativity)
                start_time = time.time()
                try:
                    assortativity = nx.degree_assortativity_coefficient(subgraph)
                except ZeroDivisionError:
                    assortativity = None  # Handle cases where the calculation is not feasible
                community_data["assortativity"] = assortativity
                self.lm.printl(f"Community {community}. Assortativity computed in {time.time() - start_time:.4f} seconds")

                metrics.append(community_data)

        df_metrics = pd.DataFrame(metrics)

        self.ch.save_dataframe(df_metrics, self.dm.path_community_analysis + f"{type_ca}_th_size_{str(community_size_th)}_metrics_communities.csv")

        self.lm.printl(f"{file_name}. compute_metrics_community co-action: {type_ca} completed.")

    def compute_coordination_communities(self, community_size_th, community_label, weight_label):
        if self.type_algorithm == 'one-layer': # single layer
            layer = self.list_ca[0].get_co_action()
        elif self.cda.get_algorithm_name() in flatten_algorithm: # flattened network
            layer = self.cda.get_algorithm_name()
        self.lm.printl(f"{file_name}. compute_coordination_communities co-action: {layer} started.")
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_community_graph) if
                       pos_csv.endswith('.p')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        # read single network
        G = self.ch.load_object(self.dm.path_community_graph + net_filename)

        communities = self.__get_community_nodes(G, community_label)
        metrics = []
        total_weight = G.size(weight=weight_label)

        i = 0
        for community, nodes in communities.items():
            i += 1
            num_nodes = len(nodes)
            self.lm.printl(f"Community {i}/{len(communities)}. Processing...")
            # Community Size
            if (community_size_th is not None and num_nodes >= community_size_th) or (community_size_th is None):  
                subgraph = G.subgraph(nodes)
                num_edges = subgraph.number_of_edges()
                community_data = {"community": community}

                start_time = time.time()
                size = num_nodes
                community_data["size"] = size
                # self.lm.printl(f"Community {community}. Size computed in {time.time() - start_time:.4f} seconds")

                # Average Weight of Internal Edges
                start_time = time.time()
                weights = [data[weight_label] for u, v, data in subgraph.edges(data=True)]
                avg_weight = np.mean(weights) if weights else 0
                community_data["avg_weight"] = avg_weight
                # self.lm.printl(f"Community {community}. Avg weight computed in {time.time() - start_time:.4f} seconds")
                
                # Standard Deviation of Weights of Internal Edges
                start_time = time.time()
                std_weight = np.std(weights) if weights else 0
                community_data["std_weight"] = std_weight
                # self.lm.printl(f"Community {community}. Std weight computed in {time.time() - start_time:.4f} seconds")

                # Median Weight of Internal Edges
                start_time = time.time()
                median_weight = np.median(weights) if weights else 0
                community_data["median_weight"] = median_weight
                # self.lm.printl(f"Community {community}. Median weight computed in {time.time() - start_time:.4f} seconds")

                # MAD of Weights of Internal Edges
                start_time = time.time()
                mad_weight = np.median(np.abs(weights - np.median(weights))) if weights else 0
                community_data["mad_weight"] = mad_weight
                # self.lm.printl(f"Community {community}. MAD weight computed in {time.time() - start_time:.4f} seconds")

                metrics.append(community_data)

        df_metrics = pd.DataFrame(metrics)
        if community_size_th == 0:
            th_size_str = ""
        else:
            th_size_str = f"_th_size_{str(community_size_th)}"
        self.ch.save_dataframe(df_metrics, self.dm.path_community_analysis + f"{layer}{th_size_str}_coordination_communities.csv")

        self.lm.printl(f"{file_name}. compute_coordination_communities co-action: {layer} completed.")


    def compute_metrics_node_communities(self, metrics, th_size, restrict_neighbors, merge_existing):
        if self.type_algorithm == 'one-layer': # single layer
            layer = self.list_ca[0].get_co_action()
        elif self.cda.get_algorithm_name() in flatten_algorithm: # flattened network
            layer = self.cda.get_algorithm_name()

        self.lm.printl(f"{file_name}. compute_metrics_node_communities: {layer} started.")

        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_community_graph) if pos_csv.endswith('.p')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        # read single layer network or flattened network
        G = self.ch.load_object(self.dm.path_community_graph + net_filename)

        node_metrics_df = self.ce.compute_node_metrics_df(G, metrics, th_size, restrict_neighbors)

        node_metrics_df['layer'] = layer
        if merge_existing:
            self.ch.update_columns_dataframe(node_metrics_df,
                                             self.dm.path_community_analysis + f"{layer}_th_size_{str(th_size)}_node_metrics_communities.csv",
                                             ['node', 'community', 'layer'], dtype)
        else:
            self.ch.save_dataframe(node_metrics_df, self.dm.path_community_analysis + f"{layer}_th_size_{str(th_size)}_node_metrics_communities.csv")

        # node_metrics_df['layer'] = type_ca
        # if merge_existing:
        #     self.ch.update_columns_dataframe(node_metrics_df,
        #                                      self.dm.path_community_analysis + f"{type_ca}_th_size_{str(th_size)}_node_metrics_communities.csv",
        #                                      ['node', 'community', 'layer'], dtype)
        # else:
        #     self.ch.save_dataframe(node_metrics_df, self.dm.path_community_analysis + f"{type_ca}_th_size_{str(th_size)}_node_metrics_communities.csv")

        self.lm.printl(f"{file_name}. compute_metrics_node_communities: {layer} completed.")

    def validate_communities(self):
        self.lm.printl(f"{file_name}. validate_communities start.")
        if self.type_algorithm == 'one-layer': # single layer
            layer = self.list_ca[0].get_co_action()
            
            data_df = self.ch.read_dataframe(f"{self.dm.path_dataset}{self.user_fraction}_{self.type_filter}_{self.dataset_name}_{layer}.csv", dtype)

        elif self.cda.get_algorithm_name() in flatten_algorithm: # flattened network
            layer = self.cda.get_algorithm_name()

            # Read all layers and concatenate the users
            df_list = []
            for ca in self.list_ca:
                ca_type = ca.get_co_action()
                df = self.ch.read_dataframe(f"{self.dm.path_dataset}{self.user_fraction}_{self.type_filter}_{self.dataset_name}_{ca_type}.csv", dtype)
                df_list.append(df)
            data_df = pd.concat(df_list)


        pre_df = data_df[['userId', 'isControl']]
        pre_df = pre_df.drop_duplicates()

        user_df = self.ch.read_dataframe(self.dm.path_user_dataframe + "com_df.csv", dtype=dtype)
       
        post_df = user_df.merge(pre_df, on='userId', how='inner')

        # Count users before and after
        n_pre = len(pre_df)
        n_post = len(post_df)

        # Split by class
        n_pre_control = sum(pre_df['isControl'])
        n_pre_coord   = sum(~pre_df['isControl'])
        n_post_control = sum(post_df['isControl'])
        n_post_coord   = sum(~post_df['isControl'])

        # Metrics
        overall_filtering = (n_pre - n_post) / n_pre
        control_filtering = (n_pre_control - n_post_control) / n_pre_control
        coord_filtering = (n_pre_coord - n_post_coord) / n_pre_coord
        coord_retention   = n_post_coord / n_pre_coord
        precision_like    = n_post_coord / n_post

        # build the lines
        lines = [
            f"Overall filtering rate: {overall_filtering:.2%}",
            f"Control filtering rate: {control_filtering:.2%}",
            f"Coordinated filtering rate: {coord_filtering:.2%}",
            # f"Coordinated retention rate: {coord_retention:.2%}",
            # f"Purity (precision-like): {precision_like:.2%}"
        ]
        self.lm.printl(lines)
        self.ch.save_txt(lines, self.dm.path_community_analysis + f"filtering_rates.txt")

        # --- Count True/False per group
        group_counts = (
            post_df.groupby(['group', 'isControl'])
            .size()
            .unstack('isControl', fill_value=0)
            .rename(columns={True: 'nControl', False: 'nCoord'})
            .reset_index()
        )

        # --- Compute percentages
        group_counts['nTotal'] = group_counts['nControl'] + group_counts['nCoord']
        group_counts['percControl'] = group_counts['nControl'] / group_counts['nTotal']
        group_counts['percCoord'] = group_counts['nCoord'] / group_counts['nTotal']

        # --- Reset index if you want a regular dataframe
        group_stats = group_counts.reset_index(drop=True)

        # Compute purity
        group_stats['purity'] = group_stats[['nControl', 'nCoord']].max(axis=1) / group_counts['nTotal']

        # Sort for convenience
        # group_counts = group_counts.sort_values('purity', ascending=False)

        self.ch.save_dataframe(group_stats, self.dm.path_community_analysis + f"{layer}_validation_communities.csv")

        self.pm.plot_histogram(self.dm.path_community_analysis, layer, group_stats['purity'], 'Purity', 'Number of Groups',
                               'Distribution of Group Purity', f"{layer}_purity_distribution.png")
        # plt.figure(figsize=(8, 5))
        # sns.histplot(group_stats['purity'], bins=20, kde=True)
        # plt.title('Distribution of Group Purity', fontsize=16)
        # plt.xlabel('Purity', fontsize=14)
        # plt.ylabel('Number of Groups', fontsize=14)
        # plt.tight_layout()
        # plt.savefig(self.dm.path_community_analysis + f"{layer}_purity_distribution.png", dpi=dpi)
        # plt.show()

        self.lm.printl(f"{file_name}. validate_communities completed.")