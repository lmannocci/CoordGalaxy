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


class NodeMeasure:
    def __init__(self, list_ca, dict_ca_filter, icm, dm, type_algorithm, cda):
        self.lm = LogManager('main')
        self.ch = Checkpoint()
        self.cm = ConversionManager()

        self.list_ca = list_ca
        self.type_ca = self.list_ca[0].get_co_action()
        self.dict_ca_filter = dict_ca_filter
        self.icm = icm
        self.dm = dm

        self.type_algorithm = type_algorithm
        self.cda = cda

    def __compute_node_metrics_df(self, G, metrics=None):
        data = {"userId": list(G.nodes)}

        # i add the community information for this single layer results
        # (it might be useful, even if currently it is not used)
        node_communities = nx.get_node_attributes(G, "group")
        data["community"] = [node_communities[node] for node in G.nodes]

        if metrics is None:
            metrics = available_node_metrics
        else:
            metrics = [m for m in metrics if m in available_node_metrics]

        if "degree_centrality" in metrics:
            start = time.time()
            degree_centrality = nx.degree_centrality(G)
            data["degree_centrality"] = [degree_centrality[node] for node in G.nodes]
            self.lm.printl(f"Degree centrality computed in {time.time() - start:.2f} seconds")

        if "betweenness_centrality" in metrics:
            start = time.time()
            betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
            data["betweenness_centrality"] = [betweenness_centrality[node] for node in G.nodes]
            self.lm.printl(f"Betweenness centrality computed in {time.time() - start:.2f} seconds")

        if "closeness_centrality" in metrics:
            start = time.time()
            closeness_centrality = nx.closeness_centrality(G)
            data["closeness_centrality"] = [closeness_centrality[node] for node in G.nodes]
            self.lm.printl(f"Closeness centrality computed in {time.time() - start:.2f} seconds")

        if "eigenvector_centrality" in metrics or "bridging_eigenvector_centrality" in metrics:
            start = time.time()
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            data["eigenvector_centrality"] = [eigenvector_centrality[node] for node in G.nodes]
            self.lm.printl(f"Eigenvector centrality computed in {time.time() - start:.2f} seconds")

        if "local_clustering_coefficient" in metrics:
            start = time.time()
            local_clustering_coefficient = nx.clustering(G)
            data["local_clustering_coefficient"] = [local_clustering_coefficient[node] for node in G.nodes]
            self.lm.printl(f"Local clustering coefficient computed in {time.time() - start:.2f} seconds")

        if "page_rank" in metrics:
            start = time.time()
            page_rank = nx.pagerank(G)
            data["page_rank"] = [page_rank[node] for node in G.nodes]
            self.lm.printl(f"PageRank computed in {time.time() - start:.2f} seconds")

        # Create DataFrame
        return pd.DataFrame(data)

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------

    def compute_node_metrics(self, metrics, merge_existing):
        """

        :param metrics: List of node metrics to be computed.
        :param merge_existing: Boolean value, if true, the method does not overwrite existing dataframe, but it adds a
        new column to the existing dataframe. It is useful when I have already computed some measure and I want to
        compute a new measure, that must be added to the existing dataframe.
        :return:
        """
        self.icm.check_node_metrics(metrics)
        if self.type_algorithm == 'one-layer': # single layer
            layer = self.list_ca[0].get_co_action()
        elif self.cda.get_algorithm_name() in flatten_algorithm: # flattened network
            layer = self.cda.get_algorithm_name()
        else:
            self.lm.printl(f"{file_name}. compute_node_metrics: multi-layer case not managed.")
            return  # multi-layer case not managed
        self.lm.printl(f"{file_name}. compute_node_metrics: {layer} started.")

        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_community_graph) if pos_csv.endswith('.p')]
        net_filename = graph_files[0]
        # read single layer network or flattened network
        G = self.ch.load_object(self.dm.path_community_graph + net_filename)

        node_metrics_df = self.__compute_node_metrics_df(G, metrics)

        node_metrics_df['layer'] = layer
        if merge_existing:
            self.ch.update_columns_dataframe(node_metrics_df,
                                             self.dm.path_community_analysis + f"{layer}_node_metrics.csv",
                                             ['userId', 'layer'], dtype)
        else:
            self.ch.save_dataframe(node_metrics_df, self.dm.path_community_analysis + f"{layer}_node_metrics.csv")

        self.lm.printl(f"{file_name}. compute_metrics_node: {layer} completed.")

