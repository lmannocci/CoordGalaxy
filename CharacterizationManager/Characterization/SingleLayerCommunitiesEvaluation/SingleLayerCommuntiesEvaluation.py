from CharacterizationManager.Characterization.CommunitiesEvaluation.CommunitiesEvaluation import CommunitiesEvaluation
from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.common_variables import *
from utils.PlotManager.PlotManager import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *
from utils.decorator_definition import log_method

import os
import networkx as nx
import numpy as np
import pandas as pd
import time
from cdlib import evaluation, NodeClustering
from typing import Any, Sequence

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class SingleLayerCommunitiesEvaluation:
    def __init__(
        self,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str | None,
        list_ca: Sequence[Any],
        dict_ca_filter: dict[str, Any],
        icm: Any,
        dm: DirectoryManager,
        type_algorithm: str,
        cda: Any,
    ) -> None:
        """
            Create the evaluator for single-layer and flattened community outputs.
            :param dataset_name: Dataset directory name.
            :param user_fraction: User-selection fraction used in path resolution.
            :param type_filter: User-selection strategy name.
            :param list_ca: Co-action objects included in the characterization.
            :param dict_ca_filter: Filter configuration by co-action id.
            :param icm: Integrity constraint manager.
            :param dm: Directory manager with community-analysis paths.
            :param type_algorithm: Algorithm type detected by DirectoryManager.
            :param cda: Community-detection algorithm configuration.
            :return: None.
        """
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

    def _plot_size_communities(self, df: pd.DataFrame) -> None:
        """
            Plot the community-size distribution.
            :param df: Dataframe with group and nUsers columns.
            :return: None. Plot is saved to the community analysis directory.
        """
        self.pm.plot_line(self.dm.path_community_analysis, self.type_ca, df['group'], df['nUsers'], "group", 'nUsers',
                          'Size communities distribution', "size_communities_distribution.png",
                          marker='o', markersize=3)

    # Function to get nodes by community
    def _get_community_nodes(self, graph: nx.Graph, community_label: str = "group") -> dict[Any, list[Any]]:
        """
            Group graph nodes by their community attribute.
            :param graph: NetworkX graph with a community attribute on nodes.
            :param community_label: Node attribute containing the community id.
            :return: Dictionary mapping community id to node list.
        """
        communities = {}
        for node, data in graph.nodes(data=True):
            community = data.get(community_label)
            if community not in communities:
                communities[community] = []
            communities[community].append(node)
        return communities


    def _filtered_co_action_file_path(self, action_name: str) -> str:
        """
            Return the filtered co-action file path, preferring the dataset-directory scoped filename template.
            :param action_name: [str] Co-action filename stem.
            :return: [str] Path to the filtered co-action CSV.
        """
        co_action_data_path = self._co_action_data_path()
        filename = f"{action_name}.csv"
        path = f"{co_action_data_path}{filename}"
        if os.path.exists(path):
            return path

        legacy_filename = f"{self.dataset_name}_{action_name}.csv"
        legacy_path = f"{co_action_data_path}{legacy_filename}"
        if os.path.exists(legacy_path):
            return legacy_path

        if self.user_fraction is not None:
            old_filtered_filename = f"{self.user_fraction}_{self.type_filter}_{action_name}.csv"
            old_filtered_path = f"{self.dm.path_co_action_data}{old_filtered_filename}"
            if os.path.exists(old_filtered_path):
                return old_filtered_path

            old_legacy_filename = f"{self.user_fraction}_{self.type_filter}_{self.dataset_name}_{action_name}.csv"
            old_legacy_path = f"{self.dm.path_co_action_data}{old_legacy_filename}"
            if os.path.exists(old_legacy_path):
                return old_legacy_path

        return legacy_path

    def _co_action_data_path(self) -> str:
        """
            Return the directory to read co-action artifacts from.
            :return: [str] Selected-user co-action directory when present, otherwise the base co_action_data directory.
        """
        filtered_path = self.dm.get_co_action_data_path(self.user_fraction, self.type_filter)
        if self.user_fraction is not None and os.path.isdir(filtered_path):
            return filtered_path
        return self.dm.path_co_action_data

    def _has_is_control_label(self, data_df: pd.DataFrame, context: str) -> bool:
        """
            Check whether the dataframe includes the optional isControl validation label.
            :param data_df: [pd.DataFrame] Source co-action dataframe.
            :param context: [str] Description of the input being validated, used in log messages.
            :return: [bool] True when isControl is available, otherwise False.
        """
        if "isControl" in data_df.columns:
            return True
        self.lm.printl(f"{file_name}. validate_communities skipped: isControl label not found in {context}.")
        return False

    def _normalize_is_control_column(self, data_df: pd.DataFrame, context: str) -> pd.DataFrame:
        """
            Normalize the isControl label to boolean values before validation.
            :param data_df: [pd.DataFrame] Dataframe containing an isControl column.
            :param context: [str] Description of the input being normalized, used in error messages.
            :return: [pd.DataFrame] Copy of the dataframe with boolean isControl values.
        """
        normalized_df = data_df.copy()
        if pd.api.types.is_bool_dtype(normalized_df["isControl"]):
            normalized_df["isControl"] = normalized_df["isControl"].astype(bool)
            return normalized_df

        true_values = {"true", "1", "yes", "y", "control"}
        false_values = {"false", "0", "no", "n", "coord", "coordinated"}

        def normalize_value(value: Any) -> bool | None:
            """
                Convert one label value to bool when it matches a supported encoding.
                :param value: [Any] Raw isControl value.
                :return: [bool | None] Boolean value, or None when the value is unknown.
            """
            if pd.isna(value):
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, np.integer)):
                if value == 1:
                    return True
                if value == 0:
                    return False
            value_str = str(value).strip().lower()
            if value_str in true_values:
                return True
            if value_str in false_values:
                return False
            return None

        converted = normalized_df["isControl"].map(normalize_value)
        if converted.isna().any():
            invalid_values = sorted(normalized_df.loc[converted.isna(), "isControl"].dropna().astype(str).unique())
            message = f"{file_name}. Unknown isControl values in {context}: {invalid_values}"
            self.lm.printl(message)
            raise ValueError(message)

        normalized_df["isControl"] = converted.astype(bool)
        return normalized_df

    def _build_label_count_dataframe(self, post_df: pd.DataFrame, groupby_columns: list[str]) -> pd.DataFrame:
        """
            Count control and coordinated users for each validation group.
            :param post_df: [pd.DataFrame] Community dataframe with a boolean isControl column.
            :param groupby_columns: [list[str]] Columns used for grouping, excluding isControl.
            :return: [pd.DataFrame] Counts and percentages for control and coordinated users.
        """
        group_counts = (
            post_df.groupby(groupby_columns + ["isControl"])
            .size()
            .unstack("isControl", fill_value=0)
        )
        for label_value in [True, False]:
            if label_value not in group_counts.columns:
                group_counts[label_value] = 0

        group_counts = group_counts.reindex(columns=[True, False], fill_value=0)
        group_counts = group_counts.rename(columns={True: "nControl", False: "nCoord"}).reset_index()
        group_counts["nTotal"] = group_counts["nControl"] + group_counts["nCoord"]
        group_counts["percControl"] = group_counts["nControl"] / group_counts["nTotal"]
        group_counts["percCoord"] = group_counts["nCoord"] / group_counts["nTotal"]
        group_counts["purity"] = group_counts[["nControl", "nCoord"]].max(axis=1) / group_counts["nTotal"]
        return group_counts

    def _safe_ratio(self, numerator: int | float, denominator: int | float) -> float:
        """
            Divide two values and return 0.0 when the denominator is zero.
            :param numerator: [int | float] Numerator.
            :param denominator: [int | float] Denominator.
            :return: [float] Ratio value.
        """
        if denominator == 0:
            return 0.0
        return numerator / denominator

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def compute_community_summary_statistics(self) -> None:
        """
            Compute descriptive statistics for single-layer communities.
            :return: None. Statistics and plots are saved to the community analysis directory.
        """
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

        self._plot_size_communities(agg_df)

    # Optimized function to calculate community metrics with timing and progress messages
    @log_method
    def compute_metrics_communities(
        self,
        community_size_th: int,
        community_label: str = "group",
        weight_label: str = "w_"
    ) -> None:
        """
            Compute graph metrics for communities above a size threshold.
            :param community_size_th: Minimum community size to include.
            :param community_label: Node attribute containing the community id.
            :param weight_label: Edge attribute containing edge weights.
            :return: None. Metrics are saved to the community analysis directory.
        """
        type_ca = self.list_ca[0].get_co_action()
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_community_graph) if
                       pos_csv.endswith('.p')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        # read single network
        G = self.ch.load_object(self.dm.path_community_graph + net_filename)

        communities = self._get_community_nodes(G, community_label)
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

    @log_method
    def compute_community_edge_weight_statistics(
        self,
        community_size_th: int | None,
        community_label: str,
        weight_label: str
    ) -> None:
        """
            Compute edge-weight coordination statistics for each community.
            :param community_size_th: Minimum community size to include, or None to include all communities.
            :param community_label: Node attribute containing the community id.
            :param weight_label: Edge attribute containing edge weights.
            :return: None. Coordination statistics are saved to the community analysis directory.
        """
        if self.type_algorithm == 'one-layer': # single layer
            layer = self.list_ca[0].get_co_action()
        elif self.cda.get_algorithm_name() in flatten_algorithm: # flattened network
            layer = self.cda.get_algorithm_name()
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_community_graph) if
                       pos_csv.endswith('.p')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        # read single network
        G = self.ch.load_object(self.dm.path_community_graph + net_filename)

        communities = self._get_community_nodes(G, community_label)
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


    @log_method
    def compute_metrics_node_communities(
        self,
        metrics: Sequence[str] | None,
        th_size: int | None,
        restrict_neighbors: bool,
        merge_existing: bool
    ) -> None:
        """
            Compute node metrics inside communities.
            :param metrics: Node metric names to compute, or None for defaults.
            :param th_size: Minimum community size to include.
            :param restrict_neighbors: Whether neighbor-based metrics should only count selected-community neighbors.
            :param merge_existing: Whether to merge computed columns with an existing metrics file.
            :return: None. Node-community metrics are saved to the community analysis directory.
        """
        if self.type_algorithm == 'one-layer': # single layer
            layer = self.list_ca[0].get_co_action()
        elif self.cda.get_algorithm_name() in flatten_algorithm: # flattened network
            layer = self.cda.get_algorithm_name()

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

    @log_method
    def validate_communities(self) -> None:
        """
            Validate communities against available isControl labels in the source co-action data.
            :return: None. Validation dataframes and plots are saved to the community analysis directory.
        """
        if self.type_algorithm == 'one-layer': # single layer
            layer = self.list_ca[0].get_co_action()
            
            data_df = self.ch.read_dataframe(self._filtered_co_action_file_path(action_map[layer]), dtype)
            if not self._has_is_control_label(data_df, layer):
                return
            data_df = self._normalize_is_control_column(data_df, layer)

        elif self.cda.get_algorithm_name() in flatten_algorithm: # flattened network
            layer = self.cda.get_algorithm_name()

            # Read all layers and concatenate the users
            df_list = []
            for ca in self.list_ca:
                ca_type = ca.get_co_action()
                df = self.ch.read_dataframe(self._filtered_co_action_file_path(action_map[ca_type]), dtype)
                if not self._has_is_control_label(df, ca_type):
                    return
                df_list.append(df)
            data_df = pd.concat(df_list)
            data_df = self._normalize_is_control_column(data_df, layer)


        pre_df = data_df[['userId', 'isControl']]
        pre_df = pre_df.drop_duplicates()

        user_df = self.ch.read_dataframe(self.dm.path_user_dataframe + "com_df.csv", dtype=dtype)
       
        post_df = user_df.merge(pre_df, on='userId', how='inner')

        # Count users before and after
        n_pre = len(pre_df)
        n_post = len(post_df)

        # Split by class
        n_pre_control = sum(pre_df['isControl'])
        n_pre_coord = sum(~pre_df['isControl'])
        n_post_control = sum(post_df['isControl'])
        n_post_coord = sum(~post_df['isControl'])

        # Metrics
        overall_filtering = self._safe_ratio(n_pre - n_post, n_pre)
        control_filtering = self._safe_ratio(n_pre_control - n_post_control, n_pre_control)
        coord_filtering = self._safe_ratio(n_pre_coord - n_post_coord, n_pre_coord)
        coord_retention = self._safe_ratio(n_post_coord, n_pre_coord)
        precision_like = self._safe_ratio(n_post_coord, n_post)

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

        group_stats = self._build_label_count_dataframe(post_df, ["group"])

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
