from jinja2.utils import concat

from CharacterizationManager.Characterization.SingleLayerCommunitiesEvaluation.SingleLayerCommuntiesEvaluation import SingleLayerCommunitiesEvaluation
from CharacterizationManager.Characterization.CommunitiesEvaluation.CommunitiesEvaluation import CommunitiesEvaluation

from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.common_variables import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *
from utils.DomainCategoryManager import DomainCategoryManager
from utils.PlotManager.PlotManager import *
from utils.decorator_definition import *

import uunet.multinet as ml
import os
import matplotlib.pyplot as plt
import networkx as nx
import statistics
import numpy as np
import pandas as pd
import math
from typing import Any, Sequence
absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class MultiplexCommunitiesEvaluation:
    """
    Compute and save metrics, summaries, and characterizations for multiplex communities.
    """

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
            Create the evaluator for multiplex community outputs.
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
        self.data_path = f"{data_path}{self.dataset_name}{os.sep}"
        self.user_fraction = user_fraction
        self.type_filter = type_filter

        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter
        self.icm = icm
        self.dm = dm

        self.list_ca_str = '_'.join(list(self.dm.dict_path_ca.keys()))
        self.type_algorithm = type_algorithm
        self.cda = cda

        self.pm = PlotManager()
        self.ce = CommunitiesEvaluation(self.lm)

    @log_method
    def compute_multiplex_community_membership_summary(self) -> None:
        """
            Compute membership summaries for multiplex communities by layer and by community.
            :return: None. Per-layer and per-community summary CSV files are saved to the community analysis directory.
        """
        graph_files = [
            pos_csv
            for pos_csv in os.listdir(self.dm.path_user_dataframe)
            if pos_csv.endswith('.csv')
        ]

        net_filename = graph_files[0]

        com_df = self.ch.read_dataframe(
            self.dm.path_user_dataframe + net_filename,
            dtype=dtype
        )

        # -----------------------------
        # INFO PER LAYER
        # -----------------------------
        layer_df = com_df.groupby('layer').agg(
            numActorLayer=('actor', 'count'),
            numActors=('actor', 'nunique'),
            numCommunities=('cid', 'nunique')
        ).reset_index()

        # -----------------------------
        # INFO PER COMMUNITY
        # -----------------------------
        community_df = com_df.groupby('cid').agg(
            numActorLayer=('actor', 'count'),
            numActors=('actor', 'nunique'),
            numLayers=('layer', 'nunique')
        ).reset_index()

        # -----------------------------
        # ADD METADATA
        # -----------------------------
        community_df['algorithm'] = self.cda.get_algorithm_name()
        layer_df["algorithm"] = self.cda.get_algorithm_name()

        for key, value in self.cda.get_parameters().items():
            community_df[key] = value
            layer_df[key] = value

        # -----------------------------
        # SAVE
        # -----------------------------
        self.ch.save_dataframe(
            community_df,
            self.dm.path_community_analysis +
            f"{self.cda.get_algorithm_name()}_info_cda_per_community.csv",
        )

        self.ch.save_dataframe(
            layer_df,
            self.dm.path_community_analysis +
            f"{self.cda.get_algorithm_name()}_info_cda_per_layer.csv"
        )


    # ------------------------------------------------------------------------------------------------------------------
    def _compute_community_weight_stats(
        self,
        df: Any,
        networkx_graphs: dict[str, nx.Graph],
        weight_label: str,
    ) -> Any:
        """
        Compute mean, median, std, MAD, number of edges, nodes, and layers
        for each community in a multiplex network.
        Communities may span multiple layers.

        Parameters
        ----------
        df : pd.DataFrame
            Columns: ['actor', 'layer', 'cid']
            Each row assigns a node (actor) in a specific layer to a community.
        networkx_graphs : dict
            Dictionary mapping layer name -> nx.Graph (intra-layer graph).
        weight_label : str, default='weight'
            Edge attribute name for weights.

        Returns
        -------
        pd.DataFrame
            Columns:
            ['cid', 'mean_w', 'median_w', 'std_w', 'mad_w',
            'n_edges', 'n_nodes', 'n_layers']
        """

        # Map layer -> {node: cid}
        layer_to_comm = {
            layer: dict(zip(sub['actor'], sub['cid']))
            for layer, sub in df.groupby('layer')
        }

        # Initialize containers
        comm_weights = {}

        # Iterate over layers
        for layer, G in networkx_graphs.items():
            if layer not in layer_to_comm:
                continue  # skip layers not in df

            mapping = layer_to_comm[layer]

            for u, v, data in G.edges(data=True):
                if u not in mapping or v not in mapping:
                    continue
                cu, cv = mapping[u], mapping[v]
                if cu == cv:  # intra-community edge
                    try:
                        w = data.get(weight_label, 1.0) # default = 'w_'
                    except:
                        w = data.get('weight', 1.0)
                    comm_weights.setdefault(cu, []).append(w)

        # Compute edge-based stats
        records = []
        for cid, weights in comm_weights.items():
            weights = np.array(weights)
            stats = dict(
                cid=cid,
                avg_weight=np.mean(weights) if len(weights) > 0 else 0,
                median_weight=np.median(weights) if len(weights) > 0 else 0,
                std_weight=np.std(weights) if len(weights) > 0 else 0,
                mad_weight=np.median(np.abs(weights - np.median(weights))) if len(weights) > 0 else 0,
            )
            records.append(stats)

        result = pd.DataFrame(records).set_index('cid')

        # Add missing communities (with only inter-layer coupling)
        all_cids = df['cid'].unique()
        for cid in all_cids:
            if cid not in result.index:
                result.loc[cid] = dict(avg_weight=0, median_weight=0, std_weight=0, mad_weight=0)

        # Add size per community
        comm_summary = (
            df.groupby('cid')
            .agg(size=('actor', 'count'))
        )

        # Combine both
        result = result.join(comm_summary, how='left').fillna({'size': 0})
        
        result = result.reset_index()
        result = result.rename(columns={'cid': 'community'})
        result = result.sort_values('size', ascending=False)
        return result

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


    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------

    @log_method
    def compute_community_summary_statistics(self) -> None:
        """
            Compute descriptive statistics for multiplex communities.
            :return: None. Statistics are saved to the community analysis directory.
        """
        # (1) the number of communities generated,
        # (2) the average community size,
        # (3) the percentage of vertices included in at least one cluster (which is 1 for complete community detection methods),
        # (4) the percentage of actors included in at least one cluster (which is 1 for complete community detection methods),
        # (5) the ratio between the number of actor-layer pairs and the number of distinct actor-layer pairs,
        # indicating the level of overlapping (which is 1 for partitioning community detection methods and higher for overlapping methods).
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]

        MG = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)

        com_df_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith('.csv')]
        com_files = [pos_csv for pos_csv in os.listdir(self.dm.path_coms) if pos_csv.endswith('.p')]
        com_df_filename = com_df_files[0]
        com_filename = com_files[0]
        comm_df = self.ch.read_dataframe(self.dm.path_user_dataframe + com_df_filename, dtype=dtype)
        comm = self.ch.load_object(self.dm.path_coms + com_filename)

        stats = {}
        stats["algorithm"] = self.cda.get_algorithm_name()
        for key, value in self.cda.get_parameters().items():
            stats[key] = value
        stats["nCommunities"] = comm_df['cid'].nunique()
        stats["avgActorPerCom"] = comm_df.groupby("cid").nunique()['actor'].mean()
        stats["avgLayerPerCom"] = comm_df.groupby("cid").nunique()['layer'].mean()
        stats["percClusteredVertices"] = comm_df[["actor", "layer"]].drop_duplicates().shape[0] / ml.num_vertices(MG)
        stats["overlapping"] = comm_df.shape[0] / comm_df[["actor", "layer"]].drop_duplicates().shape[0]
        stats["modularity"] = ml.modularity(MG, comm)
        stats_df = pd.DataFrame([stats])

        self.ch.update_dataframe(stats_df, self.dm.path_community_analysis + f"{self.cda.get_algorithm_name()}_statistics_communities.csv", dtype=dtype)

    @log_method
    def compute_metrics_node_communities(
        self,
        metrics: Sequence[str] | None,
        th_size: int | None,
        restrict_neighbors: bool,
        merge_existing: bool
    ) -> None:
        """
            Compute node metrics inside multiplex communities for each layer graph.
            :param metrics: Node metric names to compute, or None for defaults.
            :param th_size: Minimum community size to include.
            :param restrict_neighbors: Whether neighbor-based metrics should only count selected-community neighbors.
            :param merge_existing: Whether to merge computed columns with an existing metrics file.
            :return: None. Node-community metrics are saved to the community analysis directory.
        """
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]

        MG = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)

        com_df_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith('.csv')]
        com_df_filename = com_df_files[0]
        comm_df = self.ch.read_dataframe(self.dm.path_user_dataframe + com_df_filename, dtype=dtype)

        # Extract the dictionary 'layer': NetworkXGraph from the multiplex graph
        networkx_graphs = ml.to_nx_dict(MG)

        if merge_existing:
            temp_files = []

        # Iterate over each graph in the list
        for layer, G in networkx_graphs.items():
            self.lm.printl(f"{file_name}. compute_metrics_node_communities processing layer: {layer}.")
            type_ca = action_map_inverse[layer]
            layer_df = comm_df[comm_df['layer'] == layer]
            layer_df.drop(columns=['layer'], inplace=True)
            # Add the 'group' attribute to nodes based on the 'cid' column in the dataframe
            for _, row in layer_df.iterrows():
                actor = row['actor']
                cid = row['cid']
                if actor in G:
                    G.nodes[actor]['group'] = cid

            node_metrics_df = self.ce.compute_node_metrics_df(G, metrics, th_size, restrict_neighbors)
            node_metrics_df['layer'] = type_ca

            # in case i computed other metrics, and i want to compute a new one, i update the current dataframe
            # with the new metrics. So i need to merge the new metrics with the existing ones.
            # I save the new dataframe in a temporary file so that i am sure not to lose the computed info for each layer
            if merge_existing:
                temp_files.append(self.dm.path_community_analysis + f"temp_{layer}_node_metrics_communities.csv")
                self.ch.save_dataframe(node_metrics_df, self.dm.path_community_analysis + f"temp_{layer}_node_metrics_communities.csv")
            else:
                self.ch.update_dataframe(node_metrics_df, self.dm.path_community_analysis + f"{self.cda.get_algorithm_name()}_th_size_{str(th_size)}_node_metrics_communities.csv", dtype=dtype)

        # now I read again the temporary files, and I merge them with the existing dataframe
        # concat_df has the new metrics computed for each layer, so the same number of rows of the original dataframe.
        # with respect to the single layer case, here I have to join on the node, community and layer columns!
        if merge_existing:
            update_df_list = []
            for layer, G in networkx_graphs.items():
                node_metrics_df = self.ch.read_dataframe(self.dm.path_community_analysis + f"temp_{layer}_node_metrics_communities.csv", dtype=dtype)
                update_df_list.append(node_metrics_df)
            concat_df = pd.concat(update_df_list)

            self.ch.update_columns_dataframe(concat_df,
                                             self.dm.path_community_analysis + f"{self.cda.get_algorithm_name()}_th_size_{str(th_size)}_node_metrics_communities.csv",
                                             join_columns=['node', 'community', 'layer'], dtype=dtype)
            # remove the temporary files
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)
    
    @log_method
    def validate_communities(self) -> None:
        """
            Validate multiplex communities against available isControl labels in source co-action data.
            :return: None. Validation dataframes and plots are saved to the community analysis directory.
        """
        com_df_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith('.csv')]
        com_df_filename = com_df_files[0]
        comm_df = self.ch.read_dataframe(self.dm.path_user_dataframe + com_df_filename, dtype=dtype)
        
        df_list = []
        for ca in self.list_ca:
            ca_type = ca.get_co_action()
            df = self.ch.read_dataframe(self._filtered_co_action_file_path(action_map[ca_type]), dtype)
            if not self._has_is_control_label(df, ca_type):
                return
            df_list.append(df)
        data_df = pd.concat(df_list)
        data_df = self._normalize_is_control_column(data_df, self.cda.get_algorithm_name())
        pre_df = data_df[['userId', 'isControl']]
        pre_df = pre_df.drop_duplicates()
        
        post_df = comm_df.merge(
            pre_df,
            left_on='actor',
            right_on='userId',
            how='inner'
        )

        post_df = post_df.drop(columns='actor')
        post_df = post_df.rename(columns={'cid': 'group'})

        groupby_lists = [['group'], ['group', 'layer']]
        

        for groupby_list in groupby_lists:
            groupby_str = '_'.join(groupby_list + ['isControl'])
            self.lm.printl(f"{file_name}. validate_communities processing groupby: {groupby_str}.")
            group_stats = self._build_label_count_dataframe(post_df, groupby_list)

            self.ch.save_dataframe(group_stats, self.dm.path_community_analysis + f"{self.cda.get_algorithm_name()}_{groupby_str}_validation_communities.csv")

            self.pm.plot_histogram(self.dm.path_community_analysis, self.cda.get_algorithm_name(), group_stats['purity'], 'Purity', 'Number of Groups',
                               'Distribution of Group Purity', f"{self.cda.get_algorithm_name()}_{groupby_str}_purity_distribution.png")

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

    @log_method
    def compute_community_edge_weight_statistics(
        self,
        community_size_th: int | None,
        community_label: str,
        weight_label: str
    ) -> None:
        """
            Compute edge-weight coordination statistics for multiplex communities.
            :param community_size_th: Minimum community size to include, or None to include all communities.
            :param community_label: Community label column or attribute name.
            :param weight_label: Edge attribute containing edge weights.
            :return: None. Coordination statistics are saved to the community analysis directory.
        """
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]

        MG = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)

        com_df_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith('.csv')]
        com_df_filename = com_df_files[0]
        comm_df = self.ch.read_dataframe(self.dm.path_user_dataframe + com_df_filename, dtype=dtype)

        networkx_graphs = ml.to_nx_dict(MG)
        stats_df = self._compute_community_weight_stats(comm_df, networkx_graphs, weight_label)
        if community_size_th is not None:
            th_str = f"_th_size_{str(community_size_th)}"
            stats_df = stats_df[stats_df['size'] >= community_size_th]
        else:
            th_str = ""
        self.ch.save_dataframe(stats_df, self.dm.path_community_analysis + f"{self.cda.get_algorithm_name()}{th_str}_coordination_communities.csv")

    def _top_community_dataframe(self, top_n: int) -> pd.DataFrame:
        """
        Return the top multiplex communities enriched with coordination statistics.

        :param top_n: [int] Number of communities with the largest actor-layer count to keep.
        :return: [pd.DataFrame] Top community dataframe.
        """
        coord_com = self.ch.read_dataframe(
            f"{self.dm.path_community_analysis}{self.cda.get_algorithm_name()}_coordination_communities.csv",
            dtype=dtype,
        )
        info_per_com = self.ch.read_dataframe(
            f"{self.dm.path_community_analysis}{self.cda.get_algorithm_name()}_info_cda_per_community.csv",
            dtype=dtype,
        )
        info_per_com = info_per_com.drop_duplicates(subset=["cid"], keep="last").copy()
        coord_com = coord_com.drop_duplicates(subset=["community"], keep="last").copy()
        info_per_com["cid"] = info_per_com["cid"].astype(str)
        coord_com["community"] = coord_com["community"].astype(str)

        merged_df = info_per_com.merge(coord_com, left_on="cid", right_on="community")
        columns = ["cid", "numActorLayer", "numActors", "numLayers", "avg_weight", "median_weight", "std_weight"]
        for column in columns:
            if column != "cid":
                merged_df[column] = pd.to_numeric(merged_df[column], errors="coerce")
        return merged_df[columns].sort_values("numActorLayer", ascending=False).head(top_n)

    @log_method
    def save_top_communities_summary(self, top_n: int = 10) -> pd.DataFrame:
        """
        Save structural and edge-weight information for the top multiplex communities.

        :param top_n: [int] Number of communities with the largest actor-layer count to save.
        :return: [pd.DataFrame] Saved top-community summary dataframe.
        """
        top_communities_df = self._top_community_dataframe(top_n).round(3)
        self.ch.save_dataframe(
            top_communities_df,
            f"{self.dm.path_community_analysis}{self.cda.get_algorithm_name()}_top_{top_n}_communities_summary.csv",
        )
        return top_communities_df

    def _read_url_layer_dataframe(self, filename_stem: str, source: str) -> pd.DataFrame:
        """
        Read one URL co-action file and normalize columns needed for community URL analysis.

        :param filename_stem: [str] Co-action filename stem, for example postURL.
        :param source: [str] Output source label.
        :return: [pd.DataFrame] URL dataframe with userId, domain, and source columns.
        """
        url_df = self.ch.read_dataframe(self._filtered_co_action_file_path(filename_stem), dtype=dtype)
        url_df = url_df[["userId", "objectId"]].copy()
        url_df["userId"] = url_df["userId"].astype(str)
        url_df = url_df.rename(columns={"objectId": "domain"})
        url_df["source"] = source
        return url_df

    def _assign_url_rows_to_top_communities(
        self,
        com_df: pd.DataFrame,
        top_cids: Sequence[Any],
        url_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assign URL rows to the top communities containing the URL-sharing actors.

        :param com_df: [pd.DataFrame] Multiplex community membership dataframe.
        :param top_cids: [Sequence[Any]] Community ids to keep.
        :param url_df: [pd.DataFrame] URL dataframe with userId, domain, and source columns.
        :return: [pd.DataFrame] URL rows enriched with cid.
        """
        top_cids = [str(cid) for cid in top_cids]
        com_df = com_df.copy()
        com_df["cid"] = com_df["cid"].astype(str)
        com_df["actor"] = com_df["actor"].astype(str)
        actor_community = (
            com_df[com_df["cid"].isin(top_cids)][["actor", "cid"]]
            .drop_duplicates()
            .rename(columns={"actor": "userId"})
        )
        actor_community["userId"] = actor_community["userId"].astype(str)
        url_df = url_df.copy()
        url_df["userId"] = url_df["userId"].astype(str)
        merged_df = url_df.merge(actor_community, on="userId", how="inner")
        self.lm.printl(
            f"{file_name}. URL-community assignment: top_communities={len(top_cids)}, "
            f"top_actors={actor_community['userId'].nunique()}, url_rows={len(url_df)}, matched_url_rows={len(merged_df)}"
        )
        return merged_df

    @log_method
    def charactrize_url_layers_communities(
        self,
        top_n: int = 10,
        category_file: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze URL domain frequencies and domain-category percentages for top multiplex communities.

        The method saves only post/comment-separated domain frequencies and post/comment-separated
        category percentages for the selected top communities.

        :param top_n: [int] Number of communities with the largest actor-layer count to analyze.
        :param category_file: [str | None] Optional CSV file with domain,category columns.
        :return: [tuple[pd.DataFrame, pd.DataFrame]] Domain-frequency and category-percentage dataframes.
        """
        com_df_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith(".csv")]
        com_df_filename = com_df_files[0]
        com_df = self.ch.read_dataframe(self.dm.path_user_dataframe + com_df_filename, dtype=dtype)

        top_communities_df = self._top_community_dataframe(top_n)
        top_cids = top_communities_df["cid"].tolist()

        post_url = self._read_url_layer_dataframe("postURL", "postURL")
        comment_url = self._read_url_layer_dataframe("commentURL", "commentURL")
        url_df = pd.concat([post_url, comment_url], ignore_index=True)
        url_df = self._assign_url_rows_to_top_communities(com_df, top_cids, url_df)

        category_manager = DomainCategoryManager(category_file)
        url_df["domain_clean"] = url_df["domain"].apply(category_manager.clean_domain)
        url_df["domain_category"] = url_df["domain_clean"].apply(category_manager.categorize_domain)

        domain_frequency_df = (
            url_df
            .groupby(["cid", "source", "domain_clean", "domain_category"], as_index=False, dropna=False)
            .agg(
                count=("domain_clean", "size"),
                num_users=("userId", "nunique"),
            )
        )
        source_totals = domain_frequency_df.groupby(["cid", "source"])["count"].transform("sum")
        domain_frequency_df["share"] = domain_frequency_df["count"] / source_totals
        domain_frequency_df["rank"] = (
            domain_frequency_df
            .sort_values(["cid", "source", "count"], ascending=[True, True, False])
            .groupby(["cid", "source"])
            .cumcount()
            + 1
        )
        domain_frequency_df = domain_frequency_df.sort_values(["cid", "source", "rank"])

        category_percentage_df = (
            url_df
            .groupby(["cid", "source", "domain_category"], as_index=False, dropna=False)
            .agg(
                count=("domain_category", "size"),
                num_domains=("domain_clean", "nunique"),
                num_users=("userId", "nunique"),
            )
        )
        category_totals = category_percentage_df.groupby(["cid", "source"])["count"].transform("sum")
        category_percentage_df["percentage"] = category_percentage_df["count"] / category_totals
        category_percentage_df = category_percentage_df.sort_values(
            ["cid", "source", "percentage"],
            ascending=[True, True, False],
        )

        expected_category_rows = pd.DataFrame(
            [
                {
                    "cid": str(cid),
                    "source": source,
                    "domain_category": "no_url",
                    "count": 0,
                    "num_domains": 0,
                    "num_users": 0,
                    "percentage": 0.0,
                }
                for cid in top_cids
                for source in ["postURL", "commentURL"]
            ]
        )
        existing_category_pairs = set(
            zip(
                category_percentage_df["cid"].astype(str),
                category_percentage_df["source"].astype(str),
            )
        )
        missing_category_rows = expected_category_rows[
            ~expected_category_rows.apply(
                lambda row: (str(row["cid"]), str(row["source"])) in existing_category_pairs,
                axis=1,
            )
        ]
        category_percentage_df["cid"] = category_percentage_df["cid"].astype(str)
        category_percentage_df = pd.concat(
            [category_percentage_df, missing_category_rows],
            ignore_index=True,
        ).sort_values(["cid", "source", "percentage"], ascending=[True, True, False])

        output_prefix = f"{self.cda.get_algorithm_name()}_top_{top_n}_communities_url"
        self.ch.save_dataframe(
            domain_frequency_df,
            f"{self.dm.path_community_analysis}{output_prefix}_domain_frequency.csv",
        )
        self.ch.save_dataframe(
            category_percentage_df,
            f"{self.dm.path_community_analysis}{output_prefix}_domain_category_percentage.csv",
        )

        return domain_frequency_df, category_percentage_df

    # def compute_ML_intra_inter_edge(self):
    #     graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith('.csv')]
    #     net_filename = graph_files[0]
    #     comm_df = self.ch.read_dataframe(self.dm.path_user_dataframe + net_filename, dtype=dtype)
    #     multi_edge_list_df = self.ch.read_dataframe(self.dm.path_multi_edge_list_df + "edge_list_df.csv", dtype=dtype)
    
    #     # mult_edge_list_df:
    #     # userId1, userId2, weight, layer
    #     # comm_df:
    #     # actor, layer, cid
    
    #     # double join, to assign the community id to both users, userId1 and userId2
    #     result_df = multi_edge_list_df.merge(comm_df, left_on=["userId1", "layer"], right_on=['actor', 'layer'])
    #     result_df = result_df.drop(columns=["actor"])
    #     result_df = result_df.rename(columns={"cid": "community1"})
    #     result_df = result_df.merge(comm_df, left_on=["userId2", "layer"], right_on=['actor', 'layer'])
    #     result_df = result_df.rename(columns={"cid": "community2"})
    #     result_df = result_df.drop(columns=["actor"])
    #     # result_df:
    #     # userId1, userId2, community1, community2
    
    #     result_df["intra_edge"] = result_df["community1"] == result_df["community2"]
    #     result_df["community"] = np.nan
    #     result_df.loc[result_df["intra_edge"] == True, "community"] = result_df["community1"]
    
    #     self.ch.save_dataframe(result_df, self.dm.path_community_analysis + "edge_communities.csv")
    #
    #
    #
    # def compute_ML_intra_inter_edge_community_statistics(self):
    #     df = pd.read_csv(self.dm.path_analysis + "edge_communities.csv", dtype=dtype)
    #     intra_edge = df[df["intra_edge"] == True]
    #     # Group by 'community' and compute statistics
    #     comm_stats_df = intra_edge.groupby(['cid'])['weight'].agg(
    #         ['count', 'mean', 'std', 'median', 'max', 'min'])
    #     comm_stats_df = comm_stats_df.reset_index()
    #     comm_stats_df['cid'] = comm_stats_df['cid'].astype('int')
    #
    #     comm_stats_df = comm_stats_df.rename(
    #         columns={'count': 'nIntraEdges', 'mean': 'meanWeight', 'median': 'medianWeight', 'std': 'stdDevWeight',
    #                  'max': 'maxWeight', 'min': 'minWeight'})
    #
    #     inter_edge = df[df["intra_edge"] == False]
    #     com1_df = inter_edge['community1'].value_counts().reset_index()
    #     com2_df = inter_edge['community2'].value_counts().reset_index()
    #     com_inter_edge_df = pd.merge(com1_df, com2_df, left_on='community1', right_on='community2', how='outer')
    #
    #     # a community could be present only in column community1 or community2. if this is the case, i fill the nan values, derived from the outer
    #     # join for both columns. I fill with 0, nan values in the counts (if community for a column misses, its count will be null)
    #     com_inter_edge_df['community1'] = com_inter_edge_df['community1'].fillna(com_inter_edge_df['community2'])
    #     com_inter_edge_df['community2'] = com_inter_edge_df['community2'].fillna(com_inter_edge_df['community1'])
    #     com_inter_edge_df['count_x'] = com_inter_edge_df['count_x'].fillna(0)
    #     com_inter_edge_df['count_y'] = com_inter_edge_df['count_y'].fillna(0)
    #     com_inter_edge_df['nInterEdges'] = com_inter_edge_df['count_x'] + com_inter_edge_df['count_y']
    #     com_inter_edge_df = com_inter_edge_df.drop(columns=['community2', 'count_x', 'count_y'])
    #     com_inter_edge_df = com_inter_edge_df.rename(columns={'community1': 'community'})
    #
    #     # merge info for intraedge and interedge
    #     # in this way I am missing some info regarding the case a community span across multiple layers, since, here
    #     # I am just analyzing edges, which by definition are intra layer. Therefore, it can happen that a community does
    #     # not have any intra edges, but inter_edges. In this case the link are among the same nodes on different layer,
    #     # whose edges are not explicitly present in edge_list.
    #     final_df = pd.merge(comm_stats_df, com_inter_edge_df, on='cid', how='outer')
    #     final_df[['nIntraEdges', 'nInterEdges']] = final_df[['nIntraEdges', 'nInterEdges']].fillna(0)
    #
    #     self.ch.save_dataframe(final_df,self.dm.path_community_analysis + "intra_inter_weight_communities.csv")
