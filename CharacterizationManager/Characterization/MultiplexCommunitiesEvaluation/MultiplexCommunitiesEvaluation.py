from jinja2.utils import concat

from CharacterizationManager.Characterization.SingleLayerCommunitiesEvaluation.SingleLayerCommuntiesEvaluation import SingleLayerCommunitiesEvaluation
from CharacterizationManager.Characterization.CommunitiesEvaluation.CommunitiesEvaluation import CommunitiesEvaluation

from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.common_variables import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *
from utils.PlotManager.PlotManager import *

import uunet.multinet as ml
import os
import matplotlib.pyplot as plt
import networkx as nx
import statistics
import numpy as np
import math
absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class MultiplexCommunitiesEvaluation:
    def __init__(self, dataset_name, user_fraction, type_filter, list_ca, dict_ca_filter, icm, dm, type_algorithm, cda):
        self.lm = LogManager('main')
        self.ch = Checkpoint()
        self.cm = ConversionManager()

        self.dataset_name = dataset_name
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

    def compute_info_communities(self):
        self.lm.printl(f"{file_name}. compute_info_communities start.")
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith('.csv')]
        net_filename = graph_files[0]
        com_df = self.ch.read_dataframe(self.dm.path_user_dataframe + net_filename, dtype=dtype)

        layer_df = com_df.groupby('layer').agg(
            numActors=('actor', 'nunique'),
            numCommunities=('cid', 'nunique')
        ).reset_index()

        community_df = com_df.groupby('cid').agg(
            numActors=('actor', 'nunique'),
            numLayers=('layer', 'nunique')
        ).reset_index()


        community_df['algorithm'] = self.cda.get_algorithm_name()
        layer_df["algorithm"] = self.cda.get_algorithm_name()
        for key, value in self.cda.get_parameters().items():
            community_df[key] = value
            layer_df[key] = value

        self.ch.update_dataframe(community_df, self.dm.path_community_analysis + f"{self.cda.get_algorithm_name()}_info_cda_per_community.csv", dtype)
        self.ch.update_dataframe(layer_df, self.dm.path_community_analysis + f"{self.cda.get_algorithm_name()}_info_cda_per_layer.csv", dtype=dtype)

        self.lm.printl(f"{file_name}. compute_info_communities completed.")

    # ------------------------------------------------------------------------------------------------------------------
    def __compute_community_weight_stats(self, df, networkx_graphs, weight_label):
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


    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------

    def compute_statistics_communities(self):
        # (1) the number of communities generated,
        # (2) the average community size,
        # (3) the percentage of vertices included in at least one cluster (which is 1 for complete community detection methods),
        # (4) the percentage of actors included in at least one cluster (which is 1 for complete community detection methods),
        # (5) the ratio between the number of actor-layer pairs and the number of distinct actor-layer pairs,
        # indicating the level of overlapping (which is 1 for partitioning community detection methods and higher for overlapping methods).
        self.lm.printl(f"{file_name}. compute_statistics_communities start.")

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

        self.lm.printl(f"{file_name}. compute_statistics_communities completed.")

    def compute_metrics_node_communities(self, metrics, th_size, restrict_neighbors, merge_existing):
        self.lm.printl(f"{file_name}. compute_metrics_node_communities start.")

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
            self.lm.printl(f"{file_name}. compute_metrics_node_communities layer: {layer} start.")
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

        self.lm.printl(f"{file_name}. compute_metrics_node_communities completed.")
    

    def validate_communities(self):
        self.lm.printl(f"{file_name}. validate_communities start.")
        com_df_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith('.csv')]
        com_df_filename = com_df_files[0]
        comm_df = self.ch.read_dataframe(self.dm.path_user_dataframe + com_df_filename, dtype=dtype)
        
        df_list = []
        for ca in self.list_ca:
            ca_type = ca.get_co_action()
            df = self.ch.read_dataframe(f"{self.dm.path_dataset}{self.user_fraction}_{self.type_filter}_{self.dataset_name}_{ca_type}.csv", dtype)
            df_list.append(df)
        data_df = pd.concat(df_list)
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

        groupby_lists = [['group', 'isControl'], ['group', 'layer', 'isControl']]
        

        for groupby_list in groupby_lists:
            groupby_str = '_'.join(groupby_list)
            self.lm.printl(f"{file_name}. validate_communities groupby: {groupby_str} start.")
            # --- Count True/False per group
            group_counts = (
                post_df.groupby(groupby_list)
                .size()
                .unstack(fill_value=0)
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
            group_stats['purity'] = group_stats[['nControl', 'nCoord']].max(axis=1) / group_stats['nTotal']

            self.ch.save_dataframe(group_stats, self.dm.path_community_analysis + f"{self.cda.get_algorithm_name()}_{groupby_str}_validation_communities.csv")

            self.pm.plot_histogram(self.dm.path_community_analysis, self.cda.get_algorithm_name(), group_stats['purity'], 'Purity', 'Number of Groups',
                               'Distribution of Group Purity', f"{self.cda.get_algorithm_name()}_{groupby_str}_purity_distribution.png")

    def compute_coordination_communities(self, community_size_th, community_label, weight_label):
        self.lm.printl(f"{file_name}. compute_coordination_communities start.")
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]

        MG = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)

        com_df_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith('.csv')]
        com_df_filename = com_df_files[0]
        comm_df = self.ch.read_dataframe(self.dm.path_user_dataframe + com_df_filename, dtype=dtype)

        networkx_graphs = ml.to_nx_dict(MG)
        stats_df = self.__compute_community_weight_stats(comm_df, networkx_graphs, weight_label)
        if community_size_th is not None:
            th_str = f"_th_size_{str(community_size_th)}"
            stats_df = stats_df[stats_df['size'] >= community_size_th]
        else:
            th_str = ""
        self.ch.save_dataframe(stats_df, self.dm.path_community_analysis + f"{self.cda.get_algorithm_name()}{th_str}_coordination_communities.csv")
        self.lm.printl(f"{file_name}. compute_coordination_communities completed.")

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