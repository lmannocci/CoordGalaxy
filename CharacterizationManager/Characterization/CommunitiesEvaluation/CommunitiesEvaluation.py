import networkx as nx
import numpy as np
import pandas as pd
import time
from utils.common_variables import *

class CommunitiesEvaluation:
    def __init__(self, lm):
        self.lm = lm

    def compute_node_metrics_df(self, graph, metrics, th_size, restrict_neighbors):
        """
        Compute various node-level metrics, including Guimerà and Amaral roles, and additional centrality measures.

        Args:
            graph (nx.Graph): NetworkX graph where each node has a 'group' attribute indicating its community.
            metrics (list, optional): List of metrics to compute. If None, all metrics are computed.
                                      Supported metrics: 'z_score', 'p_coeff', 'bridging_centrality', 'role',
                                                         'degree_centrality', 'betweenness_centrality',
                                                         'closeness_centrality', 'eigenvector_centrality',
                                                         'local_clustering_coefficient',
                                                         'bridging_eigenvector_centrality'.
            th_size (int or None, optional): Minimum community size to consider. If None, all communities are included.
            restrict_neighbors (bool, optional): If True, considers only neighbors in selected communities for metrics.
                                                 If False, considers all neighbors in the graph.

        Returns:
            pd.DataFrame: DataFrame with selected metrics for each node in communities meeting the size threshold.
        """

        if metrics is None:
            metrics = available_node_metrics
        else:
            metrics = [m for m in metrics if m in available_node_metrics]

        # Map nodes to their communities
        node_to_comm = nx.get_node_attributes(graph, "group")
        communities = {comm: [] for comm in set(node_to_comm.values())}
        for node, comm in node_to_comm.items():
            communities[comm].append(node)

        # Filter communities by size if th_size is set
        if th_size is not None:
            communities = {comm: nodes for comm, nodes in communities.items() if len(nodes) >= th_size}
        valid_nodes = {node for nodes in communities.values() for node in nodes} if th_size else set(graph.nodes())

        self.lm.printl(f"Number of communities, th_size={th_size}: {len(communities) if th_size else 'All'}")
        self.lm.printl(f"Number of nodes: {len(valid_nodes)}")

        # Precompute centrality metrics for the whole graph if needed
        start = time.time()
        degree_centrality = (
            nx.degree_centrality(graph) if "degree_centrality" in metrics else {}
        )
        if "degree_centrality" in metrics:
            self.lm.printl(f"Degree centrality computed in {time.time() - start:.2f} seconds")

        start = time.time()
        betweenness_centrality = (
            nx.betweenness_centrality(graph, normalized=True)
            if "betweenness_centrality" in metrics
            else {}
        )
        if "betweenness_centrality" in metrics:
            self.lm.printl(f"Betweenness centrality computed in {time.time() - start:.2f} seconds")

        start = time.time()
        closeness_centrality = (nx.closeness_centrality(graph) if "closeness_centrality" in metrics else {})
        if "closeness_centrality" in metrics:
            self.lm.printl(f"Closeness centrality computed in {time.time() - start:.2f} seconds")

        start = time.time()
        eigenvector_centrality = (
            nx.eigenvector_centrality(graph, max_iter=1000)
            if "eigenvector_centrality" in metrics or "bridging_eigenvector_centrality" in metrics
            else {}
        )
        if "eigenvector_centrality" in metrics or "bridging_eigenvector_centrality" in metrics:
            self.lm.printl(f"Eigenvector centrality computed in {time.time() - start:.2f} seconds")

        start = time.time()
        local_clustering_coefficient = (nx.clustering(graph) if "local_clustering_coefficient" in metrics else {})
        if "local_clustering_coefficient" in metrics:
            self.lm.printl(f"Local clustering coefficient computed in {time.time() - start:.2f} seconds")

        start = time.time()
        page_rank = (nx.pagerank(graph) if "page_rank" in metrics else {})
        if "page_rank" in metrics:
            self.lm.printl(f"PageRank computed in {time.time() - start:.2f} seconds")

        # Metrics dictionary
        data = []
        for i, node in enumerate(valid_nodes):
            self.lm.printK(i, 1000, f"Processing node {i}/{len(valid_nodes)}")

            # Get node's community and subgraph
            comm = node_to_comm[node]
            comm_nodes = communities[comm] if th_size else list(graph.nodes())
            comm_subgraph = graph.subgraph(comm_nodes)

            # Initialize metrics for the node
            node_data = {"node": node, "community": comm}

            # Compute z_score
            if "z_score" in metrics:
                k_i = graph.degree(node)
                comm_degrees = [comm_subgraph.degree(n) for n in comm_nodes]
                mu_k = np.mean(comm_degrees)
                sigma_k = np.std(comm_degrees)
                z_score = (k_i - mu_k) / sigma_k if sigma_k != 0 else 0
                node_data["z_score"] = z_score

            # Compute participation coefficient
            if "p_coeff" in metrics:
                total_degree = graph.degree(node)
                comm_degrees = {c: 0 for c in communities.keys()}
                for neighbor in graph.neighbors(node):
                    neighbor_comm = node_to_comm.get(neighbor)
                    if not restrict_neighbors or neighbor_comm in communities:
                        comm_degrees[neighbor_comm] = comm_degrees.get(neighbor_comm, 0) + 1
                p_coeff = 1 - sum(
                    (comm_degrees[c] / total_degree) ** 2
                    for c in comm_degrees
                    if total_degree > 0
                )
                node_data["p_coeff"] = p_coeff

            # Compute bridging centrality (standard version)
            if "bridging_centrality" in metrics:
                b_centrality = betweenness_centrality.get(node, 0)
                clustering = local_clustering_coefficient.get(node, 0)
                bridging_centrality = b_centrality * (1 - clustering)
                node_data["bridging_centrality"] = bridging_centrality

            # Compute bridging eigenvector centrality
            if "bridging_eigenvector_centrality" in metrics:
                e_centrality = eigenvector_centrality.get(node, 0)
                clustering = local_clustering_coefficient.get(node, 0)
                bridging_eigenvector_centrality = e_centrality * (1 - clustering)
                node_data["bridging_eigenvector_centrality"] = bridging_eigenvector_centrality

            # Compute Guimerà and Amaral role classification
            if "role" in metrics and "z_score" in metrics and "p_coeff" in metrics:
                if z_score < 2.5:
                    if p_coeff < 0.05:
                        role = "R1: Ultra-peripheral"
                    else:
                        role = "R2: Peripheral"
                else:
                    if p_coeff < 0.3:
                        role = "R3: Provincial hub"
                    elif p_coeff < 0.75:
                        role = "R4: Connector hub"
                    else:
                        role = "R5: Kinless node"
                node_data["role"] = role

            # Add precomputed centrality metrics
            if "degree_centrality" in metrics:
                node_data["degree_centrality"] = degree_centrality.get(node, 0)
            if "betweenness_centrality" in metrics:
                node_data["betweenness_centrality"] = betweenness_centrality.get(node, 0)
            if "closeness_centrality" in metrics:
                node_data["closeness_centrality"] = closeness_centrality.get(node, 0)
            if "eigenvector_centrality" in metrics:
                node_data["eigenvector_centrality"] = eigenvector_centrality.get(node, 0)
            if "local_clustering_coefficient" in metrics:
                node_data["local_clustering_coefficient"] = local_clustering_coefficient.get(node, 0)
            if "page_rank" in metrics:
                node_data["page_rank"] = page_rank.get(node, 0)

            data.append(node_data)

        # Create DataFrame
        df = pd.DataFrame(data)
        return df
