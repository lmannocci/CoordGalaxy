import os
import time
from typing import Any

import networkx as nx
import uunet.multinet as ml
from cdlib import algorithms
from cdlib import NodeClustering
from infomap import Infomap

from IntegrityConstraintManager.IntegrityConstraintManager import IntegrityConstraintManager
from utils.ConversionManager.ConversionManager import ConversionManager
from utils.LogManager.LogManager import LogManager
from utils.common_variables import W_VAR, algorithm_map, parameters_map

file_name = os.path.splitext(os.path.basename(__file__))[0]


class CDAlgorithm:
    def __init__(self, algorithm_name: str, parameters: dict[str, Any] | None = None) -> None:
        """
            Create a community-detection algorithm configuration.
            :param algorithm_name: [str] Algorithm name, for example louvain, infomap, glouvain, or ginfomap.
            :param parameters: [dict[str, Any] | None] Algorithm parameters, or None when the algorithm has no
                parameters.
            :return: None.
        """
        self.lm = LogManager("main")
        self.icm = IntegrityConstraintManager(file_name)
        self.icm.check_CDAlgorithm(algorithm_name, parameters)

        self.algorithm_name = algorithm_name
        self.parameters = parameters

        self.cm = ConversionManager()


    def get_algorithm_name(self) -> str:
        """
            Return the configured community-detection algorithm name.
            :return: [str] Algorithm name.
        """
        return self.algorithm_name

    def get_parameters(self) -> dict[str, Any] | None:
        """
            Return the configured algorithm parameters.
            :return: [dict[str, Any] | None] Algorithm parameters, or None.
        """
        return self.parameters

    def __str__(self) -> str:
        """
            Return a human-readable representation of the algorithm configuration.
            :return: [str] Algorithm name with parameters formatted for display.
        """
        if self.parameters is None or len(self.parameters) == 0:
            return self.algorithm_name
        param_str = '_'.join(f'{k}_{v}' for k, v in self.parameters.items())
        alg_str = self.algorithm_name + '(' + param_str + ')'
        return alg_str

    def __repr__(self) -> str:
        """
            Return the path-safe representation of the algorithm configuration.
            :return: [str] Algorithm name with parameters formatted for output paths.
        """
        if self.parameters is None or len(self.parameters) == 0:
            return self.algorithm_name
        param_str = '_'.join(f'{k}_{v}' for k, v in self.parameters.items())
        alg_str = self.algorithm_name + '_' + param_str
        return alg_str

    def cda_repr_abbr(self) -> str:
        """
            Return an abbreviated algorithm representation for compact output paths.
            :return: [str] Abbreviated algorithm string.
        """
        cda_str = repr(self)
        for key, value in algorithm_map.items():
            cda_str = cda_str.replace(key, value)
        for key, value in parameters_map.items():
            cda_str = cda_str.replace(key, value)
        return cda_str


    def _custom_flattening(self, MG: Any, type_flattening: str = "sum") -> nx.Graph:
        """
            Flatten a multiplex network with a custom edge-weight aggregation rule.
            :param MG: [Any] uunet multiplex network.
            :param type_flattening: [str] Flattening strategy: sum, average, or and_sum.
            :return: [nx.Graph] Flattened NetworkX graph.
        """
        self.lm.printl(f"{file_name}. Flattening multiplex network with method: {type_flattening}.")
        attribute = 'w_'
        # Create an empty graph to hold the union of all graphs
        flattened_graph = nx.Graph()
        # Dictionary to keep track of the total weights and occurrences for averaging
        edge_weights = {}

        # Extract the dictionary 'layer': NetworkXGraph from the multiplex graph
        networkx_graphs = ml.to_nx_dict(MG)

        # Identify edges present in all layers for the 'and_sum' option
        if type_flattening == 'and_sum':
            common_edges = set.intersection(*[set(G.edges()) for G in networkx_graphs.values()])

        # Iterate over each graph in the list
        for layer, G in networkx_graphs.items():
            # Add nodes to the flattened graph
            for node in G.nodes(data=True):
                flattened_graph.add_node(node[0], **node[1])

            # Add edges and handle weights based on type_flattening
            for u, v, data in G.edges(data=True):
                weight = data.get(attribute, 1)

                if type_flattening == "sum":
                    if flattened_graph.has_edge(u, v):
                        # Sum the weights if the edge already exists
                        flattened_graph[u][v][attribute] += weight
                    else:
                        # Add the edge with its weight
                        # flattened_graph.add_edge(u, v, w_=weight)
                        flattened_graph.add_edge(u, v, **{attribute: weight})
                elif type_flattening == "average":
                    if flattened_graph.has_edge(u, v):
                        # Track the total weight and occurrence count for averaging
                        edge_weights[(u, v)]['total_weight'] += weight
                        edge_weights[(u, v)]['count'] += 1
                        flattened_graph[u][v][attribute] = edge_weights[(u, v)]['total_weight'] / edge_weights[(u, v)]['count']
                    else:
                        # Add the edge with its weight
                        # flattened_graph.add_edge(u, v, w_=weight)
                        flattened_graph.add_edge(u, v, **{attribute: weight})
                        edge_weights[(u, v)] = {'total_weight': weight, 'count': 1}
                elif type_flattening == "and_sum":
                    # Only add edges that are present in all layers
                    if (u, v) in common_edges or (v, u) in common_edges:
                        if flattened_graph.has_edge(u, v):
                            flattened_graph[u][v][attribute] += weight
                        else:
                            flattened_graph.add_edge(u, v, **{attribute: weight})
                        edge_weights[(u, v)] = {'total_weight': weight, 'count': 1}

                    # Remove isolated nodes (nodes with no edges) from the flattened graph
                    isolated_nodes = [node for node, degree in flattened_graph.degree() if degree == 0]
                    flattened_graph.remove_nodes_from(isolated_nodes)

        self.lm.printl(f"{file_name}. Flattened graph has {flattened_graph.number_of_nodes()} nodes and {flattened_graph.number_of_edges()} edges.")
        return flattened_graph


    def _infomap_to_dict(
        self,
        im: Infomap,
        id_to_node: dict[int, Any],
        id_to_layer: dict[int, str],
    ) -> dict[str, list[Any]]:
        """
            Convert Infomap results into actor/layer/community lists.
            :param im: [Infomap] Executed Infomap instance.
            :param id_to_node: [dict[int, Any]] Mapping from internal node ids to original node ids.
            :param id_to_layer: [dict[int, str]] Mapping from internal layer ids to original layer names.
            :return: [dict[str, list[Any]]] Dictionary with actor, layer, and cid lists.
        """
        actor, layer, cid = [], [], []

        for node in im.nodes:
            actor.append(id_to_node[node.node_id])        # original node id (string)
            layer.append(id_to_layer[node.layer_id])      # original layer name (string)
            cid.append(node.module_id)                    # community id
        return {"actor": actor, "layer": layer, "cid": cid}


    def _run_ginfomap(
        self,
        MG: Any,
        interlayer_weight: float = 0.1,
        log: bool = True,
    ) -> dict[str, list[Any]]:
        """
            Run Infomap on a uunet multiplex network.
            :param MG: [Any] uunet multiplex network.
            :param interlayer_weight: [float] Weight assigned to implicit inter-layer edges between the same node.
            :param log: [bool] Whether to write structural information to the log.
            :return: [dict[str, list[Any]]] Community assignment with original actor and layer ids.
        """
        # Extract dictionary 'layer' -> NetworkX graph
        networkx_graphs = ml.to_nx_dict(MG)
        original_intra_layer_edges = len(self.cm.to_df(ml.edges(MG)))
        
        # --- Build integer mappings for nodes and layers ---
        all_nodes = set().union(*[set(G.nodes()) for G in networkx_graphs.values()])
        node_to_id = {n: i for i, n in enumerate(sorted(all_nodes))}
        id_to_node = {i: n for n, i in node_to_id.items()}

        layer_names = list(networkx_graphs.keys())
        layer_to_id = {L: i for i, L in enumerate(layer_names)}
        id_to_layer = {i: L for L, i in layer_to_id.items()}

        # --- Initialize Infomap ---
        im = Infomap(directed=False)

        intra_links = 0
        inter_links = 0

        # --- Add intra-layer edges ---
        for layer_name, G in networkx_graphs.items():
            L = layer_to_id[layer_name]
            for u, v, data in G.edges(data=True):
                uid = node_to_id[u]
                vid = node_to_id[v]
                w = float(data.get('weight', 1.0))
                im.add_multilayer_intra_link(L, uid, vid, weight=w)
                intra_links += 1

        # --- Add inter-layer edges between same node across different layers ---
        for u in all_nodes:
            uid = node_to_id[u]
            # find layers where the node actually exists
            layers_of_u = [L for L, G in networkx_graphs.items() if u in G]
            for i, L1 in enumerate(layers_of_u):
                for L2 in layers_of_u[i+1:]:
                    im.add_multilayer_inter_link(
                        layer_to_id[L1],
                        uid,
                        layer_to_id[L2],
                        weight=float(interlayer_weight)
                    )
                    inter_links += 1

        # --- Optional Logging ---
        if log:
            total_edges = intra_links + inter_links
            self.lm.printl(f"[Infomap] Original intra-layer edges: {original_intra_layer_edges}")
            self.lm.printl(f"[Infomap] Added intra-layer edges: {intra_links}")
            self.lm.printl(f"[Infomap] Added inter-layer edges: {inter_links}")
            self.lm.printl(f"[Infomap] Total edges passed to Infomap: {total_edges}")
            self.lm.printl(f"[Infomap] Layers: {len(layer_names)} ({layer_names})")
            self.lm.printl(f"[Infomap] Unique nodes: {len(all_nodes)}")
            self.lm.printl(f"[Infomap] Interlayer coupling weight: {interlayer_weight}")

        # --- Run Infomap ---
        im.run()

        if log:
            self.lm.printl(f"[Infomap] Number of modules found: {im.num_top_modules}")
            self.lm.printl(f"[Infomap] Codelength: {im.codelength:.5f}")

        # --- Convert to dict ---
        coms_ginfomap = self._infomap_to_dict(im, id_to_node, id_to_layer)
        return coms_ginfomap


    def _run_infomap_cdlib_compatible(
        self,
        G: nx.Graph,
        weight: str | None = None,
        silent: bool = True,
    ) -> NodeClustering:
        """
            Run Infomap on a NetworkX graph and return a CDlib-compatible clustering.
            :param G: [nx.Graph] Input graph.
            :param weight: [str | None] Edge attribute used as weight, or None for unweighted Infomap.
            :param silent: [bool] Whether to suppress Infomap console output.
            :return: [NodeClustering] CDlib-compatible clustering object.
        """
        im = Infomap(silent=silent)

        # Map nodes to integer IDs (Infomap requires integer node IDs)
        node_to_id = {node: i for i, node in enumerate(G.nodes())}
        id_to_node = {i: node for node, i in node_to_id.items()}

        # Add edges
        for u, v, data in G.edges(data=True):
            if weight is not None and weight in data:
                w = float(data[weight])
                im.add_link(node_to_id[u], node_to_id[v], w)
            else:
                im.add_link(node_to_id[u], node_to_id[v])

        # Run Infomap
        im.run()

        # Extract communities and map back to original node labels
        communities = {}
        for node in im.nodes:
            communities.setdefault(node.module_id, []).append(id_to_node[node.node_id])

        com_list = list(communities.values())

        # Wrap result in CDlib NodeClustering for compatibility
        coms_infomap = NodeClustering(
            communities=com_list,
            graph=G,
            method_name="infomap",
        )
        return coms_infomap


    def flatten_multiplex_network(self, MG: Any) -> nx.Graph:
        """
            Flatten a multiplex network according to the selected flat algorithm variant.
            :param MG: [Any] uunet multiplex network.
            :return: [nx.Graph] Flattened NetworkX graph used by single-layer community algorithms.
        """
        self.lm.printl(f"{file_name}. Preparing multiplex flattening for algorithm: {self.get_algorithm_name()}.")
        if self.get_algorithm_name() in ['flat_ec_louvain', 'flat_ec', 'flat_ec_infomap']:
            method = 'weighted'
        elif self.get_algorithm_name() in ['flat_nw_louvain', 'flat_nw', 'flat_nw_infomap']:
            method = 'or'
        elif self.get_algorithm_name() in ['flat_weighted_sum_louvain', 'flat_weighted_sum_infomap']:
            method = 'sum'
        elif self.get_algorithm_name() in ['flat_weighted_average_louvain', 'flat_weighted_average_infomap']:
            method = 'average'
        elif self.get_algorithm_name() in ['flat_and_weighted_sum_louvain', 'flat_and_weighted_sum_infomap']:
            method = 'and_sum'
        else:
            raise ValueError(f"Unknown flattening algorithm: {self.get_algorithm_name()}.")

        self.lm.printl(
            f"{file_name}. flatten_multiplex_network algorithm: {self.get_algorithm_name()}, selected method: {method}.")
        if method in ['weighted', 'or']:
            ml.flatten(MG, layers=ml.layers(MG), method=method)
            G = ml.to_nx_dict(MG)['flattening']
            # Multinet by default uses 'weight' as the attribute name for edge weights.
            # So the flattened graph will have 'weight' attribute for edges.
            # Rename the 'weight' attribute to 'w_' to be compatible with other algorithms
            for u, v, data in G.edges(data=True):
                if 'weight' in data:
                    data[W_VAR] = data.pop('weight')
        elif method in ['sum', 'average', 'and_sum']:
            G = self._custom_flattening(MG, type_flattening=method)

        # The 'weight' attribute with "or" flattening is set to zero for all edges. This can create issues in the
        # Gephi visualization. To avoid this, we remove the 'weight' attribute from all edges.
        if method == 'or':
            # Remove the 'weight' attribute from all edges
            for u, v, attrs in G.edges(data=True):
                if W_VAR in attrs:
                    del attrs[W_VAR]

        self.lm.printl(f"{file_name}. Multiplex flattening ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G

    def compute_communities(self, G: Any) -> Any:
        """
            Compute communities with the configured algorithm.
            :param G: [Any] NetworkX graph or uunet multiplex network, depending on the algorithm.
            :return: [Any] Community assignment object returned by the selected algorithm.
        """
        start_time = time.time()
        self.lm.printl(f"{file_name}. Computing communities with {str(self)}.")
        if self.algorithm_name == "clique_percolation":
            # k = 3 : int Minimum number of actors in a clique. Must be at least 3.
            # m = 1 : int Minimum number of common layers in a clique.
            comm = ml.clique_percolation(G, k=self.parameters['k'], m=self.parameters['m'])
        elif self.algorithm_name == "ginfomap":
            comm = self._run_ginfomap(G, self.parameters['interlayer_weight'])
            # comm = ml.infomap(G)
        elif self.algorithm_name == "abacus":
             # min.actors = 3 : int Minimum number of actors to form a community.
             # min.layers = 1 : int Minimum number of times two actors must be in the same single-layer community to be
             # considered in the same multi-layer community.
            comm = ml.abacus(G, self.parameters['min_actors'], self.parameters['min_layers'])
        elif self.algorithm_name == "glouvain":
            comm = ml.glouvain(G, omega=self.parameters['omega'], gamma=self.parameters['gamma'])
        elif self.algorithm_name == "multidimensional_label_propagation":
            comm = ml.mdlp(G)
        elif self.algorithm_name == "flat_ec":
            comm = ml.flat_ec(G)
        elif self.algorithm_name == "flat_nw":
            comm = ml.flat_nw(G)
        elif self.algorithm_name in ["louvain", "flat_ec_louvain", 'flat_weighted_sum_louvain', 'flat_weighted_average_louvain', 'flat_and_weighted_sum_louvain']:
            comm = algorithms.louvain(G, weight=W_VAR, resolution=self.parameters["resolution"], randomize=False)
        elif self.algorithm_name == 'flat_nw_louvain':
            comm = algorithms.louvain(G, resolution=self.parameters["resolution"], randomize=False)
        elif self.algorithm_name in ["infomap", "flat_ec_infomap", 'flat_weighted_sum_infomap', 'flat_weighted_average_infomap', 'flat_and_weighted_sum_infomap']:
            comm = self._run_infomap_cdlib_compatible(G, weight=W_VAR)
        elif self.algorithm_name == 'flat_nw_infomap':
            comm = self._run_infomap_cdlib_compatible(G)
        else:
            raise ValueError(f"Unknown community-detection algorithm: {self.algorithm_name}.")

        finish_time = time.time()
        delta_time = finish_time - start_time
        self.lm.printl(f"{file_name}. Communities computed with {str(self)} in {delta_time:.2f} seconds.")
        return comm
