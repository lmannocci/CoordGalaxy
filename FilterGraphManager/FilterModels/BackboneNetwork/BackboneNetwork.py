import os

import networkx as nx
import numpy as np
from scipy import integrate

from utils.LogManager.LogManager import LogManager
from utils.decorator_definition import log_method


file_name = os.path.splitext(os.path.basename(__file__))[0]


class BackboneNetwork:
    def __init__(self) -> None:
        """
            Create the backbone-network helper.
            :return: None.
        """
        self.lm = LogManager('main')

    @log_method
    def disparity_filter(
        self,
        G: nx.Graph | nx.DiGraph,
        weight: str = 'weight',
    ) -> nx.Graph | nx.DiGraph:
        """
            Compute disparity-filter significance scores for weighted edges.
            :param G: [nx.Graph | nx.DiGraph] Weighted NetworkX graph.
            :param weight: [str] Edge attribute containing the edge weight.
            :return: [nx.Graph | nx.DiGraph] Graph containing original weights and alpha scores.
        """
        if nx.is_directed(G):
            return self._disparity_filter_directed(G, weight)
        return self._disparity_filter_undirected(G, weight)

    def _disparity_filter_directed(self, G: nx.DiGraph, weight: str) -> nx.DiGraph:
        """
            Compute disparity-filter alpha scores for a directed graph.
            :param G: [nx.DiGraph] Weighted directed graph.
            :param weight: [str] Edge attribute containing the edge weight.
            :return: [nx.DiGraph] Directed graph with alpha_in and alpha_out edge attributes.
        """
        filtered_graph = nx.DiGraph()
        for u in G:
            out_degree = G.out_degree(u)
            in_degree = G.in_degree(u)

            if out_degree > 1:
                sum_weight_out = sum(np.absolute(G[u][v][weight]) for v in G.successors(u))
                for v in G.successors(u):
                    edge_weight = G[u][v][weight]
                    p_ij_out = float(np.absolute(edge_weight)) / sum_weight_out
                    alpha_ij_out = 1 - (out_degree - 1) * integrate.quad(
                        lambda x: (1 - x) ** (out_degree - 2),
                        0,
                        p_ij_out,
                    )[0]
                    filtered_graph.add_edge(u, v, weight=edge_weight, alpha_out=float('%.4f' % alpha_ij_out))
            elif out_degree == 1:
                successors = list(G.successors(u))
                if len(successors) == 1 and G.in_degree(successors[0]) == 1:
                    v = successors[0]
                    edge_weight = G[u][v][weight]
                    filtered_graph.add_edge(u, v, weight=edge_weight, alpha_out=0., alpha_in=0.)

            if in_degree > 1:
                sum_weight_in = sum(np.absolute(G[v][u][weight]) for v in G.predecessors(u))
                for v in G.predecessors(u):
                    edge_weight = G[v][u][weight]
                    p_ij_in = float(np.absolute(edge_weight)) / sum_weight_in
                    alpha_ij_in = 1 - (in_degree - 1) * integrate.quad(
                        lambda x: (1 - x) ** (in_degree - 2),
                        0,
                        p_ij_in,
                    )[0]
                    filtered_graph.add_edge(v, u, weight=edge_weight, alpha_in=float('%.4f' % alpha_ij_in))
        return filtered_graph

    def _disparity_filter_undirected(self, G: nx.Graph, weight: str) -> nx.Graph:
        """
            Compute disparity-filter alpha scores for an undirected graph.
            :param G: [nx.Graph] Weighted undirected graph.
            :param weight: [str] Edge attribute containing the edge weight.
            :return: [nx.Graph] Undirected graph with alpha edge attributes.
        """
        filtered_graph = nx.Graph()
        for u in G:
            degree = len(G[u])
            if degree > 1:
                sum_weight = sum(np.absolute(G[u][v][weight]) for v in G[u])
                for v in G[u]:
                    edge_weight = G[u][v][weight]
                    p_ij = float(np.absolute(edge_weight)) / sum_weight
                    alpha_ij = 1 - (degree - 1) * integrate.quad(
                        lambda x: (1 - x) ** (degree - 2),
                        0,
                        p_ij,
                    )[0]
                    filtered_graph.add_edge(u, v, weight=edge_weight, alpha=float('%.4f' % alpha_ij))
        return filtered_graph

    @log_method
    def disparity_filter_alpha_cut(
        self,
        G: nx.Graph | nx.DiGraph,
        weight: str = 'weight',
        alpha_t: float = 0.4,
        cut_mode: str = 'or',
    ) -> nx.Graph | nx.DiGraph:
        """
            Apply an alpha threshold to a graph produced by disparity_filter.
            :param G: [nx.Graph | nx.DiGraph] Weighted graph with alpha edge attributes.
            :param weight: [str] Edge attribute containing the edge weight.
            :param alpha_t: [float] Maximum alpha value kept in the backbone.
            :param cut_mode: [str] Directed-graph rule for alpha_in and alpha_out. Accepted values are or and and.
            :return: [nx.Graph | nx.DiGraph] Graph containing only edges that pass the alpha threshold.
        """
        if nx.is_directed(G):
            return self._alpha_cut_directed(G, weight, alpha_t, cut_mode)
        return self._alpha_cut_undirected(G, weight, alpha_t)

    def _alpha_cut_directed(
        self,
        G: nx.DiGraph,
        weight: str,
        alpha_t: float,
        cut_mode: str,
    ) -> nx.DiGraph:
        """
            Apply alpha thresholding to a directed graph.
            :param G: [nx.DiGraph] Directed graph with alpha_in and alpha_out edge attributes.
            :param weight: [str] Edge attribute containing the edge weight.
            :param alpha_t: [float] Maximum alpha value kept in the backbone.
            :param cut_mode: [str] Rule for combining alpha_in and alpha_out.
            :return: [nx.DiGraph] Thresholded directed graph.
        """
        filtered_graph = nx.DiGraph()
        for u, v, edge_data in G.edges(data=True):
            alpha_in = edge_data.get('alpha_in', 1)
            alpha_out = edge_data.get('alpha_out', 1)
            if cut_mode == 'or' and (alpha_in < alpha_t or alpha_out < alpha_t):
                filtered_graph.add_edge(u, v, weight=edge_data[weight])
            elif cut_mode == 'and' and (alpha_in < alpha_t and alpha_out < alpha_t):
                filtered_graph.add_edge(u, v, weight=edge_data[weight])
        return filtered_graph

    def _alpha_cut_undirected(self, G: nx.Graph, weight: str, alpha_t: float) -> nx.Graph:
        """
            Apply alpha thresholding to an undirected graph.
            :param G: [nx.Graph] Undirected graph with alpha edge attributes.
            :param weight: [str] Edge attribute containing the edge weight.
            :param alpha_t: [float] Maximum alpha value kept in the backbone.
            :return: [nx.Graph] Thresholded undirected graph.
        """
        filtered_graph = nx.Graph()
        for u, v, edge_data in G.edges(data=True):
            if edge_data.get('alpha', 1) < alpha_t:
                filtered_graph.add_edge(u, v, weight=edge_data[weight])
        return filtered_graph

    def _example_main(self) -> None:
        """
            Run a small local example of the disparity-filter backbone.
            :return: None. Example results are printed to stdout.
        """
        G = nx.barabasi_albert_graph(1000, 5)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 100)
        alpha = 0.05
        G = self.disparity_filter(G)
        G2 = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if d['alpha'] < alpha])
        print('alpha = %s' % alpha)
        print('original: nodes = %s, edges = %s' % (G.number_of_nodes(), G.number_of_edges()))
        print('backbone: nodes = %s, edges = %s' % (G2.number_of_nodes(), G2.number_of_edges()))
        print(G2.edges(data=True))
