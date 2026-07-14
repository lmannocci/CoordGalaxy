import os
import time
from typing import Any, Sequence

import networkx as nx
import pandas as pd

from utils.Checkpoint.Checkpoint import Checkpoint
from utils.ConversionManager.ConversionManager import ConversionManager
from utils.LogManager.LogManager import LogManager
from utils.common_variables import available_node_metrics, dtype, flatten_algorithm
from utils.decorator_definition import log_method


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class NodeMeasure:
    def __init__(
        self,
        list_ca: Sequence[Any],
        dict_ca_filter: dict[str, Any],
        icm: Any,
        dm: Any,
        type_algorithm: str,
        cda: Any | None
    ) -> None:
        """
            Create the node-measure helper for single-layer and flattened community graphs.
            :param list_ca: [Sequence[Any]] Co-action objects included in the characterization.
            :param dict_ca_filter: [dict[str, Any]] Filter configuration by co-action id.
            :param icm: [IntegrityConstraintManager] Constraint checker.
            :param dm: [DirectoryManager] Directory manager with analysis paths.
            :param type_algorithm: [str] Algorithm type detected by DirectoryManager.
            :param cda: [Any | None] Community-detection algorithm configuration.
            :return: None.
        """
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

    def _compute_node_metrics_df(self, graph: nx.Graph, metrics: Sequence[str] | None = None) -> pd.DataFrame:
        """
            Compute selected node-level metrics for a graph.
            :param graph: [nx.Graph] Graph whose nodes may include a group community attribute.
            :param metrics: [Sequence[str] | None] Node metrics to compute. None computes all available metrics.
            :return: [pd.DataFrame] Node metrics dataframe with one row per graph node.
        """
        selected_metrics = self._select_metrics(metrics)
        data: dict[str, list[Any]] = {"userId": list(graph.nodes)}
        node_communities = nx.get_node_attributes(graph, "group")
        data["community"] = [node_communities.get(node) for node in graph.nodes]

        metric_functions = {
            "degree_centrality": lambda: nx.degree_centrality(graph),
            "betweenness_centrality": lambda: nx.betweenness_centrality(graph, normalized=True),
            "closeness_centrality": lambda: nx.closeness_centrality(graph),
            "eigenvector_centrality": lambda: nx.eigenvector_centrality(graph, max_iter=1000),
            "local_clustering_coefficient": lambda: nx.clustering(graph),
            "page_rank": lambda: nx.pagerank(graph),
        }

        for metric_name, metric_function in metric_functions.items():
            if metric_name not in selected_metrics:
                continue
            metric_values = self._time_metric(metric_name, metric_function)
            data[metric_name] = [metric_values[node] for node in graph.nodes]

        return pd.DataFrame(data)

    def _select_metrics(self, metrics: Sequence[str] | None) -> list[str]:
        """
            Return supported node metrics requested by the caller.
            :param metrics: [Sequence[str] | None] Requested metric names. None selects all available node metrics.
            :return: [list[str]] Supported metric names to compute.
        """
        if metrics is None:
            return list(available_node_metrics)
        return [metric for metric in metrics if metric in available_node_metrics]

    def _time_metric(self, metric_name: str, metric_function: Any) -> dict[Any, float]:
        """
            Execute one NetworkX metric function and log its runtime.
            :param metric_name: [str] Metric name used in log messages.
            :param metric_function: [Any] Callable returning a node-to-value dictionary.
            :return: [dict[Any, float]] Metric value by node.
        """
        start = time.time()
        metric_values = metric_function()
        self.lm.printl(f"{metric_name} computed in {time.time() - start:.2f} seconds")
        return metric_values

    def _layer_name(self) -> str | None:
        """
            Return the layer name for node-metric output, or None when unsupported.
            :return: [str | None] Single-layer co-action id or flattened algorithm name.
        """
        if self.type_algorithm == 'one-layer':
            return self.list_ca[0].get_co_action()
        if self.cda is not None and self.cda.get_algorithm_name() in flatten_algorithm:
            return self.cda.get_algorithm_name()
        return None

    def _community_graph_path(self) -> str:
        """
            Return the first community graph pickle path.
            :return: [str] Path to the community graph pickle file.
        """
        graph_files = [filename for filename in os.listdir(self.dm.path_community_graph) if filename.endswith('.p')]
        if len(graph_files) == 0:
            raise FileNotFoundError(f"No community graph pickle files in {self.dm.path_community_graph}.")
        return self.dm.path_community_graph + graph_files[0]

    @log_method
    def compute_network_node_metrics(self, metrics: Sequence[str] | None, merge_existing: bool) -> None:
        """
            Compute node metrics for a single-layer or flattened community graph.
            :param metrics: [Sequence[str] | None] Node metric names to compute. None computes all available metrics.
            :param merge_existing: [bool] Whether to merge new metric columns into an existing output dataframe.
            :return: None. Node metrics are saved under DirectoryManager.path_community_analysis.
        """
        self.icm.check_node_metrics(metrics)
        layer = self._layer_name()
        if layer is None:
            self.lm.printl(f"{file_name}. compute_network_node_metrics: multi-layer case not managed.")
            return

        graph = self.ch.load_object(self._community_graph_path())
        node_metrics_df = self._compute_node_metrics_df(graph, metrics)
        node_metrics_df['layer'] = layer
        output_path = self.dm.path_community_analysis + f"{layer}_node_metrics.csv"

        if merge_existing:
            self.ch.update_columns_dataframe(node_metrics_df, output_path, ['userId', 'layer'], dtype)
        else:
            self.ch.save_dataframe(node_metrics_df, output_path)
