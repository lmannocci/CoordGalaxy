import pandas as pd

from SimilarityFunctionManager.methods.similarityFunction import *
from utils.PlotManager.PlotManager import PlotManager
from utils.common_variables import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *
from DirectoryManager import DirectoryManager
from utils.decorator_definition import log_method

import uunet.multinet as ml
import os
import time
import matplotlib.pyplot as plt
import networkx as nx
import statistics
import numpy as np
from collections import defaultdict
import seaborn as sns
from itertools import combinations
from typing import Any, Sequence

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class NetworkMeasure:
    NETWORK_METRICS_FILENAME = "network_metrics.csv"
    LEGACY_METRICS_FILENAME = "metrics.csv"

    def __init__(
        self,
        list_ca: Sequence[Any],
        dict_ca_filter: dict[str, Any],
        icm: Any,
        dm: DirectoryManager,
        type_algorithm: str
    ) -> None:
        """
            Create the network-measure helper for single-layer and multiplex network characterization.
            :param list_ca: Co-action objects included in the characterization.
            :param dict_ca_filter: Filter configuration by co-action id.
            :param icm: Integrity constraint manager.
            :param dm: Directory manager with analysis paths.
            :param type_algorithm: Algorithm type detected by DirectoryManager.
            :return: None.
        """

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

    def _get_files_path_info(self, filter_instance: Any | None, dict_path: dict[str, str]) -> tuple[bool, str, str, list[str]]:
        """
            Return the network artifact path and file list for filtered or not-filtered networks.
            :param filter_instance: Filter instance for the co-action, or None for not-filtered networks.
            :param dict_path: DirectoryManager path dictionary for one co-action.
            :return: Tuple containing is_graph, path, analysis path, and sorted filenames.
        """
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

    def _largest_component_subgraph(self, graph: nx.Graph) -> nx.Graph:
        """
            Return the largest connected component as a graph.
            :param graph: NetworkX graph.
            :return: Largest connected component subgraph, or the original graph when it has no nodes.
        """
        if graph.number_of_nodes() == 0:
            return graph
        if graph.is_directed():
            components = nx.weakly_connected_components(graph)
        else:
            components = nx.connected_components(graph)
        largest_component = max(components, key=len)
        return graph.subgraph(largest_component).copy()

    def _metric_output_columns(self, metric: str) -> list[str]:
        """
            Return dataframe columns produced by one metric.
            :param metric: Metric name requested by the caller.
            :return: Output column names, or an empty list for plot-only metrics.
        """
        output_columns = {
            'nNodes': ['nNodes'],
            'nEdges': ['nEdges'],
            'weight_statistics': ['meanWeight', 'medianWeight', 'stdDevWeight', 'maxWeight', 'minWeight'],
            'connected_components': ['nConnectedComponents'],
            'nConnectedComponents': ['nConnectedComponents'],
            'sizeLargestComponent': ['sizeLargestComponent'],
            'density': ['density'],
            'directed': ['directed'],
            'clusteringCoefficient': ['clusteringCoefficient'],
            'averagePathLength': ['averagePathLength'],
            'diameter': ['diameter'],
            'assortativity': ['assortativity'],
            'degree_centrality': ['degreeCentrality'],
            'betweenness_centrality': ['betweennessCentrality'],
            'closeness_centrality': ['closenessCentrality'],
            'shortest_path_lengths': ['shortestPathLengths'],
            'eccentricity': ['eccentricity'],
        }
        return output_columns.get(metric, [])

    def _metric_already_available(self, row: dict[str, Any], metric: str) -> bool:
        """
            Check whether one metric is already available in an existing metrics row.
            :param row: Existing metrics row keyed by column name.
            :param metric: Metric requested by the caller.
            :return: True when all dataframe columns for the metric are present and not null.
        """
        output_columns = self._metric_output_columns(metric)
        if len(output_columns) == 0:
            return False
        for column in output_columns:
            if column not in row or pd.isna(row[column]):
                return False
        return True

    def _metrics_output_path(self, path_analysis: str) -> str:
        """
            Return the canonical network-metrics output path.
            :param path_analysis: Analysis directory.
            :return: Path to network_metrics.csv.
        """
        return path_analysis + self.NETWORK_METRICS_FILENAME

    def _legacy_metrics_output_path(self, path_analysis: str) -> str:
        """
            Return the legacy metrics output path.
            :param path_analysis: Analysis directory.
            :return: Path to metrics.csv.
        """
        return path_analysis + self.LEGACY_METRICS_FILENAME

    def _load_existing_network_metrics(self, path_analysis: str, use_existing: bool) -> pd.DataFrame:
        """
            Load an existing network metrics dataframe, including the legacy filename when needed.
            :param path_analysis: Analysis directory.
            :param use_existing: Whether existing outputs should be read.
            :return: Existing metrics dataframe, or an empty dataframe.
        """
        if not use_existing:
            return pd.DataFrame()

        output_path = self._metrics_output_path(path_analysis)
        legacy_path = self._legacy_metrics_output_path(path_analysis)
        if os.path.exists(output_path):
            return pd.read_csv(output_path, dtype=dtype)
        if os.path.exists(legacy_path):
            self.lm.printl(
                f"{file_name}. Found legacy metrics file {legacy_path}; "
                f"future output will be saved as {output_path}."
            )
            return pd.read_csv(legacy_path, dtype=dtype)
        return pd.DataFrame()

    def _row_key(self, layer: str, filter_label: str, tw: str) -> tuple[str, str, str]:
        """
            Build the unique row key for a network metrics row.
            :param layer: Co-action layer id.
            :param filter_label: Filter abbreviation.
            :param tw: Time-window or merged edge-list filename.
            :return: Row key.
        """
        return layer, filter_label, tw

    def _metrics_dataframe_to_rows(self, metrics_df: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
        """
            Convert a metrics dataframe to keyed row dictionaries.
            :param metrics_df: Existing metrics dataframe.
            :return: Dictionary keyed by layer, filter, and tw.
        """
        rows = {}
        if metrics_df.empty:
            return rows
        for _, row in metrics_df.iterrows():
            key = self._row_key(str(row['layer']), str(row['filter']), str(row['tw']))
            rows[key] = row.to_dict()
        return rows

    def _save_network_metrics_outputs(
        self,
        multi_layer_rows: dict[tuple[str, str, str], dict[str, Any]],
        layer_rows_by_path: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
    ) -> None:
        """
            Save current network metric rows to the correct output CSV.
            :param multi_layer_rows: Aggregated metric rows for multi-layer output.
            :param layer_rows_by_path: Per-layer metric rows keyed by analysis path.
            :return: None. CSV files are written to disk.
        """
        if self.type_algorithm == "one-layer":
            for path_analysis, layer_rows in layer_rows_by_path.items():
                metrics_df = pd.DataFrame(layer_rows.values())
                self.ch.save_dataframe(metrics_df, self._metrics_output_path(path_analysis))
        elif self.type_algorithm == "multi-layer":
            metrics_df = pd.DataFrame(multi_layer_rows.values())
            self.ch.save_dataframe(metrics_df, self._metrics_output_path(self.dm.path_analysis))

    def _log_metric_start(self, metric: str, type_ca: str) -> float:
        """
            Log that one metric computation is starting.
            :param metric: Metric name.
            :param type_ca: Co-action id being characterized.
            :return: Start timestamp.
        """
        self.lm.printl(f"{file_name}. metric={metric}, layer={type_ca}: start.")
        return time.perf_counter()

    def _log_metric_completed(self, metric: str, type_ca: str, start_time: float) -> None:
        """
            Log that one metric computation completed.
            :param metric: Metric name.
            :param type_ca: Co-action id being characterized.
            :param start_time: Start timestamp returned by _log_metric_start.
            :return: None.
        """
        elapsed = time.perf_counter() - start_time
        self.lm.printl(f"{file_name}. metric={metric}, layer={type_ca}: completed in {elapsed:.3f}s.")

    def _connected_components(self, graph: nx.Graph) -> list[set]:
        """
            Return connected components for directed or undirected graphs.
            :param graph: NetworkX graph.
            :return: List of connected component node sets.
        """
        if graph.is_directed():
            return list(nx.weakly_connected_components(graph))
        return list(nx.connected_components(graph))

    def _first_multiplex_graph_file(self) -> tuple[str, str, str]:
        """
            Return the first multiplex graph txt file and derived identifiers.
            :return: Tuple with filename, filename without extension, and absolute path.
        """
        graph_files = sorted(pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt'))
        if len(graph_files) == 0:
            message = f"{file_name}. No multiplex graph txt files found in {self.dm.path_multi_graph}."
            self.lm.printl(message)
            raise FileNotFoundError(message)
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        return net_filename, net_filename_no_ext, self.dm.path_multi_graph + net_filename

    def _layer_comparison_filename(self, comparison: str, net_filename_no_ext: str = "multiplex_graph") -> str:
        """
            Build the output filename for one layer-comparison matrix.
            :param comparison: Layer-comparison method name.
            :param net_filename_no_ext: Multiplex graph filename without extension.
            :return: CSV filename for the comparison matrix.
        """
        return comparison.replace(".", "_") + "_" + net_filename_no_ext + '.csv'

    def _compute_layer_comparison_matrix(self, multiplex_graph: Any, comparison: str) -> pd.DataFrame:
        """
            Compute one multiplex layer-comparison matrix.
            :param multiplex_graph: Uunet multiplex graph.
            :param comparison: Uunet layer-comparison method.
            :return: Comparison matrix dataframe.
        """
        df = self.cm.to_df(ml.layer_comparison(multiplex_graph, method=comparison))
        df.columns = ml.layers(multiplex_graph)
        df.index = ml.layers(multiplex_graph)
        return df.reset_index(names='layer')

    def _comparison_heatmap_style(self, comparison: str) -> tuple[float, float, Any]:
        """
            Return heatmap bounds and palette for one comparison metric.
            :param comparison: Layer-comparison method name.
            :return: Tuple with minimum value, maximum value, and seaborn palette.
        """
        if comparison == 'pearson.degree':
            return -1, 1, sns.color_palette("Spectral", as_cmap=True)
        return 0, 1, sns.color_palette("viridis", as_cmap=True).reversed()

    def _prepare_layer_comparison_plot_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Prepare a layer-comparison dataframe for heatmap plotting.
            :param df: Raw layer-comparison dataframe.
            :return: Ordered dataframe indexed by display layer name.
        """
        available_layers = [column for column in df.columns if column != 'layer']
        configured_layers = []
        for ca in self.list_ca:
            layer_name = get_co_action_layer_name(ca.get_co_action())
            if layer_name in available_layers and layer_name not in configured_layers:
                configured_layers.append(layer_name)

        # Keep unexpected layers instead of failing. This is useful when reading legacy
        # multiplex files or layer names produced outside the current co-action registry.
        layer_order = configured_layers + [
            layer_name
            for layer_name in available_layers
            if layer_name not in configured_layers
        ]

        plot_df = df[['layer'] + layer_order].copy()
        plot_df = plot_df[plot_df['layer'].isin(layer_order)]

        display_label_map = {
            layer_name: co_action_column_print2.get(layer_name, layer_name)
            for layer_name in layer_order
        }
        display_order = [display_label_map[layer_name] for layer_name in layer_order]

        plot_df = plot_df.rename(columns=display_label_map)
        plot_df['layer'] = plot_df['layer'].replace(display_label_map)
        plot_df['layer'] = pd.Categorical(plot_df['layer'], categories=display_order, ordered=True)
        return plot_df[display_order + ['layer']].sort_values('layer').set_index('layer')

    def _plot_layer_comparison_heatmap(self, df: pd.DataFrame, comparison: str) -> None:
        """
            Plot one multiplex layer-comparison heatmap.
            :param df: Raw layer-comparison dataframe.
            :param comparison: Layer-comparison method name.
            :return: None. The heatmap is saved to the analysis directory.
        """
        aggregated_df = self._prepare_layer_comparison_plot_df(df)
        min_v, max_v, custom_palette = self._comparison_heatmap_style(comparison)
        fig, ax = plt.subplots(figsize=(8, 6.5))

        if comparison == 'coverage.edges' or comparison == 'coverage.actors':
            sns.heatmap(
                aggregated_df,
                annot=True,
                fmt=".3f",
                cmap=custom_palette,
                cbar=True,
                ax=ax,
                vmin=min_v,
                vmax=max_v,
                linewidths=0.5,
                linecolor='white',
                annot_kws={'size': 12}
            )
        else:
            mask = np.triu(np.ones_like(aggregated_df, dtype=bool), k=1)
            sns.heatmap(
                aggregated_df,
                mask=mask,
                annot=True,
                fmt=".3f",
                cmap=custom_palette,
                cbar=True,
                ax=ax,
                vmin=min_v,
                vmax=max_v,
                linewidths=0.5,
                linecolor='white',
                annot_kws={'size': 12}
            )

        ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=90, fontsize=16)
        plt.savefig(
            f"{self.dm.path_analysis}{comparison.replace('.', '_')}_layer_comparison_heatmaps.png",
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)

    def _plot_saved_layer_comparisons(self, net_filename_no_ext: str = "multiplex_graph") -> None:
        """
            Plot all saved multiplex layer-comparison matrices.
            :param net_filename_no_ext: Multiplex graph filename without extension.
            :return: None. Heatmap figures are saved to the analysis directory.
        """
        for comparison in comparison_type:
            file_path = os.path.join(
                self.dm.path_analysis,
                self._layer_comparison_filename(comparison, net_filename_no_ext)
            )
            df = pd.read_csv(file_path)
            self._plot_layer_comparison_heatmap(df, comparison)

    def _compute_network_measure(
        self,
        network: Any,
        is_graph: bool,
        type_ca: str,
        path_analysis: str,
        metrics_to_compute: Sequence[str],
        metrics_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
            Compute selected metrics for one network artifact.
            :param network: NetworkX graph or edge-list object.
            :param is_graph: Whether network is already a NetworkX graph.
            :param type_ca: Co-action id being characterized.
            :param path_analysis: Output analysis directory.
            :param metrics_to_compute: Metric names requested by the caller.
            :param metrics_dict: Existing metric dictionary to update.
            :return: Updated metric dictionary.
        """
        if 'weight_statistics' in metrics_to_compute:
            start_time = self._log_metric_start('weight_statistics', type_ca)
            stats_dict = self._compute_edge_weight_statistics(is_graph, network)
            metrics_dict.update(stats_dict)
            self._plot_edge_weight_distribution(path_analysis, type_ca, is_graph, network)
            self._log_metric_completed('weight_statistics', type_ca, start_time)

        if 'nNodes' in metrics_to_compute:
            start_time = self._log_metric_start('nNodes', type_ca)
            metrics_dict['nNodes'] = self._nNodes(is_graph, network)
            self._log_metric_completed('nNodes', type_ca, start_time)

        if 'nEdges' in metrics_to_compute:
            start_time = self._log_metric_start('nEdges', type_ca)
            metrics_dict['nEdges'] = self._nEdges(is_graph, network)
            self._log_metric_completed('nEdges', type_ca, start_time)

        if "node_topEdge_trend" in metrics_to_compute:
            start_time = self._log_metric_start('node_topEdge_trend', type_ca)
            self._plot_node_topEdge_trend(path_analysis, type_ca, is_graph, network)
            self._log_metric_completed('node_topEdge_trend', type_ca, start_time)

        if 'degree_distribution' in metrics_to_compute:
            start_time = self._log_metric_start('degree_distribution', type_ca)
            self._plot_degree_distribution(path_analysis, type_ca, is_graph, network)
            self._log_metric_completed('degree_distribution', type_ca, start_time)

        if 'nAction_distribution' in metrics_to_compute:
            start_time = self._log_metric_start('nAction_distribution', type_ca)
            self._show_nAction_distribution(path_analysis, type_ca, is_graph, network)
            self._log_metric_completed('nAction_distribution', type_ca, start_time)

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

            connected_components = None
            largest_component_graph = None

            # This metric corresponds to the "dir" field in uunet.summary.
            if 'directed' in metrics_to_compute:
                start_time = self._log_metric_start('directed', type_ca)
                metrics_dict['directed'] = G.is_directed()
                self._log_metric_completed('directed', type_ca, start_time)

            # This metric corresponds to the "dens" field in uunet.summary.
            if 'density' in metrics_to_compute:
                start_time = self._log_metric_start('density', type_ca)
                metrics_dict['density'] = nx.density(G)
                self._log_metric_completed('density', type_ca, start_time)

            # Assortativity.
            if "assortativity" in metrics_to_compute:
                start_time = self._log_metric_start('assortativity', type_ca)
                assortativity = nx.degree_assortativity_coefficient(G)
                metrics_dict['assortativity'] = assortativity
                self._log_metric_completed('assortativity', type_ca, start_time)

            # Degree centrality.
            if 'degree_centrality' in metrics_to_compute:
                start_time = self._log_metric_start('degree_centrality', type_ca)
                degree_centrality = nx.degree_centrality(G)
                metrics_dict['degreeCentrality'] = degree_centrality
                self._log_metric_completed('degree_centrality', type_ca, start_time)

            # Betweenness centrality.
            if 'betweenness_centrality' in metrics_to_compute:
                start_time = self._log_metric_start('betweenness_centrality', type_ca)
                betweenness_centrality = nx.betweenness_centrality(G)
                metrics_dict['betweennessCentrality'] = betweenness_centrality
                self._log_metric_completed('betweenness_centrality', type_ca, start_time)

            # Closeness centrality.
            if 'closeness_centrality' in metrics_to_compute:
                start_time = self._log_metric_start('closeness_centrality', type_ca)
                closeness_centrality = nx.closeness_centrality(G)
                metrics_dict['closenessCentrality'] = closeness_centrality
                self._log_metric_completed('closeness_centrality', type_ca, start_time)

            # Shortest path lengths.
            if 'shortest_path_lengths' in metrics_to_compute:
                start_time = self._log_metric_start('shortest_path_lengths', type_ca)
                shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
                metrics_dict['shortestPathLengths'] = shortest_path_lengths
                self._log_metric_completed('shortest_path_lengths', type_ca, start_time)

            # Eccentricity.
            if 'eccentricity' in metrics_to_compute:
                start_time = self._log_metric_start('eccentricity', type_ca)
                eccentricity = nx.eccentricity(G)
                metrics_dict['eccentricity'] = eccentricity
                self._log_metric_completed('eccentricity', type_ca, start_time)

            # Connected components and largest component size correspond to "nc" and "slc" in uunet.summary.
            if 'connected_components' in metrics_to_compute:
                start_time = self._log_metric_start('connected_components', type_ca)
                connected_components = self._connected_components(G)
                metrics_dict['nConnectedComponents'] = len(connected_components)
                self._plot_component_sizes_distribution(connected_components, path_analysis, type_ca)
                self._log_metric_completed('connected_components', type_ca, start_time)

            if 'nConnectedComponents' in metrics_to_compute and 'connected_components' not in metrics_to_compute:
                start_time = self._log_metric_start('nConnectedComponents', type_ca)
                connected_components = connected_components or self._connected_components(G)
                metrics_dict['nConnectedComponents'] = len(connected_components)
                self._log_metric_completed('nConnectedComponents', type_ca, start_time)

            if 'sizeLargestComponent' in metrics_to_compute:
                start_time = self._log_metric_start('sizeLargestComponent', type_ca)
                connected_components = connected_components or self._connected_components(G)
                metrics_dict['sizeLargestComponent'] = max((len(component) for component in connected_components), default=0)
                self._log_metric_completed('sizeLargestComponent', type_ca, start_time)

            # This metric corresponds to the "cc" field in uunet.summary.
            if 'clusteringCoefficient' in metrics_to_compute:
                start_time = self._log_metric_start('clusteringCoefficient', type_ca)
                if G.number_of_nodes() == 0:
                    metrics_dict['clusteringCoefficient'] = 0
                else:
                    metrics_dict['clusteringCoefficient'] = nx.average_clustering(G.to_undirected())
                self._log_metric_completed('clusteringCoefficient', type_ca, start_time)

            if 'averagePathLength' in metrics_to_compute:
                start_time = self._log_metric_start('averagePathLength', type_ca)
                largest_component_graph = largest_component_graph or self._largest_component_subgraph(G).to_undirected()
                if largest_component_graph.number_of_nodes() <= 1:
                    metrics_dict['averagePathLength'] = 0
                else:
                    metrics_dict['averagePathLength'] = nx.average_shortest_path_length(largest_component_graph)
                self._log_metric_completed('averagePathLength', type_ca, start_time)

            if 'diameter' in metrics_to_compute:
                start_time = self._log_metric_start('diameter', type_ca)
                largest_component_graph = largest_component_graph or self._largest_component_subgraph(G).to_undirected()
                if largest_component_graph.number_of_nodes() <= 1:
                    metrics_dict['diameter'] = 0
                else:
                    metrics_dict['diameter'] = nx.diameter(largest_component_graph)
                self._log_metric_completed('diameter', type_ca, start_time)

        return metrics_dict

    def _show_nAction_distribution(self, path_analysis: str, type_ca: str, is_graph: bool, network: Any) -> None:
        """
            Plot the distribution of nAction values for one network.
            :param path_analysis: Output analysis directory.
            :param type_ca: Co-action id being characterized.
            :param is_graph: Whether network is a NetworkX graph.
            :param network: NetworkX graph or edge-list object.
            :return: None.
        """
        if is_graph:
            nAction = [data[NA_VAR] for _, _, data in network.edges(data=True)]
        else:
            nAction = [e[tuple_index[NA_VAR]] for e in network]
        # Plotting the distribution
        self.pm.plot_histogram(path_analysis, type_ca, nAction,
                               f"Number of common actions per edge", 'Frequency',
                               f"Distribution of number of common actions involved to construct an edge, {type_ca}.",
                               "number_co_action_distribution.png")

    def _compute_edge_weight_statistics(self, is_graph: bool, network: Any) -> dict[str, float]:
        """
            Compute descriptive statistics of edge weights.
            :param is_graph: Whether network is a NetworkX graph.
            :param network: NetworkX graph or edge-list object.
            :return: Dictionary with mean, median, standard deviation, max, and min edge weight.
        """
        if is_graph:
            weights = [data[W_VAR] for _, _, data in network.edges(data=True)]
        else:
            weights = [e[tuple_index[W_VAR]] for e in network]

        if len(weights) == 0:
            self.lm.printl(
                f"{file_name}. Edge-weight statistics skipped: filtered network has no edges."
            )
            return {
                'meanWeight': None,
                'medianWeight': None,
                'stdDevWeight': None,
                'maxWeight': None,
                'minWeight': None
            }

        # Compute statistics
        mean_weight = statistics.mean(weights)
        median_weight = statistics.median(weights)
        std_dev = 0.0 if len(weights) == 1 else statistics.stdev(weights)
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
        return stats_dict

    def _plot_component_sizes_distribution(self, connected_components: Sequence[set], path_analysis: str, type_ca: str) -> None:
        """
            Plot connected-component sizes for one network.
            :param connected_components: Iterable of connected components.
            :param path_analysis: Output analysis directory.
            :param type_ca: Co-action id being characterized.
            :return: None.
        """
        component_sizes = [len(component) for component in connected_components]
        df = pd.DataFrame(component_sizes, columns=['componentSize'])
        df_sorted = df.sort_values(by='componentSize', ascending=False).reset_index(drop=True)
        self.ch.save_dataframe(df_sorted, path_analysis + "size_connected_components.csv")
        self.pm.plot_line(path_analysis, type_ca, df_sorted.index+1, df_sorted['componentSize'], 'Index',
                          'Size of connected components', f'Size of connected components {type_ca}',
                          'plot_size_connected_components.png')
    def _plot_degree_distribution(self, path_analysis: str, type_ca: str, is_graph: bool, network: Any) -> None:
        """
            Plot the node-degree distribution for one network.
            :param path_analysis: Output analysis directory.
            :param type_ca: Co-action id being characterized.
            :param is_graph: Whether network is a NetworkX graph.
            :param network: NetworkX graph or edge-list object.
            :return: None.
        """
        if is_graph:
            degrees = [network.degree(node) for node in network.nodes()]
        else:
            degree_dict = defaultdict(int)
            for edge in network:
                degree_dict[edge[tuple_index[NODE1_VAR]]] += 1
                degree_dict[edge[tuple_index[NODE2_VAR]]] += 1
            degrees = list(degree_dict.values())

        self.pm.plot_histogram(path_analysis, type_ca, degrees, 'Degree', 'Frequency',
                               f'{type_ca} Degree Distribution', 'degree_distribution.png')
    def _plot_edge_weight_distribution(self, path_analysis: str, type_ca: str, is_graph: bool, network: Any) -> None:
        """
            Plot the edge-weight distribution for one network.
            :param path_analysis: Output analysis directory.
            :param type_ca: Co-action id being characterized.
            :param is_graph: Whether network is a NetworkX graph.
            :param network: NetworkX graph or edge-list object.
            :return: None.
        """
        if is_graph:
            weights = [data[W_VAR] for _, _, data in network.edges(data=True)]
        else:
            weights = [e[tuple_index[W_VAR]] for e in network]

        # Plotting the distribution
        self.pm.plot_histogram(path_analysis, type_ca, weights, 'Weights', 'Frequency',
                               f'{type_ca} Distribution of Weights', 'edge_weight_distribution.png')

    def _plot_node_topEdge_trend(self, path_analysis: str, type_ca: str, is_graph: bool, network: Any) -> None:
        """
            Plot how the number of nodes grows as top-weighted edges are added.
            :param path_analysis: Output analysis directory.
            :param type_ca: Co-action id being characterized.
            :param is_graph: Whether network is a NetworkX graph.
            :param network: NetworkX graph or edge-list object.
            :return: None.
        """
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

    def _nNodes(self, is_graph: bool, network: Any) -> int:
        """
            Return the number of nodes in a graph or edge list.
            :param is_graph: Whether network is a NetworkX graph.
            :param network: NetworkX graph or edge-list object.
            :return: Number of nodes.
        """
        if is_graph:
            return network.number_of_nodes()
        else:
            # compute nNodes. edge list in format userId1, userId2, weight, nAction, twCount
            user_set1 = set([e[tuple_index[NODE1_VAR]] for e in network])
            user_set2 = set([e[tuple_index[NODE2_VAR]] for e in network])
            user_set = user_set1.union(user_set2)
            return len(user_set)

    def _nEdges(self, is_graph: bool, network: Any) -> int:
        """
            Return the number of edges in a graph or edge list.
            :param is_graph: Whether network is a NetworkX graph.
            :param network: NetworkX graph or edge-list object.
            :return: Number of edges.
        """
        if is_graph:
            return network.number_of_edges()
        else:
            return len(network)

    def _setUsers(self, is_graph: bool, network: Any) -> set:
        """
            Return the set of users/nodes in a graph or edge list.
            :param is_graph: Whether network is a NetworkX graph.
            :param network: NetworkX graph or edge-list object.
            :return: Set of users/nodes.
        """
        if is_graph:
            return set(network.nodes())
        else:
            user_set1 = set([e[tuple_index[NODE1_VAR]] for e in network])
            user_set2 = set([e[tuple_index[NODE2_VAR]] for e in network])
            user_set = user_set1.union(user_set2)
            return user_set

    def _filter_threshold(self, is_graph: bool, network: Any, threshold: float, filter_par_type: str) -> tuple[bool, list]:
        """
            Filter a graph or edge list by a threshold on an edge attribute.
            :param is_graph: Whether network is a NetworkX graph.
            :param network: NetworkX graph or edge-list object.
            :param threshold: Threshold value.
            :param filter_par_type: Edge attribute used for filtering.
            :return: Tuple containing False and the filtered edge list.
        """
        if is_graph:
            filtered_edge_list = [
                (
                    u,
                    v,
                    data.get(W_VAR, data.get('weight')),
                    data.get(NA_VAR),
                    data.get(TW_VAR),
                )
                for u, v, data in network.edges(data=True)
                if data[filter_par_type] >= threshold
            ]
        else:
            filtered_edge_list = [edge for edge in network if edge[tuple_index[filter_par_type]] >= threshold]
        is_graph_filtered = False
        # Create a new graph with filtered edges
        # filtered_graph = self.__create_weighted_graph(filtered_edge_list)

        return is_graph_filtered, filtered_edge_list

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def compute_network_metrics(
        self,
        metrics_to_compute: Sequence[str],
        recompute_existing: bool = False,
        use_existing_output: bool = True,
    ) -> None:
        """
            Compute requested network metrics for all configured co-action layers.
            :param metrics_to_compute: Metric names to compute.
            :param recompute_existing: If True, recompute requested metrics even when their columns already exist.
            :param use_existing_output: If True, update an existing network_metrics.csv or legacy metrics.csv.
                If False, ignore previous outputs and overwrite network_metrics.csv with newly computed rows.
            :return: None. Network metric dataframes and plots are saved to analysis directories.
        """
        self.icm.check_metrics_networks(metrics_to_compute)

        # The multi-layer output aggregates one row per layer/filter/window. Load it once at the beginning so new
        # metrics can be merged into existing rows without losing columns computed in previous runs.
        multi_layer_rows = self._metrics_dataframe_to_rows(
            self._load_existing_network_metrics(self.dm.path_analysis, use_existing_output)
        )

        # Build an explicit list of graph/edge-list artifacts to process. This lets the method compute one metric
        # across all layers first, save it, and only then move to the next metric.
        work_items = []
        layer_rows_by_path = {}
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            filter_ca = self.dict_ca_filter[type_ca]
            is_graph, path, path_analysis, list_files = self._get_files_path_info(filter_ca, dict_path)

            # One-layer runs save inside the layer analysis directory. Keep those rows separately because they are the
            # canonical output in one-layer mode.
            if path_analysis not in layer_rows_by_path:
                layer_rows_by_path[path_analysis] = self._metrics_dataframe_to_rows(
                    self._load_existing_network_metrics(path_analysis, use_existing_output)
                )
            layer_rows = layer_rows_by_path[path_analysis]

            for elf in list_files:
                if filter_ca is not None:
                    filter_label = filter_ca.filter_repr_abbr()
                else:
                    filter_label = 'None'

                # A metric row is uniquely identified by layer, filter chain, and the graph/edge-list filename.
                # This lets us update an existing row with newly requested metrics while preserving older columns.
                row_key = self._row_key(type_ca, filter_label, elf)
                existing_row = layer_rows.get(row_key, multi_layer_rows.get(row_key, {}))
                metrics_dict = dict(existing_row)
                metrics_dict.update({'layer': type_ca, 'filter': filter_label, 'tw': elf})

                # Update both dictionaries with the final row so one-layer and multi-layer outputs stay aligned.
                layer_rows[row_key] = metrics_dict
                multi_layer_rows[row_key] = metrics_dict
                work_items.append({
                    'type_ca': type_ca,
                    'filter_label': filter_label,
                    'tw': elf,
                    'row_key': row_key,
                    'is_graph': is_graph,
                    'path': path,
                    'path_analysis': path_analysis,
                })

        # Save the base rows before starting expensive metrics. This creates network_metrics.csv even if a later
        # metric fails, and it preserves any rows discovered from legacy metrics.csv files.
        self._save_network_metrics_outputs(multi_layer_rows, layer_rows_by_path)

        # Compute one requested metric across all rows and immediately save the updated dataframe. If the process is
        # interrupted during a later metric, all previously completed metrics remain safely persisted.
        for metric in metrics_to_compute:
            self.lm.printl(f"{file_name}. compute_network_metrics metric={metric}: all layers start.")
            metric_computed = False
            for item in work_items:
                row_key = item['row_key']
                metrics_dict = multi_layer_rows[row_key]

                # If recompute_existing=False, keep this metric when all of its output columns already exist.
                # Plot-only metrics have no output columns and are therefore executed whenever requested.
                if not recompute_existing and self._metric_already_available(metrics_dict, metric):
                    self.lm.printl(
                        f"{file_name}. compute_network_metrics metric={metric}: skip existing "
                        f"layer={item['type_ca']}, filter={item['filter_label']}, tw={item['tw']}."
                    )
                    continue

                self.lm.printl(
                    f"{file_name}. compute_network_metrics metric={metric}: compute "
                    f"layer={item['type_ca']}, filter={item['filter_label']}, tw={item['tw']}."
                )
                # The artifact can be either a NetworkX graph or an edge list, depending on the selected path.
                network = self.ch.load_object(item['path'] + item['tw'])
                metrics_dict = self._compute_network_measure(
                    network,
                    item['is_graph'],
                    item['type_ca'],
                    item['path_analysis'],
                    [metric],
                    metrics_dict,
                )

                # Update both the aggregate and layer-specific dictionaries before moving to the next artifact.
                multi_layer_rows[row_key] = metrics_dict
                layer_rows_by_path[item['path_analysis']][row_key] = metrics_dict
                metric_computed = True

            # Persist after every metric, not only at the end of the whole method. This is the important safety point
            # for expensive metrics such as diameter or betweenness centrality.
            self._save_network_metrics_outputs(multi_layer_rows, layer_rows_by_path)
            if metric_computed:
                self.lm.printl(f"{file_name}. compute_network_metrics metric={metric}: saved output.")
            else:
                self.lm.printl(f"{file_name}. compute_network_metrics metric={metric}: already available for all rows.")

    @log_method
    def compute_threshold_statistics(self, min_th: float, max_th: float, step: float, filter_par_type: str) -> None:
        """
            Compute threshold statistics across all co-action layers.
            :param min_th: Minimum threshold value.
            :param max_th: Maximum threshold value.
            :param step: Threshold step.
            :param filter_par_type: Edge attribute used for filtering, such as nAction or w_.
            :return: None. Dataframes are saved to the global analysis directory.
        """
        layer_data = []
        set_users_dict = {}
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            self.lm.printl(f"{file_name}. {type_ca} start.")
            filter_ca = self.dict_ca_filter[type_ca]
            is_graph, path, path_analysis, list_files = self._get_files_path_info(filter_ca, dict_path)
            elf = list_files[0]
            # this can be a NetworkX graph or an edge list, depending on the attribute is_graph.
            # Both are pickle file, so I can read them in the same mode
            network = self.ch.load_object(path + elf)

            set_users_dict[type_ca] = {}

            nNodes = self._nNodes(is_graph, network)
            nEdges = self._nEdges(is_graph, network)

            threshold = min_th
            while threshold <= max_th:
                self.lm.printl(f"{file_name}. {type_ca}-{str(threshold)} start.")
                # always returns a filtered edge list (is_graph_filtered=False)
                is_graph_filtered, filtered_edge_list = self._filter_threshold(is_graph, network, threshold, filter_par_type)
                self.lm.printl(f"{file_name}. {type_ca}-{str(threshold)} filtered.")

                # set_users_dict contains the set of users/nodes for each co-action/layer, for each threshold
                # of the filtered graph
                set_users_dict[type_ca][threshold] = self._setUsers(is_graph_filtered, filtered_edge_list)

                nFilterNodes = self._nNodes(is_graph_filtered, filtered_edge_list)
                nFilterEdges = self._nEdges(is_graph_filtered, filtered_edge_list)

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
        overlapping_df = pd.DataFrame(
            overlapping_data,
            columns=["layer1", "layer2", "threshold", "overlapping", "percOverlapping"],
        )

        self.ch.save_dataframe(layer_df, self.dm.path_analysis + f'{filter_par_type}_layer_df.csv')
        self.ch.save_dataframe(threshold_df, self.dm.path_analysis + f'{filter_par_type}_threshold_df.csv')
        self.ch.save_dataframe(overlapping_df, self.dm.path_analysis + f'{filter_par_type}_overlapping_df.csv')

    @log_method
    def plot_threshold_overlapping(self, filter_par_type: str, step: float) -> None:
        """
            Plot node-overlap percentages across threshold values.
            :param filter_par_type: Edge attribute used for thresholding.
            :param step: Threshold step used to compute statistics.
            :return: None. Plot artifacts are saved to the analysis directory.
        """
        df = self.ch.read_dataframe(self.dm.path_analysis + f'{filter_par_type}_overlapping_df.csv', dtype=dtype)
        self.pm.plot_grid_combinations(df, self.dm.path_analysis, f"plot_{filter_par_type}_threshold_overlapping.png", "layer1",
                                       'layer2', 'threshold', 'percOverlapping',
                                       'threshold', 'percOverlapping', step)
    @log_method
    def plot_nodes_edges_threshold(self, filter_par_type: str) -> None:
        """
            Plot filtered node and edge counts across threshold values.
            :param filter_par_type: Edge attribute used for thresholding.
            :return: None. Plot artifacts are saved to the analysis directory.
        """
        df = self.ch.read_dataframe(self.dm.path_analysis + f'{filter_par_type}_layer_df.csv', dtype=dtype)

        # plot for each layer/co-action, the number of filtered nodes, for each threshold value
        self.pm.plot_grid_line(self.dm.path_analysis, f"plot_{filter_par_type}_threshold_nNodes.png",df,
                               'layer', 'threshold', 'nFilterNodes',
                               'threshold', 'nFilterNodes', 'nFilterNodes layer')

        # plot for each layer/co-action, the number of filtered edges, for each threshold value
        self.pm.plot_grid_line(self.dm.path_analysis, f"plot_{filter_par_type}_threshold_nEdges.png", df,
                               'layer', 'threshold', 'nFilterEdges',
                               'threshold', 'nFilterEdges', 'nFilterEdges layer')

    @log_method
    def select_threshold_statistics(
        self,
        min_th: float,
        max_th: float,
        step: float,
        absolute_th_mode: bool,
        filter_par_type: str,
        target_type: str
    ) -> None:
        """
            Select per-layer thresholds that match target node or edge constraints.
            :param min_th: Minimum target threshold or percentage.
            :param max_th: Maximum target threshold or percentage.
            :param step: Target threshold step.
            :param absolute_th_mode: Whether target thresholds are absolute counts instead of percentages.
            :param filter_par_type: Edge attribute used for filtering.
            :param target_type: Target type, either node or edge.
            :return: None. Selection and overlap dataframes are saved to the analysis directory.
        """
        # filter_par_type nAction - weight
        # target - node/edge
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
            is_graph, path, path_analysis, list_files = self._get_files_path_info(filter_ca, dict_path)
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
                    is_graph_filtered, filtered_edge_list = self._filter_threshold(is_graph, network, th_layer_parameter[th][type_ca], filter_par_type)

                    # set_users_dict contains the set of users/nodes for each co-action/layer, for each threshold
                    # of the filtered graph
                    set_users_dict[type_ca][th] = self._setUsers(is_graph_filtered, filtered_edge_list)

                    # compute weight statistics for filtered network
                    weight_stat_dict[type_ca][th] = self._compute_edge_weight_statistics(is_graph_filtered, filtered_edge_list)
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

    # Multilayer Characterization Network Measures
    # -------------------------------------------
    @log_method
    def get_ML_layer_comparison(self, mode: str = "both") -> None:
        """
            Compute and/or plot multiplex-layer comparison matrices.
            :param mode: Operation mode. Accepted values are compute, plot, and both.
            :return: None. CSV matrices and/or heatmaps are saved to the multiplex analysis directory.
        """
        if mode not in {"compute", "plot", "both"}:
            raise ValueError("mode must be one of: compute, plot, both.")

        _, net_filename_no_ext, net_path = self._first_multiplex_graph_file()

        if mode in {"compute", "both"}:
            multiplex_graph = self.ch.read_multiplex_network(net_path)
            for comparison in comparison_type:
                df = self._compute_layer_comparison_matrix(multiplex_graph, comparison)
                filename = self._layer_comparison_filename(comparison, net_filename_no_ext)
                self.ch.save_dataframe(df, self.dm.path_analysis + filename)

        if mode in {"plot", "both"}:
            self._plot_saved_layer_comparisons(net_filename_no_ext)

    # Multilayer Characterization Network Measures
    # -------------------------------------------

    @log_method
    def edge_weight_temporal_mean_std(self) -> None:
        """
            Compute temporal mean and standard deviation of edge weights for each co-action layer.
            :return: None. Temporal edge-weight statistics are saved to each layer analysis directory.
        """
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
