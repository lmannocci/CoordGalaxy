import os
from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
import uunet.multinet as ml

from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import IntegrityConstraintManager
from Objects.CoAction.CoAction import CoAction
from Objects.Filter.Filter import Filter
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.ConversionManager.ConversionManager import ConversionManager
from utils.LogManager.LogManager import LogManager
from utils.common_variables import action_map, dtype
from utils.decorator_definition import log_method


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class NetworkManager:
    def __init__(
        self,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str | None,
        tw: Any,
        list_ca: Sequence[CoAction],
        dict_ca_filter: Mapping[str, Filter | None],
    ) -> None:
        """
            Create the manager that materializes graph files from filtered edge lists.
            :param dataset_name: [str] Dataset directory name.
            :param user_fraction: [float | None] User-selection fraction used in result-path construction.
            :param type_filter: [str | None] User-selection strategy used in result-path construction.
            :param tw: [TimeWindow] Time-window configuration used by the network pipeline.
            :param list_ca: [Sequence[CoAction]] Co-action layers to convert into graphs.
            :param dict_ca_filter: [Mapping[str, Filter | None]] Filter configuration by co-action id. NetworkManager
                expects filtered edge-list paths to exist for the configured filters.
            :return: None.
        """
        self.lm = LogManager('main')
        self.ch = Checkpoint()

        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.list_ca = list_ca
        self.dict_ca_filter = dict(dict_ca_filter)

        self.icm = IntegrityConstraintManager(file_name)
        self.dm = DirectoryManager(
            file_name,
            dataset_name,
            results=results,
            user_fraction=self.user_fraction,
            type_filter=self.type_filter,
            tw=tw,
            list_ca=list_ca,
            dict_ca_filter=dict_ca_filter,
        )
        self.dict_ca_filter = self.dm.dict_ca_filter
        self.cm = ConversionManager(dataset_name)

        self.icm.check_co_action(list_ca, self.dict_ca_filter)
        self.type_algorithm = self.dm.type_algorithm
        self.list_ca_str = '_'.join(list(self.dm.dict_path_ca.keys()))

    def _filter_label(self, type_ca: str) -> str:
        """
            Return a readable label for the filter applied to one co-action.
            :param type_ca: [str] Co-action id.
            :return: [str] Filter representation or None.
        """
        return repr(self.dict_ca_filter.get(type_ca))

    def _layer_name(self, type_ca: str) -> str:
        """
            Return the multiplex layer name for one co-action id.
            :param type_ca: [str] Co-action id.
            :return: [str] Layer name accepted by uunet.
        """
        return action_map[type_ca]

    def _required_path(self, dict_path: Mapping[str, str], path_key: str, type_ca: str) -> str:
        """
            Return a required path from a co-action path dictionary.
            :param dict_path: [Mapping[str, str]] Path dictionary created by DirectoryManager for one co-action.
            :param path_key: [str] Required path key.
            :param type_ca: [str] Co-action id used to produce a clear error message.
            :return: [str] Requested path.
        """
        if path_key not in dict_path:
            message = (
                f"{file_name}. Missing {path_key} for {type_ca}. "
                "NetworkManager expects a filtered network configuration. "
                "Run FilterGraphManager first or pass the filtered dict_ca_filter."
            )
            self.lm.printl(message)
            raise KeyError(message)
        return dict_path[path_key]

    def _list_files(self, directory_path: str, extension: str, context: str) -> list[str]:
        """
            List files with a required extension from a directory.
            :param directory_path: [str] Directory to inspect.
            :param extension: [str] File extension including the dot.
            :param context: [str] Human-readable context used in error messages.
            :return: [list[str]] Sorted matching filenames.
        """
        files = sorted(filename for filename in os.listdir(directory_path) if filename.endswith(extension))
        if len(files) == 0:
            message = f"{file_name}. No {extension} files found for {context} in {directory_path}."
            self.lm.printl(message)
            raise FileNotFoundError(message)
        return files

    def _edge_list_files(self, type_ca: str, dict_path: Mapping[str, str]) -> list[str]:
        """
            Return filtered edge-list files for one co-action.
            :param type_ca: [str] Co-action id.
            :param dict_path: [Mapping[str, str]] Path dictionary created by DirectoryManager for one co-action.
            :return: [list[str]] Sorted edge-list pickle filenames.
        """
        path_filter_edge_list = self._required_path(dict_path, "path_filter_edge_list", type_ca)
        return self._list_files(path_filter_edge_list, ".p", f"{type_ca} filtered edge lists")

    def _graph_files(self, type_ca: str, dict_path: Mapping[str, str]) -> list[str]:
        """
            Return graph pickle files for one co-action.
            :param type_ca: [str] Co-action id.
            :param dict_path: [Mapping[str, str]] Path dictionary created by DirectoryManager for one co-action.
            :return: [list[str]] Sorted graph pickle filenames.
        """
        path_filter_graph = self._required_path(dict_path, "path_filter_graph", type_ca)
        return self._list_files(path_filter_graph, ".p", f"{type_ca} graph files")

    def _load_edge_list(self, dict_path: Mapping[str, str], filename: str) -> list[tuple]:
        """
            Load one filtered edge-list pickle.
            :param dict_path: [Mapping[str, str]] Path dictionary created by DirectoryManager for one co-action.
            :param filename: [str] Edge-list filename.
            :return: [list[tuple]] Edge list.
        """
        return self.ch.load_object(dict_path["path_filter_edge_list"] + filename)

    def _load_graph(self, dict_path: Mapping[str, str], filename: str) -> nx.Graph:
        """
            Load one graph pickle.
            :param dict_path: [Mapping[str, str]] Path dictionary created by DirectoryManager for one co-action.
            :param filename: [str] Graph filename.
            :return: [nx.Graph] NetworkX graph.
        """
        return self.ch.load_object(dict_path["path_filter_graph"] + filename)

    def _save_graph(self, graph: nx.Graph, dict_path: Mapping[str, str], filename: str) -> None:
        """
            Save one NetworkX graph pickle.
            :param graph: [nx.Graph] Graph to save.
            :param dict_path: [Mapping[str, str]] Path dictionary created by DirectoryManager for one co-action.
            :param filename: [str] Output pickle filename.
            :return: None.
        """
        self.ch.save_object(graph, dict_path["path_filter_graph"] + filename)

    def _first_graph(self, type_ca: str, dict_path: Mapping[str, str]) -> tuple[str, nx.Graph]:
        """
            Load the first graph file for a co-action.
            :param type_ca: [str] Co-action id.
            :param dict_path: [Mapping[str, str]] Path dictionary created by DirectoryManager for one co-action.
            :return: [tuple[str, nx.Graph]] Filename and loaded graph.
        """
        graph_files = self._graph_files(type_ca, dict_path)
        filename = graph_files[0]
        return filename, self._load_graph(dict_path, filename)

    def _save_layer_edge_dataframe(self, graph: nx.Graph, layer: str) -> None:
        """
            Append one layer edge list to the multiplex edge-list dataframe.
            :param graph: [nx.Graph] Layer graph.
            :param layer: [str] Multiplex layer name.
            :return: None. The dataframe is appended under the multiplex edge-list directory.
        """
        edge_list_df = nx.to_pandas_edgelist(graph)
        edge_list_df['layer'] = layer
        self.ch.update_dataframe(edge_list_df, self.dm.path_multi_edge_list_df + "edge_list_df.csv", dtype=dtype)

    def _log_multiplex_summary(self, multiplex_graph: Any) -> None:
        """
            Write basic multiplex edge-attribute and layer-size information to the log.
            :param multiplex_graph: [Any] uunet multiplex graph.
            :return: None.
        """
        self.lm.printl(self.cm.to_df(ml.attributes(multiplex_graph, target="edge")))
        for type_ca in self.dm.dict_path_ca.keys():
            layer = self._layer_name(type_ca)
            edge_layer_df = self.cm.to_df(ml.edges(multiplex_graph, layers1=[layer]))
            self.lm.printl(f"{file_name}. layer={layer}, edges={edge_layer_df.shape[0]}.")

    @log_method
    def create_weighted_graph(self) -> None:
        """
            Convert each filtered edge list into a NetworkX weighted graph.
            :return: None. Graph pickle files are saved under each co-action filter graph directory.
        """
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            edge_list_files = self._edge_list_files(type_ca, dict_path)
            for filename in edge_list_files:
                edge_list = self._load_edge_list(dict_path, filename)
                graph = self.cm.from_edge_list_to_graph(edge_list)
                self._save_graph(graph, dict_path, filename)
                self.lm.printl(
                    f"{file_name}. Created graph for co-action={type_ca}, file={filename}, "
                    f"filter={self._filter_label(type_ca)}, nodes={graph.number_of_nodes()}, "
                    f"edges={graph.number_of_edges()}."
                )

    @log_method
    def create_weighted_multiplex_network(self, save_edge_list_df: bool = False) -> None:
        """
            Create a uunet multiplex network from the first graph file of each co-action layer.
            :param save_edge_list_df: [bool] If True, append the layer edge-list dataframe to the multiplex output
                directory. This can be very large.
            :return: None. The multiplex graph is saved as multiplex_graph.txt.
        """
        if self.tw.get_type_output_network() != "merged":
            message = (
                f"{file_name}. Multiplex creation expects merged networks, but "
                f"type_output_network={self.tw.get_type_output_network()}."
            )
            self.lm.printl(message)
            raise ValueError(message)

        self.lm.printl(f"{file_name}. Creating multiplex network for co-actions {self.list_ca_str}.")
        multiplex_graph = ml.empty()

        for type_ca, dict_path in self.dm.dict_path_ca.items():
            layer = self._layer_name(type_ca)
            filename, graph = self._first_graph(type_ca, dict_path)
            multiplex_graph = self.cm.add_layer_multiplex_network(multiplex_graph, graph, layer)
            if save_edge_list_df:
                self._save_layer_edge_dataframe(graph, layer)
            self.lm.printl(
                f"{file_name}. Added layer={layer} from graph={filename}, "
                f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}."
            )

        self.ch.save_multiplex_network(multiplex_graph, self.dm.path_multi_graph + "multiplex_graph.txt")
        self._log_multiplex_summary(multiplex_graph)

    @log_method
    def save_gephi_network(self) -> None:
        """
            Export each co-action graph to Gephi GEXF format.
            :return: None. GEXF files are saved under each co-action filter gephi_graph directory.
        """
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            filename, graph = self._first_graph(type_ca, dict_path)
            net_filename_no_ext = os.path.splitext(filename)[0]
            gephi_filename = f"{net_filename_no_ext}.gexf"
            gephi_path = self._required_path(dict_path, "path_filter_gephi_graph", type_ca) + gephi_filename
            self.cm.from_graph_to_gephi(graph, gephi_path)
            self.lm.printl(f"{file_name}. Saved Gephi graph for co-action={type_ca}: {gephi_filename}.")
