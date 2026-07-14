import os
from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
import pandas as pd

from DirectoryManager import DirectoryManager
from FilterGraphManager.FilterModels.BackboneNetwork.BackboneNetwork import BackboneNetwork
from MergeNetworkManager import MergeNetworkManager
from Objects.CoAction.CoAction import CoAction
from Objects.Filter.Filter import Filter
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.ConversionManager.ConversionManager import ConversionManager
from utils.LogManager.LogManager import LogManager
from utils.common_variables import NA_VAR, NODE1_VAR, NODE2_VAR, W_VAR, tuple_index
from utils.decorator_definition import log_method


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class FilterGraphManager:
    def __init__(
        self,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str | None,
        tw: Any,
        list_ca: Sequence[CoAction] | CoAction,
        dict_ca_filter: Mapping[str, Filter | None] | Filter | None,
    ) -> None:
        """
            Create the graph filtering manager.
            :param dataset_name: [str] Dataset directory name.
            :param user_fraction: [float | None] User-selection fraction used in path construction.
            :param type_filter: [str | None] User-selection strategy used in path construction.
            :param tw: [TimeWindow] Time-window configuration used to locate edge lists.
            :param list_ca: [Sequence[CoAction] | CoAction] Co-actions to filter. A single CoAction is accepted for
                backward compatibility.
            :param dict_ca_filter: [Mapping[str, Filter | None] | Filter | None] Filter configuration by co-action id.
                A single Filter is accepted for backward compatibility.
            :return: None.
        """
        self.lm = LogManager('main')
        self.ch = Checkpoint()
        self.cm = ConversionManager()

        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.list_ca = self._normalize_co_actions(list_ca)
        self.dict_ca_filter = self._normalize_filter_mapping(dict_ca_filter)

    def _normalize_co_actions(self, list_ca: Sequence[CoAction] | CoAction) -> list[CoAction]:
        """
            Normalize a single co-action or a sequence of co-actions into a list.
            :param list_ca: [Sequence[CoAction] | CoAction] Co-action input received by the constructor.
            :return: [list[CoAction]] Co-actions to process.
        """
        if isinstance(list_ca, CoAction):
            return [list_ca]
        return list(list_ca)

    def _normalize_filter_mapping(
        self,
        dict_ca_filter: Mapping[str, Filter | None] | Filter | None,
    ) -> dict[str, Filter | None]:
        """
            Normalize the filter input into a dictionary keyed by co-action id.
            :param dict_ca_filter: [Mapping[str, Filter | None] | Filter | None] Filter input received by the
                constructor.
            :return: [dict[str, Filter | None]] Filter configuration by co-action id.
        """
        if isinstance(dict_ca_filter, Mapping):
            return dict(dict_ca_filter)
        if len(self.list_ca) != 1:
            raise ValueError("A filter dictionary is required when filtering more than one co-action.")
        return {self.list_ca[0].get_co_action(): dict_ca_filter}

    def _build_directory_manager(self, ca: CoAction, filter_instance: Filter) -> DirectoryManager:
        """
            Build the directory context for one co-action and one filter.
            :param ca: [CoAction] Co-action being filtered.
            :param filter_instance: [Filter] Filter configuration for the co-action.
            :return: [DirectoryManager] Directory manager with resolved input and output paths.
        """
        return DirectoryManager(
            file_name,
            self.dataset_name,
            results=results,
            user_fraction=self.user_fraction,
            type_filter=self.type_filter,
            tw=self.tw,
            ca=ca,
            filter_instance=filter_instance,
        )

    def _input_edge_list_path(self, dm: DirectoryManager, filter_instance: Filter) -> str:
        """
            Return the edge-list directory used as input by the current filter.
            :param dm: [DirectoryManager] Directory manager for the current co-action.
            :param filter_instance: [Filter] Filter configuration for the current co-action.
            :return: [str] Directory containing the source edge-list pickle files.
        """
        if filter_instance.get_previous_filter() is None:
            return dm.path_edge_list
        return dm.path_previous_filter_edge_list

    def _list_edge_files(self, path_read: str) -> list[str]:
        """
            List edge-list pickle files from one directory.
            :param path_read: [str] Directory to inspect.
            :return: [list[str]] Sorted pickle filenames.
        """
        edge_files = sorted(filename for filename in os.listdir(path_read) if filename.endswith('.p'))
        if len(edge_files) == 0:
            message = f"No edge-list pickle files found in {path_read}."
            self.lm.printl(f"{file_name}. {message}")
            raise FileNotFoundError(message)
        return edge_files

    def _filter_edges_threshold(
        self,
        dm: DirectoryManager,
        filter_instance: Filter,
        tuple_element: str,
    ) -> None:
        """
            Filter edges by a threshold on one edge tuple field.
            :param dm: [DirectoryManager] Directory manager for the current co-action.
            :param filter_instance: [Filter] Filter configuration with threshold and previous-filter information.
            :param tuple_element: [str] Edge tuple field to threshold, for example w_ or nAction.
            :return: None. Filtered edge lists are saved in the filter output directory.
        """
        threshold = filter_instance.get_threshold()
        path_read = self._input_edge_list_path(dm, filter_instance)
        edge_list_files = self._list_edge_files(path_read)
        self.lm.printl(
            f"[FILTER][{dm.ca.get_co_action()}] threshold on {tuple_element}: {threshold}; "
            f"files={len(edge_list_files)}"
        )

        tuple_position = tuple_index[tuple_element]
        for filename in edge_list_files:
            edge_list = self.ch.load_object(path_read + filename)
            filtered_edge_list = [edge for edge in edge_list if edge[tuple_position] >= threshold]
            self.ch.save_object(filtered_edge_list, dm.path_filter_edge_list + filename)
            self.lm.printl(
                f"[FILTER][{dm.ca.get_co_action()}] {filename}: "
                f"{len(edge_list)} -> {len(filtered_edge_list)} edges"
            )

    def _filter_backbone(self, dm: DirectoryManager, filter_instance: Filter) -> None:
        """
            Filter edges with the disparity-filter backbone method.
            :param dm: [DirectoryManager] Directory manager for the current co-action.
            :param filter_instance: [Filter] Filter configuration. The threshold is interpreted as an alpha cutoff.
            :return: None. Filtered backbone edge lists are saved in the filter output directory.
        """
        threshold = filter_instance.get_threshold()
        path_read = self._input_edge_list_path(dm, filter_instance)
        edge_list_files = self._list_edge_files(path_read)
        backbone_network = BackboneNetwork()

        self.lm.printl(
            f"[FILTER][{dm.ca.get_co_action()}] backbone alpha threshold: {threshold}; "
            f"files={len(edge_list_files)}"
        )
        for filename in edge_list_files:
            edge_list = self.ch.load_object(path_read + filename)
            graph = nx.Graph()
            graph.add_weighted_edges_from(
                (edge[tuple_index[NODE1_VAR]], edge[tuple_index[NODE2_VAR]], edge[tuple_index[W_VAR]])
                for edge in edge_list
            )
            alpha_graph = backbone_network.disparity_filter(graph, weight='weight')
            self.ch.save_object(alpha_graph, dm.path_filter_processed + filename)

            filtered_edge_list = [
                (u, v, data['alpha'])
                for u, v, data in alpha_graph.edges(data=True)
                if data['alpha'] < threshold
            ]
            self.ch.save_object(filtered_edge_list, dm.path_filter_edge_list + filename)
            self.lm.printl(
                f"[FILTER][{dm.ca.get_co_action()}] {filename}: "
                f"{len(edge_list)} -> {len(filtered_edge_list)} backbone edges"
            )

    def _filter_node_top_edge(self, dm: DirectoryManager, filter_instance: Filter) -> None:
        """
            Keep strongest edges until the filtered graph reaches the configured maximum number of nodes.
            :param dm: [DirectoryManager] Directory manager for the current co-action.
            :param filter_instance: [Filter] Filter configuration. The threshold is the maximum number of nodes.
            :return: None. Filtered edge lists and dataframe copies are saved in the filter output directory.
        """
        max_nodes = filter_instance.get_threshold()
        path_read = self._input_edge_list_path(dm, filter_instance)
        edge_list_files = self._list_edge_files(path_read)
        self.lm.printl(
            f"[FILTER][{dm.ca.get_co_action()}] node_topEdge max_nodes: {max_nodes}; "
            f"files={len(edge_list_files)}"
        )

        for filename in edge_list_files:
            edge_list = self.ch.load_object(path_read + filename)
            filtered_edge_list = self._select_top_edges_by_node_limit(edge_list, max_nodes)
            self.ch.save_object(filtered_edge_list, dm.path_filter_edge_list + filename)
            self._save_edge_list_dataframe(filtered_edge_list, dm.path_filter_edge_list_df, filename)
            self.lm.printl(
                f"[FILTER][{dm.ca.get_co_action()}] {filename}: "
                f"{len(edge_list)} -> {len(filtered_edge_list)} edges"
            )

    def _select_top_edges_by_node_limit(self, edge_list: list[tuple], max_nodes: int) -> list[tuple]:
        """
            Select strongest edges while keeping the number of distinct nodes below the configured limit.
            :param edge_list: [list[tuple]] Source edge list.
            :param max_nodes: [int] Maximum number of nodes allowed in the filtered edge list.
            :return: [list[tuple]] Filtered edge list.
        """
        selected_edges = []
        selected_nodes = set()
        sorted_edge_list = sorted(edge_list, key=lambda edge: edge[tuple_index[W_VAR]], reverse=True)

        for edge in sorted_edge_list:
            user1 = edge[tuple_index[NODE1_VAR]]
            user2 = edge[tuple_index[NODE2_VAR]]
            both_new = user1 not in selected_nodes and user2 not in selected_nodes
            one_new = (user1 not in selected_nodes and user2 in selected_nodes) or (
                user1 in selected_nodes and user2 not in selected_nodes
            )
            both_present = user1 in selected_nodes and user2 in selected_nodes

            has_room_for_two = both_new and len(selected_nodes) <= max_nodes - 2
            has_room_for_one = one_new and len(selected_nodes) <= max_nodes - 1
            if has_room_for_two or has_room_for_one or both_present:
                selected_nodes.add(user1)
                selected_nodes.add(user2)
                selected_edges.append(edge)

        return selected_edges

    def _save_edge_list_dataframe(self, edge_list: list[tuple], output_path: str, filename: str) -> None:
        """
            Save a dataframe representation of one edge list.
            :param edge_list: [list[tuple]] Edge list to convert.
            :param output_path: [str] Directory where the dataframe must be saved.
            :param filename: [str] Original pickle filename used to derive the CSV filename.
            :return: None. The dataframe is saved as CSV.
        """
        if len(edge_list) == 0:
            columns = [NODE1_VAR, NODE2_VAR, W_VAR]
        else:
            columns = list(tuple_index.keys())[0:len(edge_list[0])]
        df = pd.DataFrame(edge_list, columns=columns)
        self.ch.save_dataframe(df, output_path + filename.split('.')[0] + '.csv')

    def _filter_one_co_action(self, ca: CoAction, filter_instance: Filter) -> None:
        """
            Apply one filter configuration to one co-action.
            :param ca: [CoAction] Co-action being filtered.
            :param filter_instance: [Filter] Filter configuration to apply.
            :return: None. Filtered edge lists are saved under the co-action filter directory.
        """
        dm = self._build_directory_manager(ca, filter_instance)
        filter_instance = dm.get_filter()
        type_graph_filter = filter_instance.get_type_filter()

        self.lm.printl(f"[FILTER][{ca.get_co_action()}] start {repr(filter_instance)}")
        if type_graph_filter in ["low_std", "mean", "median", "high_std", "th"]:
            self._filter_edges_threshold(dm, filter_instance, W_VAR)
        elif type_graph_filter == "backbone":
            self._filter_backbone(dm, filter_instance)
        elif type_graph_filter == "filter_merge_action":
            self._filter_edges_threshold(dm, filter_instance, NA_VAR)
            merge_manager = MergeNetworkManager(dm, self.dataset_name, self.user_fraction, self.type_filter, self.tw, ca)
            merge_manager.merge_edge_list(dm.path_filter_edge_list_temporal, dm.path_filter_edge_list)
        elif type_graph_filter == "merge_filter_action":
            self._filter_edges_threshold(dm, filter_instance, NA_VAR)
        elif type_graph_filter == "node_topEdge":
            self._filter_node_top_edge(dm, filter_instance)
        else:
            raise ValueError(f"Unsupported filter type: {type_graph_filter}")
        self.lm.printl(f"[FILTER][{ca.get_co_action()}] completed {repr(filter_instance)}")

    @log_method
    def filter_graph(self) -> None:
        """
            Filter all configured co-action edge lists.
            :return: None. Each filtered co-action is written into its own filter directory.
        """
        for ca in self.list_ca:
            co_action = ca.get_co_action()
            if co_action not in self.dict_ca_filter:
                raise KeyError(f"Missing filter configuration for co-action {co_action}.")

            filter_instance = self.dict_ca_filter[co_action]
            if filter_instance is None:
                self.lm.printl(f"[FILTER][{co_action}] skipped because filter is None.")
                continue

            self._filter_one_co_action(ca, filter_instance)
