import os
from typing import Any

import networkx as nx
import pandas as pd

from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import IntegrityConstraintManager
from Objects.CDAlgorithm.CDAlgorithm import CDAlgorithm
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.ConversionManager.ConversionManager import ConversionManager
from utils.LogManager.LogManager import LogManager
from utils.common_variables import custom_flatten_algorithm, flatten_algorithm
from utils.decorator_definition import log_method


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class CommunityDetectionManager:
    def __init__(
        self,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str | None,
        tw: Any,
        list_ca: list[Any],
        dict_ca_filter: dict[str, Any],
        cda: CDAlgorithm,
    ) -> None:
        """
            Initialize the community-detection manager.
            :param dataset_name: [str] Dataset identifier used to resolve result directories.
            :param user_fraction: [float | None] Selected-user fraction used in previous pipeline steps, or None when
                no user selection was applied.
            :param type_filter: [str | None] User-selection strategy used in previous pipeline steps, or None.
            :param tw: [Any] TimeWindow configuration used to select merged or temporal network outputs.
            :param list_ca: [list[Any]] Co-action objects that identify the layers or single layer to analyze.
            :param dict_ca_filter: [dict[str, Any]] Filter settings for each co-action.
            :param cda: [CDAlgorithm] Community-detection algorithm configuration.
            :return: None.
        """
        self.lm = LogManager("main")
        self.ch = Checkpoint()

        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter
        self.cda = cda

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
            cda=cda,
        )

        self.icm.check_co_action(list_ca, dict_ca_filter)
        self.icm.check_type_algorithm(tw, list_ca, cda.get_algorithm_name())

        self.type_algorithm = self.dm.get_type_algorithm()
        self.list_ca_str = "_".join(list(self.dm.dict_path_ca.keys()))
        self.cm = ConversionManager()

    def _first_file(self, directory_path: str, extension: str) -> str:
        """
            Return the first file with the requested extension from a directory.
            :param directory_path: [str] Directory where graph files are stored.
            :param extension: [str] Required file extension, for example ".p" or ".txt".
            :return: [str] Filename found in the directory.
        """
        graph_files = sorted(filename for filename in os.listdir(directory_path) if filename.endswith(extension))
        if len(graph_files) == 0:
            raise FileNotFoundError(f"No {extension} graph file found in {directory_path}.")
        return graph_files[0]

    def _normalize_multiplex_communities(self, df: pd.DataFrame) -> tuple[list[list[Any]], pd.DataFrame]:
        """
            Convert multiplex community assignments to the single-layer user/community format.
            :param df: [pd.DataFrame] Multiplex assignment dataframe with actor and cid columns.
            :return: [tuple[list[list[Any]], pd.DataFrame]] Communities as lists of users and a dataframe with
                userId/group columns.
        """
        df = df[["actor", "cid"]]
        df = df.rename(columns={"actor": "userId", "cid": "group"})
        df = df.drop_duplicates(keep="first")

        # Multiplex algorithms do not necessarily order community ids by size. Reindex the ids so that the largest
        # community gets group 0, matching the convention used by the single-layer algorithms.
        group_counts = df["group"].value_counts().sort_values(ascending=False)
        group_mapping = {group: idx for idx, group in enumerate(group_counts.index)}
        df["group"] = df["group"].map(group_mapping)

        communities = []
        for group in df["group"].unique():
            communities.append(df[df["group"] == group]["userId"].tolist())
        return communities, df

    def _communities_to_user_dataframe(self, communities: list[list[Any]]) -> pd.DataFrame:
        """
            Convert a list-of-lists community assignment into a user dataframe.
            :param communities: [list[list[Any]]] Communities where each inner list contains the users in one group.
            :return: [pd.DataFrame] Dataframe with userId and group columns.
        """
        unfolded_communities = {"userId": [], "group": []}
        for ind_com, community in enumerate(communities):
            for node in community:
                unfolded_communities["userId"].append(node)
                unfolded_communities["group"].append(ind_com)

        return pd.DataFrame(unfolded_communities)

    def _set_node_attribute_communities(self, graph: nx.Graph, communities: list[list[Any]]) -> nx.Graph:
        """
            Add the community id as a node attribute on a NetworkX graph.
            :param graph: [nx.Graph] Graph whose nodes must be annotated.
            :param communities: [list[list[Any]]] Communities where each inner list contains the users in one group.
            :return: [nx.Graph] Input graph with a group node attribute.
        """
        coms_dictionary = {}
        for ind_com, community in enumerate(communities):
            for node in community:
                coms_dictionary[node] = {"group": ind_com}

        nx.set_node_attributes(graph, coms_dictionary)
        return graph

    def _save_community_dataframe(self, com_df: pd.DataFrame) -> None:
        """
            Save the user-to-community dataframe.
            :param com_df: [pd.DataFrame] Dataframe with userId and group columns.
            :return: None.
        """
        self.ch.save_dataframe(com_df, self.dm.path_user_dataframe + "com_df.csv")

    def _save_annotated_graph(self, graph: nx.Graph, communities: list[list[Any]], net_filename_no_ext: str) -> None:
        """
            Save the graph annotated with community ids in pickle and Gephi formats.
            :param graph: [nx.Graph] NetworkX graph to annotate and persist.
            :param communities: [list[list[Any]]] Communities used to populate the node group attribute.
            :param net_filename_no_ext: [str] Graph filename without extension.
            :return: None.
        """
        annotated_graph = self._set_node_attribute_communities(graph, communities)
        self.ch.save_object(annotated_graph, self.dm.path_community_graph + net_filename_no_ext + ".p")
        self.cm.from_graph_to_gephi(
            annotated_graph,
            self.dm.path_community_gephi_graph + net_filename_no_ext + ".gexf",
        )

    def _compute_one_layer_community_detection(self) -> None:
        """
            Compute and save community detection for a single filtered graph.
            :return: None.
        """
        type_ca = self.list_ca[0].get_co_action()
        graph_path = self.dm.dict_path_ca[type_ca]["path_filter_graph"]
        net_filename = self._first_file(graph_path, ".p")
        net_filename_no_ext = os.path.splitext(net_filename)[0]

        graph = self.ch.load_object(graph_path + net_filename)
        coms = self.cda.compute_communities(graph)
        communities = coms.communities

        self.ch.save_object(coms, self.dm.path_coms + net_filename)
        self._save_community_dataframe(self._communities_to_user_dataframe(communities))
        self._save_annotated_graph(graph, communities, net_filename_no_ext)

    def _compute_flattened_multiplex_community_detection(
        self,
        multiplex_graph: Any,
        net_filename_no_ext: str,
    ) -> tuple[Any, list[list[Any]], pd.DataFrame]:
        """
            Flatten a multiplex graph and compute communities on the flattened NetworkX graph.
            :param multiplex_graph: [Any] uunet multiplex graph.
            :param net_filename_no_ext: [str] Original multiplex filename without extension.
            :return: [tuple[Any, list[list[Any]], pd.DataFrame]] Raw community object, normalized communities, and
                user/community dataframe.
        """
        graph = self.cda.flatten_multiplex_network(multiplex_graph)
        self.ch.save_object(graph, self.dm.path_community_graph + net_filename_no_ext + ".p")

        if self.cda.get_algorithm_name() in custom_flatten_algorithm:
            coms = self.cda.compute_communities(graph)
            communities = coms.communities
            com_df = self._communities_to_user_dataframe(communities)
        else:
            coms = self.cda.compute_communities(multiplex_graph)
            communities, com_df = self._normalize_multiplex_communities(self.cm.to_df(coms))

        self._save_annotated_graph(graph, communities, net_filename_no_ext)
        return coms, communities, com_df

    def _compute_native_multiplex_community_detection(self, multiplex_graph: Any) -> tuple[Any, pd.DataFrame]:
        """
            Compute communities with a native multiplex algorithm.
            :param multiplex_graph: [Any] uunet multiplex graph.
            :return: [tuple[Any, pd.DataFrame]] Raw community object and dataframe returned by the conversion manager.
        """
        coms = self.cda.compute_communities(multiplex_graph)
        com_df = self.cm.to_df(coms)
        return coms, com_df

    def _compute_multi_layer_community_detection(self) -> None:
        """
            Compute and save community detection for a merged multiplex graph.
            :return: None.
        """
        net_filename = self._first_file(self.dm.path_multi_graph, ".txt")
        net_filename_no_ext = os.path.splitext(net_filename)[0]
        multiplex_graph = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)

        if self.cda.get_algorithm_name() in flatten_algorithm:
            coms, _, com_df = self._compute_flattened_multiplex_community_detection(
                multiplex_graph,
                net_filename_no_ext,
            )
        else:
            coms, com_df = self._compute_native_multiplex_community_detection(multiplex_graph)

        self.ch.save_object(coms, self.dm.path_coms + "coms.p")
        self._save_community_dataframe(com_df)

    @log_method
    def compute_community_detection(self) -> None:
        """
            Run the configured community-detection algorithm and save its outputs.
            :return: None.
        """
        self.lm.printl(f"{file_name}. algorithm: {str(self.cda)}")
        if self.tw.get_type_output_network() == "temporal":
            self.lm.printl(f"{file_name}. temporal community detection is not implemented yet.")
            return

        if self.type_algorithm == "one-layer":
            self._compute_one_layer_community_detection()
        elif self.type_algorithm == "multi-layer":
            self._compute_multi_layer_community_detection()
        else:
            raise ValueError(f"Unknown community-detection type: {self.type_algorithm}.")
