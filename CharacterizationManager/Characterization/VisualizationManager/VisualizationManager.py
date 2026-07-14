import math
import os
from typing import Any, Sequence

import pandas as pd
import uunet.multinet as ml

from utils.Checkpoint.Checkpoint import Checkpoint
from utils.ConversionManager.ConversionManager import ConversionManager
from utils.LogManager.LogManager import LogManager
from utils.common_variables import dtype
from utils.decorator_definition import log_method


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class VisualizationManager:
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
            Create the visualization helper for multiplex and single-layer community outputs.
            :param list_ca: [Sequence[Any]] Co-action objects included in the characterization.
            :param dict_ca_filter: [dict[str, Any]] Filter configuration by co-action id.
            :param icm: [IntegrityConstraintManager] Constraint checker.
            :param dm: [DirectoryManager] Directory manager with visualization paths.
            :param type_algorithm: [str] Algorithm type detected by DirectoryManager.
            :param cda: [Any | None] Community-detection algorithm configuration.
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
        self.cda = cda

    def _find_grid_dimensions(self, n_layers: int) -> tuple[int, int]:
        """
            Return grid dimensions close to a square layout for multiplex layer plotting.
            :param n_layers: [int] Number of layers to plot.
            :return: [tuple[int, int]] Number of grid rows and columns.
        """
        n_rows = math.floor(math.sqrt(n_layers))
        n_cols = math.ceil(math.sqrt(n_layers))
        while n_rows * n_cols < n_layers:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1
        return n_rows, n_cols

    def _delete_unclustered_vertices(self, multiplex_graph: Any, community_df: pd.DataFrame) -> Any:
        """
            Delete vertices that do not belong to clustered actor-layer pairs.
            :param multiplex_graph: [Any] uunet multiplex graph.
            :param community_df: [pd.DataFrame] Community dataframe with actor and layer columns.
            :return: [Any] Multiplex graph after vertex deletion.
        """
        vertices_df = pd.DataFrame(ml.vertices(multiplex_graph))
        clustered_actor_df = community_df[['actor', 'layer']].copy()
        clustered_actor_df['actor'] = clustered_actor_df['actor'].astype(str)
        unclustered_df = pd.concat([clustered_actor_df, vertices_df]).reset_index(drop=True).drop_duplicates(
            subset=['actor', 'layer'],
            keep=False
        )
        ml.delete_vertices(multiplex_graph, unclustered_df.to_dict(orient='list'))
        return multiplex_graph

    def _first_file(self, directory_path: str, extension: str) -> str:
        """
            Return the first file in a directory with a given extension.
            :param directory_path: [str] Directory to scan.
            :param extension: [str] File extension including the dot.
            :return: [str] Filename with the requested extension.
        """
        files = [filename for filename in os.listdir(directory_path) if filename.endswith(extension)]
        if len(files) == 0:
            raise FileNotFoundError(f"No {extension} files in {directory_path}.")
        return files[0]

    def _plot_multiplex_graph(self, multiplex_graph: Any, communities: Any, output_path: str) -> None:
        """
            Plot a multiplex graph with the configured number of layers.
            :param multiplex_graph: [Any] uunet multiplex graph.
            :param communities: [Any] Community object accepted by uunet plotting.
            :param output_path: [str] Output PNG path.
            :return: None.
        """
        layout = ml.layout_multiforce(multiplex_graph)
        n_rows, n_cols = self._find_grid_dimensions(len(self.list_ca))
        ml.plot(
            multiplex_graph,
            com=communities,
            vertex_labels=[],
            layout=layout,
            grid=[n_rows, n_cols],
            vertex_size=[4],
            format='png',
            file=output_path
        )

    @log_method
    def visualize_multiplex_network(self) -> None:
        """
            Plot the configured multiplex network and its detected communities.
            :return: None. A PNG file is saved under DirectoryManager.path_community_visualization.
        """
        net_filename = self._first_file(self.dm.path_multi_graph, '.txt')
        net_filename_no_ext = net_filename.split('.')[0]
        multiplex_graph = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)
        communities = self.ch.load_object(self.dm.path_coms + 'coms.p')
        self._plot_multiplex_graph(
            multiplex_graph,
            communities,
            f'{self.dm.path_community_visualization}{net_filename_no_ext}.png'
        )

    @log_method
    def delete_edges_visualize_multiplex_network(self) -> None:
        """
            Plot a multiplex network after deleting actor-layer vertices outside the selected communities.
            :return: None. A PNG file is saved when the number of communities is small enough to visualize.
        """
        community_df_filename = self._first_file(self.dm.path_user_dataframe, '.csv')
        community_df = self.ch.read_dataframe(self.dm.path_user_dataframe + community_df_filename, dtype=dtype)

        if len(community_df['cid'].unique()) >= 10:
            self.lm.printl(f"{file_name}. The number of communities is too high to be visualized.")
            return

        net_filename = self._first_file(self.dm.path_multi_graph, '.txt')
        net_filename_no_ext = net_filename.split('.')[0]
        multiplex_graph = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)
        communities = self.ch.load_object(self.dm.path_coms + self._first_file(self.dm.path_coms, '.p'))
        multiplex_graph = self._delete_unclustered_vertices(multiplex_graph, community_df)
        self._plot_multiplex_graph(
            multiplex_graph,
            communities,
            f'{self.dm.path_community_visualization}{net_filename_no_ext}.png'
        )

    @log_method
    def delete_small_communities_single_layer(self, th_size: int) -> None:
        """
            Remove small single-layer communities from graph and gephi outputs.
            :param th_size: [int] Minimum community size to keep.
            :return: None. Filtered graph, gephi graph, and community dataframe artifacts are saved.
        """
        type_ca = self.list_ca[0].get_co_action()
        net_filename = self._first_file(self.dm.dict_path_ca[type_ca]['path_filter_graph'], '.p')
        net_filename_no_ext = net_filename.split('.')[0]
        graph = self.ch.load_object(self.dm.path_community_graph + net_filename)
        community_df = self.ch.read_dataframe(self.dm.path_user_dataframe + "com_df.csv", dtype=dtype)

        community_size_df = community_df.groupby(['group']).size().reset_index(name='nUsers')
        kept_community_df = pd.DataFrame(community_size_df[community_size_df['nUsers'] >= th_size]['group'])
        filtered_community_df = pd.merge(community_df, kept_community_df, on='group', how='inner')
        self.lm.printl(f"{file_name}. Number of communities: {len(kept_community_df['group'].unique())}")

        nodes_to_keep = set(filtered_community_df['userId'].values)
        nodes_to_remove = set(graph.nodes) - nodes_to_keep
        self.lm.printl(f"{file_name}. Number of nodes to be removed: {len(nodes_to_remove)}")
        graph.remove_nodes_from(nodes_to_remove)

        self.ch.save_dataframe(community_df, self.dm.path_user_dataframe + f"th_size_{str(th_size)}_com_df.csv")
        self.ch.save_object(graph, self.dm.path_community_graph + f"th_size_{str(th_size)}_{net_filename}")
        self.cm.from_graph_to_gephi(
            graph,
            self.dm.path_community_gephi_graph + f"th_size_{str(th_size)}_{net_filename_no_ext}.gexf"
        )
