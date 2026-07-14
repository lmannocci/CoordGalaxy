import os
from typing import Any

import numpy as np

from Objects.CoAction.CoAction import CoAction
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.LogManager.LogManager import LogManager
from utils.common_variables import (
    co_action_abbreviation_map,
    level,
    normalize_co_action_id,
    similarity_function_map,
)
from utils.pipeline_io import create_directory


class DirectoryManager:
    """
    Centralized path builder for framework data, network, community, and comparison artifacts.

    DirectoryManager is intentionally stateful: each caller receives an instance with
    the path attributes that match its framework level. The class does not own the
    business logic of the modules; it only translates dataset/configuration objects
    into stable filesystem locations and creates output directories when that level
    is responsible for writing new artifacts.
    """

    def __init__(
        self,
        file_name: str,
        dataset_name: str,
        data_path: str | None = None,
        results: str | None = None,
        user_fraction: float | None = None,
        type_filter: str | None = None,
        tw: Any | None = None,
        ca: Any | None = None,
        filter_instance: Any | None = None,
        list_ca: list[Any] | None = None,
        dict_ca_filter: dict[str, Any] | None = None,
        cda: Any | None = None,
    ) -> None:
        """
        Create and expose the framework paths needed by a manager module.

        The constructor receives the name of the caller module and uses the `level`
        mapping in `utils.common_variables` to decide which path groups are needed.
        Lower levels create input and intermediate paths; higher levels reuse those
        roots and add network, community-detection, characterization, and community
        comparison output directories.

        :param file_name: [str] Name of the manager requesting paths, for example
            InputManager, FilterGraphManager, CharacterizationManager, or
            CommunityComparisonManager.
        :param dataset_name: [str] Dataset identifier used under data and results,
            for example moltbook or uk.
        :param data_path: [str | None] Root data directory. The dataset directory is
            built as data_path/dataset_name/.
        :param results: [str | None] Root results directory. The dataset results
            directory is built as results/dataset_name/.
        :param user_fraction: [float | None] Optional user-selection threshold or
            fraction used in the result path.
        :param type_filter: [str | None] Optional user-selection strategy used in the
            result path.
        :param tw: [Any | None] TimeWindow object used to build temporal and merged
            network paths.
        :param ca: [Any | None] CoAction object used by single co-action modules.
        :param filter_instance: [Any | None] Filter object used by FilterGraphManager.
        :param list_ca: [list[Any] | None] CoAction objects used to build one-layer or
            multi-layer paths.
        :param dict_ca_filter: [dict[str, Any] | None] Mapping from co-action id to
            the Filter object applied to that co-action. Keys are normalized to
            canonical co-action ids.
        :param cda: [Any | None] Community-detection algorithm object.
        :return: None. Path attributes are attached to the instance.
        """
        self.lm = LogManager("main")
        self.ch = Checkpoint()
        self.file_name = file_name
        self.module_level = self._get_module_level(file_name)

        self.dataset_name = dataset_name
        self.data_path = data_path
        self.results = results
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.ca = ca
        self.filter_instance = filter_instance
        self.list_ca = list_ca or []
        self.dict_ca_filter = self._normalize_co_action_dict(dict_ca_filter)
        self.cda = cda

        self._configure_paths_for_level()

    def _configure_paths_for_level(self) -> None:
        """
        Dispatch path creation according to the caller module level.

        :return: None. The relevant path attributes are initialized on the instance.
        """
        if self.module_level == -1:
            self._configure_input_level()
            return

        if self.module_level == 0:
            self._configure_user_selection_level()
            return

        if self.module_level == 1:
            self._configure_similarity_level()
            return

        if self.module_level == 2:
            self._configure_filter_graph_level()
            return

        if self.module_level >= 3:
            self._configure_network_level()

        if self.module_level >= 4 and self.cda is not None:
            self._configure_community_detection_level()

        if self.module_level >= 5:
            self._configure_characterization_level()

        if self.module_level == 6:
            self._configure_community_comparison_level()

    def _configure_input_level(self) -> None:
        """
        Configure paths used by InputManager.

        :return: None. Creates dataset, temp_data, co_action_data, and analysis
            directories under the dataset data path.
        """
        self._set_dataset_paths(include_analysis=True)
        self._ensure_directories(
            self.path_dataset,
            self.path_temp_data,
            self.path_co_action_data,
            self.path_data_analysis,
        )

    def _configure_user_selection_level(self) -> None:
        """
        Configure paths used by SelectionUserManager.

        :return: None. Exposes the base dataset input directories and result path
            attributes without creating result directories during exploratory user-selection analysis.
        """
        self._set_dataset_paths(include_analysis=True)
        self._ensure_directory(self.path_data_analysis)
        self._set_result_dataset_path(create=False)
        self._set_user_selection_result_path(create=False)

    def _configure_similarity_level(self) -> None:
        """
        Configure paths used by SimilarityFunctionManager.

        :return: None. Creates paths for one co-action, its similarity function,
            edge lists, processed files, temporal edge lists, and edge-list metadata.
        """
        self._set_dataset_paths()
        self._set_result_dataset_path(create=True)
        self._set_user_selection_result_path(create=True)
        self._set_network_time_window_paths(create=True)
        self._set_single_co_action_paths(create=True)

    def _configure_filter_graph_level(self) -> None:
        """
        Configure paths used by FilterGraphManager.

        :return: None. Creates the same paths as SimilarityFunctionManager and then
            adds the current filter path, previous-filter read paths, graph paths,
            gephi paths, analysis paths, and community paths.
        """
        self._configure_similarity_level()
        self._set_filter_paths()

    def _configure_network_level(self) -> None:
        """
        Configure paths used by NetworkManager and higher-level modules.

        :return: None. Exposes per-co-action paths in `dict_path_ca` and creates the
            multi-layer output directories when more than one co-action is selected.
        """
        self._set_dataset_paths(include_analysis=True, include_temp=True)
        self._set_result_dataset_path(create=False)
        self.type_algorithm = self._get_type_algorithm(self.list_ca)
        self.dict_path_ca: dict[str, dict[str, str]] = {}

        self._set_user_selection_result_path(create=False)
        self._set_network_time_window_paths(create=False)
        self._set_network_co_action_dictionary()

        if self.type_algorithm == "multi-layer":
            self._set_multi_layer_paths()
        elif self.type_algorithm == "one-layer":
            one_layer_paths = list(self.dict_path_ca.values())[0]
            if "path_filter_analysis" in one_layer_paths:
                self.path_analysis = one_layer_paths["path_filter_analysis"]
            else:
                self.path_analysis = one_layer_paths["path_NF_analysis"]

            if self.cda is None:
                return

            if "path_filter_community" not in one_layer_paths:
                message = (
                    "Community-detection paths for a one-layer network require a filtered network path. "
                    "Run community detection on a filtered co-action, or add explicit no-filter graph/community support."
                )
                self.lm.printl(message)
                raise KeyError(message)
            self.path_community = one_layer_paths["path_filter_community"]

    def _configure_community_detection_level(self) -> None:
        """
        Configure paths used by CommunityDetectionManager and higher-level modules.

        :return: None. Creates the community-detection algorithm directory and its
            subdirectories for communities, graphs, user dataframes, visualizations,
            and analysis outputs.
        """
        self.path_algorithm = self._get_path_algorithm(self.path_community, self.cda)
        self._ensure_directory(self.path_algorithm)

        self.path_coms = f"{self.path_algorithm}coms{os.sep}"
        self.path_community_graph = f"{self.path_algorithm}graph{os.sep}"
        self.path_community_gephi_graph = f"{self.path_algorithm}gephi_graph{os.sep}"
        self.path_user_dataframe = f"{self.path_algorithm}user_dataframe{os.sep}"
        self.path_community_visualization = f"{self.path_algorithm}visualization{os.sep}"
        self.path_community_analysis = f"{self.path_algorithm}analysis{os.sep}"
        self._ensure_directories(
            self.path_coms,
            self.path_community_graph,
            self.path_community_gephi_graph,
            self.path_user_dataframe,
            self.path_community_visualization,
            self.path_community_analysis,
        )

    def _configure_characterization_level(self) -> None:
        """
        Configure paths used only by CharacterizationManager.

        Characterization currently reuses the network and community-detection paths,
        so this method is intentionally a no-op placeholder for future
        characterization-specific directories.

        :return: None.
        """
        return None

    def _configure_community_comparison_level(self) -> None:
        """
        Configure paths used by CommunityComparisonManager.

        :return: None. Creates comparison output directories for overlap heatmaps,
            stacked-flux plots, dimensionality-reduction plots, NMI, gained/lost node
            metrics, validation, and cosine-similarity outputs.
        """
        self.path_overlapping_heatmap = f"{self.path_overlapping_analysis}heatmap{os.sep}"
        self.path_overlapping_stacked_plot = f"{self.path_overlapping_analysis}stacked_plot{os.sep}"
        self.path_overlapping_flux_df = f"{self.path_overlapping_stacked_plot}flux_df{os.sep}"
        self.path_overlapping_t_sne_plot = f"{self.path_overlapping_analysis}t_sne_plot{os.sep}"
        self.path_overlapping_umap_plot = f"{self.path_overlapping_analysis}umap_plot{os.sep}"
        self.path_overlapping_starplot = f"{self.path_overlapping_analysis}starplot{os.sep}"
        self.path_overlapping_pca_plot = f"{self.path_overlapping_analysis}pca_plot{os.sep}"
        self.path_overlapping_NMI = f"{self.path_overlapping_analysis}NMI{os.sep}"
        self.path_overlapping_node_metrics_gained_lost = (
            f"{self.path_overlapping_analysis}node_metrics_gained_lost{os.sep}"
        )
        self.path_overlapping_KDE_plot = f"{self.path_overlapping_node_metrics_gained_lost}KDE_plot{os.sep}"
        self.path_overlapping_distribution_plot = (
            f"{self.path_overlapping_node_metrics_gained_lost}distribution_plot{os.sep}"
        )
        self.path_node_metrics_boxplot = f"{self.path_overlapping_analysis}node_metrics_boxplot{os.sep}"
        self.path_validation = f"{self.path_overlapping_analysis}validation{os.sep}"
        self.path_cosine_similarity = f"{self.path_overlapping_analysis}cosine_similarity{os.sep}"
        self._ensure_directories(
            self.path_overlapping_heatmap,
            self.path_overlapping_stacked_plot,
            self.path_overlapping_flux_df,
            self.path_overlapping_t_sne_plot,
            self.path_overlapping_umap_plot,
            self.path_overlapping_starplot,
            self.path_overlapping_pca_plot,
            self.path_overlapping_NMI,
            self.path_overlapping_node_metrics_gained_lost,
            self.path_overlapping_KDE_plot,
            self.path_overlapping_distribution_plot,
            self.path_node_metrics_boxplot,
            self.path_validation,
            self.path_cosine_similarity,
        )

    def _set_dataset_paths(self, include_analysis: bool = False, include_temp: bool = False) -> None:
        """
        Build common dataset paths under the data root.

        :param include_analysis: [bool] If True, expose `path_data_analysis`.
        :param include_temp: [bool] If True, expose legacy `path_temp`.
        :return: None. Dataset path attributes are initialized.
        """
        self.path_dataset = f"{self.data_path}{self.dataset_name}{os.sep}"
        self.path_temp_data = f"{self.path_dataset}temp_data{os.sep}"
        self.path_co_action_data = f"{self.path_dataset}co_action_data{os.sep}"
        if include_analysis:
            self.path_data_analysis = f"{self.path_dataset}analysis{os.sep}"
        if include_temp:
            self.path_temp = f"{self.path_dataset}temp{os.sep}"

    def _set_result_dataset_path(self, create: bool) -> None:
        """
        Build the dataset result root.

        :param create: [bool] If True, create the directory.
        :return: None. `result_dataset` is initialized.
        """
        self.result_dataset = f"{self.results}{self.dataset_name}{os.sep}"
        if create:
            self._ensure_directory(self.result_dataset)

    def _set_user_selection_result_path(self, create: bool) -> None:
        """
        Build the result path associated with the user-selection configuration.

        :param create: [bool] If True, create the directory.
        :return: None. `path_type_filter` is initialized.
        """
        self.path_type_filter = f"{self.result_dataset}{self.type_filter}_{self.user_fraction}{os.sep}"
        if create:
            self._ensure_directory(self.path_type_filter)

    def _set_network_time_window_paths(self, create: bool) -> None:
        """
        Build the network root associated with the output-network and time-window settings.

        :param create: [bool] If True, create the directories.
        :return: None. Network, time-window, and root path attributes are initialized.
        """
        self.network_result = self._get_network_result_path()
        self.path_type_time_window = f"{self.network_result}{self.tw.get_type_time_window()}{os.sep}"
        self.path_tw = self._get_window_path(self.path_type_time_window)
        self.path_root = self._get_network_root_path()

        if create:
            self._ensure_directories(self.network_result, self.path_type_time_window, self.path_tw)
            if self.tw.get_type_output_network() == "merged":
                self._ensure_directory(self.path_root)

        self.path_info_tw = f"{self.path_tw}info_tw{os.sep}"
        if create:
            self._ensure_directory(self.path_info_tw)

    def _set_single_co_action_paths(self, create: bool) -> None:
        """
        Build paths for the single co-action being processed by level 1 or level 2 modules.

        :param create: [bool] If True, create the co-action directories.
        :return: None. Single co-action path attributes are initialized.
        """
        self.path_ca, self.path_ca_sf = self._get_co_action_path(self.ca, self.path_root)
        self.path_edge_list = f"{self.path_ca_sf}edge_list{os.sep}"
        self.path_processed = f"{self.path_ca_sf}processed{os.sep}"
        self.path_edge_list_temporal = f"{self.path_edge_list}temporal{os.sep}"
        self.path_NF_analysis = f"{self.path_ca_sf}analysis{os.sep}"
        self.path_info_edge_list = f"{self.path_ca_sf}info_edge_list{os.sep}"
        self.path_info_edge_list_temporal = f"{self.path_info_edge_list}temporal{os.sep}"

        _, self.path_ca_overlapping = self._get_co_action_path(
            CoAction(self.ca.get_co_action(), "overlapping_coefficient"),
            self.path_root,
        )
        self.path_overlapping_info_edge_list = f"{self.path_ca_overlapping}info_edge_list{os.sep}"
        self.path_overlapping_info_edge_list_temporal = f"{self.path_overlapping_info_edge_list}temporal{os.sep}"

        if create:
            self._ensure_directories(
                self.path_ca,
                self.path_ca_sf,
                self.path_edge_list,
                self.path_processed,
                self.path_edge_list_temporal,
                self.path_NF_analysis,
                self.path_info_edge_list,
                self.path_info_edge_list_temporal,
            )

    def _set_filter_paths(self) -> None:
        """
        Build paths for the current graph filter and, when needed, its previous filter.

        :return: None. Filter path attributes are initialized and the selected filter
            threshold is updated when the filter type is based on previous edge weights.
        """
        self._resolve_dynamic_filter_threshold(self.path_ca_sf, self.filter_instance)
        self.path_previous_filter, self.path_filter = self._get_path_previous_and_filter(
            self.path_ca_sf,
            self.filter_instance,
        )

        if self.filter_instance.get_previous_filter() is not None:
            self._set_previous_filter_paths()

        self._set_current_filter_paths()

    def _set_previous_filter_paths(self) -> None:
        """
        Build read-only paths for a previous filter in a chained filtering pipeline.

        :return: None. Previous-filter path attributes are initialized.
        """
        self.path_previous_filter_edge_list = f"{self.path_previous_filter}edge_list{os.sep}"
        self.path_previous_filter_edge_list_temporal = f"{self.path_previous_filter_edge_list}temporal{os.sep}"
        self.path_previous_filter_info_edge_list = f"{self.path_previous_filter}info_edge_list{os.sep}"
        self.path_previous_filter_info_edge_list_temporal = (
            f"{self.path_previous_filter_info_edge_list}temporal{os.sep}"
        )
        self.path_previous_filter_processed = f"{self.path_previous_filter}processed{os.sep}"
        self.path_previous_filter_edge_list_df = f"{self.path_previous_filter}edge_list_df{os.sep}"
        self.path_previous_filter_graph = f"{self.path_previous_filter}graph{os.sep}"
        self.path_previous_gephi_graph = f"{self.path_previous_filter}gephi_graph{os.sep}"
        self.path_previous_analysis = f"{self.path_previous_filter}analysis{os.sep}"
        self.path_previous_community = f"{self.path_previous_filter}community{os.sep}"

    def _set_current_filter_paths(self) -> None:
        """
        Build and create output paths for the current filter.

        :return: None. Current-filter path attributes are initialized and created.
        """
        self._ensure_directory(self.path_filter)
        self.path_filter_edge_list = f"{self.path_filter}edge_list{os.sep}"
        self.path_filter_edge_list_temporal = f"{self.path_filter_edge_list}temporal{os.sep}"
        self.path_filter_info_edge_list = f"{self.path_filter}info_edge_list{os.sep}"
        self.path_filter_info_edge_list_temporal = f"{self.path_filter_info_edge_list}temporal{os.sep}"
        self.path_filter_processed = f"{self.path_filter}processed{os.sep}"
        self.path_filter_edge_list_df = f"{self.path_filter}edge_list_df{os.sep}"
        self.path_filter_graph = f"{self.path_filter}graph{os.sep}"
        self.path_gephi_graph = f"{self.path_filter}gephi_graph{os.sep}"
        self.path_analysis = f"{self.path_filter}analysis{os.sep}"
        self.path_community = f"{self.path_filter}community{os.sep}"
        self._ensure_directories(
            self.path_filter_edge_list,
            self.path_filter_edge_list_temporal,
            self.path_filter_info_edge_list,
            self.path_filter_info_edge_list_temporal,
            self.path_filter_processed,
            self.path_filter_edge_list_df,
            self.path_filter_graph,
            self.path_gephi_graph,
            self.path_analysis,
            self.path_community,
        )

    def _set_network_co_action_dictionary(self) -> None:
        """
        Build per-co-action paths used by network, community, and characterization modules.

        :return: None. `dict_path_ca` is populated with not-filtered, filtered, and
            overlapping-info paths for each selected co-action.
        """
        for ca in self.list_ca:
            type_ca = ca.get_co_action()
            filter_instance = self.dict_ca_filter[type_ca]
            path_ca, path_ca_sf = self._get_co_action_path(ca, self.path_root)

            self.dict_path_ca[type_ca] = self._build_not_filtered_co_action_paths(path_ca_sf)

            self._resolve_dynamic_filter_threshold(path_ca_sf, filter_instance)
            _, path_filter = self._get_path_previous_and_filter(path_ca_sf, filter_instance)
            if path_filter is not None:
                self.dict_path_ca[type_ca].update(self._build_filtered_co_action_paths(path_filter))

            _, path_ca_overlapping = self._get_co_action_path(
                CoAction(ca.get_co_action(), "overlapping_coefficient"),
                self.path_root,
            )
            self.dict_path_ca[type_ca].update(self._build_overlapping_info_paths(path_ca_overlapping))

    def _build_not_filtered_co_action_paths(self, path_ca_sf: str) -> dict[str, str]:
        """
        Build paths for unfiltered edge-list artifacts of a co-action.

        :param path_ca_sf: [str] Co-action/similarity-function directory.
        :return: [dict[str, str]] Path mapping for unfiltered co-action artifacts.
        """
        path_edge_list = f"{path_ca_sf}edge_list{os.sep}"
        path_info_edge_list = f"{path_ca_sf}info_edge_list{os.sep}"
        return {
            "path_NF_edge_list": path_edge_list,
            "path_NF_edge_list_temporal": f"{path_edge_list}temporal{os.sep}",
            "path_NF_processed": f"{path_ca_sf}processed{os.sep}",
            "path_NF_analysis": f"{path_ca_sf}analysis{os.sep}",
            "path_NF_info_edge_list": path_info_edge_list,
            "path_NF_info_edge_list_temporal": f"{path_info_edge_list}temporal{os.sep}",
        }

    def _build_filtered_co_action_paths(self, path_filter: str) -> dict[str, str]:
        """
        Build paths for filtered graph artifacts of a co-action.

        :param path_filter: [str] Directory for the filter instance.
        :return: [dict[str, str]] Path mapping for filtered graph artifacts.
        """
        return {
            "path_filter": path_filter,
            "path_filter_graph": f"{path_filter}graph{os.sep}",
            "path_filter_gephi_graph": f"{path_filter}gephi_graph{os.sep}",
            "path_filter_edge_list": f"{path_filter}edge_list{os.sep}",
            "path_filter_processed": f"{path_filter}processed{os.sep}",
            "path_filter_edge_list_df": f"{path_filter}edge_list_df{os.sep}",
            "path_filter_analysis": f"{path_filter}analysis{os.sep}",
            "path_filter_community": f"{path_filter}community{os.sep}",
        }

    def _build_overlapping_info_paths(self, path_ca_overlapping: str) -> dict[str, str]:
        """
        Build paths to overlapping-coefficient edge metadata for a co-action.

        These paths are needed by threshold filters even when the current similarity
        function is not overlapping_coefficient.

        :param path_ca_overlapping: [str] Co-action path using overlapping_coefficient.
        :return: [dict[str, str]] Path mapping for overlapping edge metadata.
        """
        path_overlapping_info_edge_list = f"{path_ca_overlapping}info_edge_list{os.sep}"
        return {
            "path_ca_overlapping": path_ca_overlapping,
            "path_overlapping_info_edge_list": path_overlapping_info_edge_list,
            "path_overlapping_info_edge_list_temporal": f"{path_overlapping_info_edge_list}temporal{os.sep}",
        }

    def _set_multi_layer_paths(self) -> None:
        """
        Build and create multi-layer network output paths.

        :return: None. Multi-layer path attributes are initialized and created.
        """
        self.path_multi_co_action = f"{self.path_root}multi_co_action{os.sep}"
        self._ensure_directory(self.path_multi_co_action)

        self.path_multi_co_action_instance = self._get_path_multi_co_action(self.list_ca, self.dict_ca_filter)
        self.path_multi_graph = f"{self.path_multi_co_action_instance}graph{os.sep}"
        self.path_multi_edge_list_df = f"{self.path_multi_co_action_instance}edge_list_df{os.sep}"
        self.path_analysis = f"{self.path_multi_co_action_instance}analysis{os.sep}"
        self.path_processed = f"{self.path_multi_co_action_instance}processed{os.sep}"
        self.path_community = f"{self.path_multi_co_action_instance}community{os.sep}"
        self.path_visualization = f"{self.path_multi_co_action_instance}visualization{os.sep}"
        self.path_overlapping_analysis = f"{self.path_multi_co_action_instance}overlapping_analysis{os.sep}"
        self.path_latex = f"{self.path_multi_co_action_instance}latex{os.sep}"
        self._ensure_directories(
            self.path_multi_co_action_instance,
            self.path_multi_graph,
            self.path_multi_edge_list_df,
            self.path_analysis,
            self.path_processed,
            self.path_community,
            self.path_visualization,
            self.path_overlapping_analysis,
            self.path_latex,
        )

    def _normalize_co_action_dict(self, co_action_dict: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Return a co-action keyed dictionary using canonical framework co-action ids.

        :param co_action_dict: [dict[str, Any] | None] Dictionary keyed by co-action id,
            layer name, or alias.
        :return: [dict[str, Any] | None] Dictionary keyed by canonical co-action ids.
        """
        if co_action_dict is None:
            return None
        return {
            normalize_co_action_id(co_action): value
            for co_action, value in co_action_dict.items()
        }

    def get_co_action_data_path(
        self,
        user_fraction: float | None = None,
        type_filter: str | None = None,
        create: bool = False,
    ) -> str:
        """
        Return the co-action input directory for the current user-selection configuration.

        :param user_fraction: [float | None] Fraction used for user selection. If None,
            the base co_action_data directory is returned.
        :param type_filter: [str | None] User-selection strategy name. If None, the
            base co_action_data directory is returned.
        :param create: [bool] If True, create the selected co-action directory when needed.
        :return: [str] Path to the co-action data directory.
        """
        if user_fraction is None or type_filter is None:
            return self.path_co_action_data

        path = f"{self.path_dataset}co_action_data_th_{user_fraction}_{type_filter}{os.sep}"
        if create:
            self._ensure_directory(path)
        return path

    def _get_module_level(self, file_name: str) -> int:
        """
        Return the configured framework level for a manager module.

        :param file_name: [str] Manager module name.
        :return: [int] Level value from `utils.common_variables.level`.
        """
        if file_name not in level:
            message = f"DirectoryManager cannot find a framework level for module '{file_name}'."
            self.lm.printl(message)
            raise KeyError(message)
        return level[file_name]

    def _get_network_result_path(self) -> str:
        """
        Return the result directory for merged or temporal network outputs.

        :return: [str] Network result directory.
        """
        output_network = self.tw.get_type_output_network()
        if output_network == "merged":
            return f"{self.path_type_filter}merged_network{os.sep}"
        if output_network == "temporal":
            return f"{self.path_type_filter}temporal_network{os.sep}"
        message = f"Unsupported output network type: {output_network}"
        self.lm.printl(message)
        raise ValueError(message)

    def _get_network_root_path(self) -> str:
        """
        Return the root path below the time-window directory.

        :return: [str] The merge-specific root for merged networks, or the time-window
            path for temporal networks.
        """
        output_network = self.tw.get_type_output_network()
        if output_network == "merged":
            return f"{self.path_tw}{self.tw.get_type_merge()}{os.sep}"
        if output_network == "temporal":
            return self.path_tw
        message = f"Unsupported output network type: {output_network}"
        self.lm.printl(message)
        raise ValueError(message)

    def _get_window_path(self, path_type_time_window: str) -> str:
        """
        Build the path segment associated with the configured time-window strategy.

        :param path_type_time_window: [str] Path ending at the time-window type, for
            example .../OTW/.
        :return: [str] Window-specific path. ATW uses only tw_str, OTW uses tw_str and
            tw_slide_interval_str, and ANY keeps the type-time-window path.
        """
        type_time_window = self.tw.get_type_time_window()
        if type_time_window == "ATW":
            return f"{path_type_time_window}tw_{self.tw.get_tw_str()}{os.sep}"
        if type_time_window == "OTW":
            return (
                f"{path_type_time_window}tw_{self.tw.get_tw_str()}"
                f"-tw_slide_interval_{self.tw.get_tw_slide_interval_str()}{os.sep}"
            )
        if type_time_window == "ANY":
            return f"{path_type_time_window}{os.sep}"
        message = f"Unsupported time-window type: {type_time_window}"
        self.lm.printl(message)
        raise ValueError(message)

    def _get_co_action_path(self, ca: Any, path_tw: str) -> tuple[str, str]:
        """
        Build co-action and similarity-function paths.

        :param ca: [Any] CoAction object exposing get_co_action and get_similarity_function.
        :param path_tw: [str] Root path below the time-window configuration.
        :return: [tuple[str, str]] Co-action path and co-action/similarity-function path.
        """
        path_ca = f"{path_tw}{ca.get_co_action()}{os.sep}"
        path_ca_sf = f"{path_ca}{ca.get_similarity_function()}{os.sep}"
        return path_ca, path_ca_sf

    def _get_path_previous_and_filter(
        self,
        path_ca_sf: str,
        filter_instance: Any | None,
    ) -> tuple[str | None, str | None]:
        """
        Build the previous-filter and current-filter paths for a filter chain.

        :param path_ca_sf: [str] Co-action/similarity-function root path.
        :param filter_instance: [Any | None] Current Filter object. If None, no filter
            path is produced.
        :return: [tuple[str | None, str | None]] Previous-filter path and current-filter
            path, or (None, None) when no filter is configured.
        """
        if filter_instance is None:
            return None, None

        previous_filter = filter_instance.get_previous_filter()
        filter_concat = ""
        while previous_filter is not None:
            filter_concat = f"{str(previous_filter)}{filter_concat}{os.sep}"
            previous_filter = previous_filter.get_previous_filter()

        path_previous_filter = f"{path_ca_sf}{filter_concat}"
        path_filter = f"{path_ca_sf}{filter_concat}{str(filter_instance)}{os.sep}"
        return path_previous_filter, path_filter

    def _resolve_dynamic_filter_threshold(self, path_ca_sf: str, filter_instance: Any | None) -> None:
        """
        Resolve data-driven filter thresholds before deriving output paths.

        Filters such as median, mean, low_std, and high_std use the edge weights from
        the previous filtering stage. Config files therefore contain a placeholder
        threshold, but all downstream paths must use the resolved value, for example
        median_0.908 instead of median_0.0.

        :param path_ca_sf: [str] Co-action/similarity-function root path.
        :param filter_instance: [Any | None] Filter whose threshold may need resolution.
        :return: None. The filter object is updated in place when it is data-driven.
        """
        if filter_instance is None:
            return

        if filter_instance.get_type_filter() not in ["low_std", "mean", "high_std", "median"]:
            return

        if filter_instance.get_previous_filter() is None:
            path_read = f"{path_ca_sf}edge_list{os.sep}"
        else:
            path_previous_filter, _ = self._get_path_previous_and_filter(path_ca_sf, filter_instance)
            path_read = f"{path_previous_filter}edge_list{os.sep}"

        threshold = self._get_threshold_mean_std(filter_instance, path_read)
        filter_instance.set_threshold(threshold)

    def _get_threshold_mean_std(self, filter_instance: Any, path_read: str) -> float:
        """
        Compute a threshold from the edge weights of the previous filtering step.

        :param filter_instance: [Any] Filter object whose type is one of low_std, mean,
            high_std, or median.
        :param path_read: [str] Directory containing the source edge-list pickle files.
        :return: [float] Threshold value rounded to three decimals.
        """
        edge_list_files = [filename for filename in os.listdir(path_read) if filename.endswith(".p")]
        if len(edge_list_files) == 0:
            message = f"No edge-list pickle files found while resolving {filter_instance.get_type_filter()} threshold: {path_read}"
            self.lm.printl(message)
            raise FileNotFoundError(message)

        list_upper_bound = []
        list_mean = []
        list_median = []
        list_lower_bound = []

        for filename in edge_list_files:
            edge_list = self.ch.load_object(path_read + filename)
            weights = np.array([edge[2] for edge in edge_list])
            mean_value = np.mean(weights)
            std_deviation = np.std(weights)
            median_value = np.median(weights)

            list_upper_bound.append(round(mean_value + std_deviation, 2))
            list_lower_bound.append(round(mean_value - std_deviation, 2))
            list_mean.append(mean_value)
            list_median.append(median_value)

        threshold_by_filter = {
            "low_std": list_lower_bound[0],
            "mean": list_mean[0],
            "high_std": list_upper_bound[0],
            "median": list_median[0],
        }
        return round(threshold_by_filter[filter_instance.get_type_filter()], 3)

    def _get_type_algorithm(self, list_ca: list[Any]) -> str:
        """
        Return whether the selected co-actions define a one-layer or multi-layer network.

        :param list_ca: [list[Any]] Selected CoAction objects.
        :return: [str] one-layer when one co-action is selected, multi-layer otherwise.
        """
        if len(list_ca) == 0:
            message = "ValueError: Select at least one co-action."
            self.lm.printl(message)
            raise ValueError(message)
        if len(list_ca) == 1:
            return "one-layer"
        return "multi-layer"

    def _get_path_multi_co_action(
        self,
        list_ca: list[Any],
        dict_ca_filter: dict[str, Any],
    ) -> str:
        """
        Build the compact directory name for a multi-layer co-action instance.

        :param list_ca: [list[Any]] Selected CoAction objects.
        :param dict_ca_filter: [dict[str, Any]] Mapping from co-action id to Filter
            object or None.
        :return: [str] Full path to the multi-co-action instance directory.
        """
        path = f"{self.path_multi_co_action}"
        for index, ca_object in enumerate(list_ca):
            ca_type = ca_object.get_co_action()
            similarity_function_ca = ca_object.get_similarity_function()
            filter_ca = dict_ca_filter[ca_type]

            if filter_ca is not None:
                filter_ca_str_abbr = filter_ca.filter_repr_abbr()
                text_path_ca = (
                    f"{co_action_abbreviation_map[ca_type]}_"
                    f"{similarity_function_map[similarity_function_ca]}_"
                    f"{filter_ca_str_abbr}"
                )
            else:
                text_path_ca = (
                    f"{co_action_abbreviation_map[ca_type]}_"
                    f"{similarity_function_map[similarity_function_ca]}"
                )

            if index == 0:
                path = f"{path}{text_path_ca}"
            else:
                path = f"{path}__{text_path_ca}"
        return f"{path}{os.sep}"

    def _get_path_algorithm(self, path_community: str, cda: Any) -> str:
        """
        Build the community-detection algorithm directory.

        :param path_community: [str] Community root path for the current network.
        :param cda: [Any] Community-detection algorithm object.
        :return: [str] Algorithm-specific path.
        """
        return f"{path_community}{repr(cda)}{os.sep}"

    def _ensure_directory(self, path: str) -> None:
        """
        Create a directory through the shared pipeline helper.

        :param path: [str] Directory path to create.
        :return: None.
        """
        create_directory(self.file_name, path)

    def _ensure_directories(self, *paths: str) -> None:
        """
        Create multiple directories through the shared pipeline helper.

        :param paths: [str] Directory paths to create.
        :return: None.
        """
        for path in paths:
            self._ensure_directory(path)

    def get_filter(self) -> Any | None:
        """
        Return the filter object managed by this directory instance.

        :return: [Any | None] Current Filter object, possibly updated with a computed
            threshold, or None when no filter is configured.
        """
        return self.filter_instance

    def get_type_algorithm(self) -> str:
        """
        Return whether the current network is one-layer or multi-layer.

        :return: [str] one-layer or multi-layer.
        """
        return self.type_algorithm

    def update_data_path(self, prefix_path: str) -> None:
        """
        Prefix the configured data root path.

        :param prefix_path: [str] Prefix to prepend to `data_path`.
        :return: None. `data_path` is updated in place.
        """
        self.data_path = f"{prefix_path}{self.data_path}"
