import os
from typing import Any, Sequence

from CharacterizationManager.Characterization.MultiplexCommunitiesEvaluation.MultiplexCommunitiesEvaluation import (
    MultiplexCommunitiesEvaluation,
)
from CharacterizationManager.Characterization.NetworkMeasure.NetworkMeasure import NetworkMeasure
from CharacterizationManager.Characterization.NodeMeasure.NodeMeasure import NodeMeasure
from CharacterizationManager.Characterization.SingleLayerCommunitiesEvaluation.SingleLayerCommuntiesEvaluation import (
    SingleLayerCommunitiesEvaluation,
)
from CharacterizationManager.Characterization.VisualizationManager.VisualizationManager import VisualizationManager
from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import IntegrityConstraintManager
from Objects.TimeWindow.TimeWindow import TimeWindow
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.LogManager.LogManager import LogManager
from utils.common_variables import flatten_algorithm
from utils.decorator_definition import log_method


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")


class CharacterizationManager:
    def __init__(
        self,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str,
        tw: TimeWindow,
        list_ca: Sequence[Any],
        dict_ca_filter: dict[str, Any],
        cda: Any | None = None
    ) -> None:
        """
            Create the characterization facade for network, node, visualization, and community-analysis modules.
            :param dataset_name: [str] Dataset directory name.
            :param user_fraction: [float | None] User-selection fraction used to resolve selected co-action data.
            :param type_filter: [str] User-selection strategy name used in directory paths.
            :param tw: [TimeWindow] Time-window configuration used by the network results.
            :param list_ca: [Sequence[Any]] Co-action objects included in the characterization.
            :param dict_ca_filter: [dict[str, Any]] Filter configuration by co-action id.
            :param cda: [Any | None] Community-detection algorithm configuration, when characterization follows CDA.
            :return: None.
        """
        self.lm = LogManager('main')
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
            data_path=data_path,
            results=results,
            user_fraction=self.user_fraction,
            type_filter=self.type_filter,
            tw=tw,
            list_ca=list_ca,
            dict_ca_filter=dict_ca_filter,
            cda=cda
        )
        self.type_algorithm = self.dm.get_type_algorithm()
        self.list_ca_str = '_'.join(list(self.dm.dict_path_ca.keys()))

        self._validate_configuration()
        self._build_sub_managers()

    def _validate_configuration(self) -> None:
        """
            Validate co-action/filter consistency and community-detection algorithm compatibility.
            :return: None.
        """
        self.icm.check_co_action(self.list_ca, self.dict_ca_filter)
        if self.cda is not None:
            self.icm.check_type_algorithm(self.tw, self.list_ca, self.cda.get_algorithm_name())

    def _build_sub_managers(self) -> None:
        """
            Build the specialized characterization helpers used by the public facade methods.
            :return: None.
        """
        self.nm = NetworkMeasure(self.list_ca, self.dict_ca_filter, self.icm, self.dm, self.type_algorithm)
        self.nom = NodeMeasure(self.list_ca, self.dict_ca_filter, self.icm, self.dm, self.type_algorithm, self.cda)
        self.vm = VisualizationManager(self.list_ca, self.dict_ca_filter, self.icm, self.dm, self.type_algorithm, self.cda)
        self.mce = MultiplexCommunitiesEvaluation(
            self.dataset_name,
            self.user_fraction,
            self.type_filter,
            self.list_ca,
            self.dict_ca_filter,
            self.icm,
            self.dm,
            self.type_algorithm,
            self.cda
        )
        self.sle = SingleLayerCommunitiesEvaluation(
            self.dataset_name,
            self.user_fraction,
            self.type_filter,
            self.list_ca,
            self.dict_ca_filter,
            self.icm,
            self.dm,
            self.type_algorithm,
            self.cda
        )

    def _uses_single_layer_community_flow(self) -> bool:
        """
            Return whether community analysis should use the single-layer/flattened-network implementation.
            :return: [bool] True for one-layer algorithms and flattened multiplex algorithms.
        """
        if self.type_algorithm == 'one-layer':
            return True
        if self.cda is None:
            return False
        return self.cda.get_algorithm_name() in flatten_algorithm

    def _current_algorithm_name(self) -> str | None:
        """
            Return the configured community-detection algorithm name.
            :return: [str | None] Algorithm name when a CDA object is configured, otherwise None.
        """
        if self.cda is None:
            return None
        return self.cda.get_algorithm_name()

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def compute_threshold_statistics(
        self,
        min_th: float,
        max_th: float,
        step: float,
        filter_par_type: str
    ) -> None:
        """
            Compute network statistics across a range of filtering thresholds.
            :param min_th: [float] Minimum threshold value.
            :param max_th: [float] Maximum threshold value.
            :param step: [float] Threshold step.
            :param filter_par_type: [str] Edge attribute used for thresholding, for example nAction or w_.
            :return: None. Statistics are saved by NetworkMeasure.
        """
        self.nm.compute_threshold_statistics(min_th, max_th, step, filter_par_type)

    @log_method
    def plot_threshold_statistics(self, filter_par_type: str, step: float) -> None:
        """
            Plot threshold-overlap and node/edge threshold statistics.
            :param filter_par_type: [str] Edge attribute used for thresholding.
            :param step: [float] Threshold step used in saved statistics.
            :return: None. Plots are saved by NetworkMeasure.
        """
        self.nm.plot_threshold_overlapping(filter_par_type, step)
        self.nm.plot_nodes_edges_threshold(filter_par_type)

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
            Select threshold values from saved threshold statistics.
            :param min_th: [float] Minimum threshold value.
            :param max_th: [float] Maximum threshold value.
            :param step: [float] Threshold step.
            :param absolute_th_mode: [bool] Whether thresholds are interpreted as absolute values.
            :param filter_par_type: [str] Edge attribute used for thresholding.
            :param target_type: [str] Target selection type, for example node.
            :return: None. Selection output is saved by NetworkMeasure.
        """
        self.nm.select_threshold_statistics(min_th, max_th, step, absolute_th_mode, filter_par_type, target_type)

    @log_method
    def compute_network_metrics(
        self,
        metrics_to_compute: Sequence[str],
        recompute_existing: bool = False,
        use_existing_output: bool = True,
    ) -> None:
        """
            Compute network-level metrics for configured single-layer or multiplex networks.
            :param metrics_to_compute: [Sequence[str]] Metric names to compute.
            :param recompute_existing: [bool] If True, recompute metrics already present in the output CSV.
            :param use_existing_output: [bool] If True, update an existing network_metrics.csv or legacy metrics.csv.
            :return: None. Metric artifacts are saved by NetworkMeasure.
        """
        self.nm.compute_network_metrics(metrics_to_compute, recompute_existing, use_existing_output)

    @log_method
    def edge_weight_temporal_mean_std(self) -> None:
        """
            Compute temporal mean and standard deviation of edge weights.
            :return: None. Statistics are saved by NetworkMeasure.
        """
        self.nm.edge_weight_temporal_mean_std()

    @log_method
    def get_ML_layer_comparison(self, mode: str = "both") -> None:
        """
            Compute multiplex-layer comparison metrics before community detection.
            :param mode: [str] Operation mode: compute, plot, or both.
            :return: None. Comparison metrics are saved by NetworkMeasure.
        """
        self.nm.get_ML_layer_comparison(mode)

    @log_method
    def compute_network_node_metrics(self, metrics: Sequence[str], merge_existing: bool = False) -> None:
        """
            Compute node-level metrics for configured networks.
            :param metrics: [Sequence[str]] Node metric names to compute.
            :param merge_existing: [bool] Whether to merge with an existing node-metrics file.
            :return: None. Node metrics are saved by NodeMeasure.
        """
        self.nom.compute_network_node_metrics(metrics, merge_existing)

    @log_method
    def compute_community_summary_statistics(self) -> None:
        """
            Compute community-size and community-composition statistics after community detection.
            :return: None. Statistics are saved by the selected community-evaluation module.
        """
        if self._uses_single_layer_community_flow():
            self.sle.compute_community_summary_statistics()
        else:
            self.mce.compute_community_summary_statistics()

    @log_method
    def compute_multiplex_community_membership_summary(self) -> None:
        """
            Compute multiplex community information after multiplex community detection.
            :return: None. Information is saved by MultiplexCommunitiesEvaluation when applicable.
        """
        if self._uses_single_layer_community_flow():
            self.lm.printl(
                f"{file_name}. compute_multiplex_community_membership_summary implemented only for multiplex community discovery algorithm."
            )
            return
        self.mce.compute_multiplex_community_membership_summary()

    @log_method
    def compute_metrics_communities(self, community_size_th: int | None) -> None:
        """
            Compute community-level metrics for single-layer or flattened communities.
            :param community_size_th: [int | None] Minimum community size threshold.
            :return: None. Metrics are saved by SingleLayerCommunitiesEvaluation when applicable.
        """
        if self._uses_single_layer_community_flow():
            self.sle.compute_metrics_communities(community_size_th)
            return
        self.lm.printl(
            f"{file_name}. compute_metrics_communities implemented only for single layer community discovery algorithm."
        )

    @log_method
    def compute_metrics_nodes_communities(
        self,
        metrics: Sequence[str] | None = None,
        th_size: int | None = None,
        restrict_neighbors: bool = True,
        merge_existing: bool = False
    ) -> None:
        """
            Compute node metrics inside communities after community detection.
            :param metrics: [Sequence[str] | None] Node metrics to compute. None uses module defaults.
            :param th_size: [int | None] Minimum community size threshold.
            :param restrict_neighbors: [bool] Whether node-neighbor computations are restricted to community members.
            :param merge_existing: [bool] Whether to merge with existing community-node metrics.
            :return: None. Metrics are saved by the selected community-evaluation module.
        """
        if self._uses_single_layer_community_flow():
            self.sle.compute_metrics_node_communities(metrics, th_size, restrict_neighbors, merge_existing)
        else:
            self.mce.compute_metrics_node_communities(metrics, th_size, restrict_neighbors, merge_existing)

    @log_method
    def validate_communities(self) -> None:
        """
            Validate detected communities against isControl labels when the input includes them.
            :return: None. Validation artifacts are saved by the selected community-evaluation module, or validation is
                skipped when isControl is not available.
        """
        if self._uses_single_layer_community_flow():
            self.sle.validate_communities()
        else:
            self.mce.validate_communities()

    @log_method
    def compute_community_edge_weight_statistics(
        self,
        community_size_th: int | None = None,
        community_label: str = "group",
        weight_label: str = "w_"
    ) -> None:
        """
            Compute within-community edge-weight statistics for detected communities.
            :param community_size_th: [int | None] Minimum community size threshold.
            :param community_label: [str] Node attribute containing the community id.
            :param weight_label: [str] Edge attribute containing edge weights.
            :return: None. Edge-weight statistics are saved by the selected community-evaluation module.
        """
        if not self._uses_single_layer_community_flow():
            self.mce.compute_community_edge_weight_statistics(community_size_th, community_label, weight_label)
            return

        algorithm_name = self._current_algorithm_name()
        if algorithm_name in {'flat_nw_infomap', 'flat_nw_louvain'}:
            self.lm.printl(
                f"{file_name}. compute_community_edge_weight_statistics not applicable for flatten algorithms without weights."
            )
            return
        self.sle.compute_community_edge_weight_statistics(community_size_th, community_label, weight_label)

    @log_method
    def visualize_multiplex_network(self) -> None:
        """
            Visualize the configured multiplex network.
            :return: None. Visualization artifacts are saved by VisualizationManager.
        """
        self.vm.visualize_multiplex_network()

    @log_method
    def delete_edges_visualize_multiplex_network(self) -> None:
        """
            Visualize a multiplex network after removing intra-community or selected edges.
            :return: None. Visualization artifacts are saved by VisualizationManager when applicable.
        """
        if self._uses_single_layer_community_flow():
            self.lm.printl(
                f"{file_name}. delete_edges_visualize_multiplex_network implemented only for multiplex community discovery algorithm."
            )
            return
        self.vm.delete_edges_visualize_multiplex_network()

    @log_method
    def delete_small_communities_single_layer(self, th_size: int) -> None:
        """
            Remove small single-layer communities from visualization outputs.
            :param th_size: [int] Minimum community size to keep.
            :return: None. Filtered visualization artifacts are saved by VisualizationManager.
        """
        self.vm.delete_small_communities_single_layer(th_size)

    @log_method
    def charactrize_url_layers_communities(self, top_n: int = 10, category_file: str | None = None) -> None:
        """
            Characterize URL domain frequencies and category percentages for multiplex communities.
            :param top_n: [int] Number of top communities to analyze.
            :param category_file: [str | None] Optional CSV file with domain,category columns.
            :return: None. URL characterization artifacts are saved by MultiplexCommunitiesEvaluation when applicable.
        """
        if self._uses_single_layer_community_flow():
            self.lm.printl(
                f"{file_name}. charactrize_url_layers_communities implemented only for multiplex community discovery algorithm."
            )
            return
        self.mce.charactrize_url_layers_communities(top_n, category_file)

    @log_method
    def save_top_communities_summary(self, top_n: int = 10) -> None:
        """
            Save structural and edge-weight information for the top multiplex communities.
            :param top_n: [int] Number of communities with the largest actor-layer count to save.
            :return: None. The summary is saved by MultiplexCommunitiesEvaluation when applicable.
        """
        if self._uses_single_layer_community_flow():
            self.lm.printl(
                f"{file_name}. save_top_communities_summary implemented only for multiplex community discovery algorithm."
            )
            return
        self.mce.save_top_communities_summary(top_n)

    def get_directory_manager(self) -> DirectoryManager:
        """
            Return the DirectoryManager used by this characterization manager.
            :return: [DirectoryManager] Directory manager instance.
        """
        return self.dm

    def get_checkpoint(self) -> Checkpoint:
        """
            Return the Checkpoint helper used by this characterization manager.
            :return: [Checkpoint] Checkpoint instance.
        """
        return self.ch

    def get_type_algorithm(self) -> str:
        """
            Return the detected characterization/community algorithm type.
            :return: [str] Algorithm type, for example one-layer or multi-layer.
        """
        return self.type_algorithm

    def get_list_ca(self) -> Sequence[Any]:
        """
            Return the configured co-action objects.
            :return: [Sequence[Any]] Co-action objects.
        """
        return self.list_ca

    def get_dict_ca_filter(self) -> dict[str, Any]:
        """
            Return the configured co-action filter dictionary.
            :return: [dict[str, Any]] Filter configuration by co-action id.
        """
        return self.dict_ca_filter

    def get_cda(self) -> Any | None:
        """
            Return the configured community-detection algorithm object.
            :return: [Any | None] CDA object when configured, otherwise None.
        """
        return self.cda
