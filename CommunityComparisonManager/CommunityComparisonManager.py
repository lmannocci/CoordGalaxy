from __future__ import annotations

from typing import Any, Sequence

from CommunityComparisonManager.core.ComparisonWorker import CommunityComparisonWorker
from utils.decorator_definition import log_method
from utils.LogManager.LogManager import LogManager


class CommunityComparisonManager:
    """
    Front-end class for comparing community-detection outputs from different characterization instances.

    This is the public interface used by main scripts. It forwards requests to the internal
    CommunityComparisonWorker, which delegates the actual work to focused computation,
    preparation, validation, coordination, and plotting modules.
    """

    def __init__(
        self,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str,
        tw: Any,
        list_ca: Sequence[Any],
        dict_ca_filter: dict[str, Any],
        file_prefix: str,
        community_size_th: int | None = None,
    ) -> None:
        """
        Create a community-comparison front end.

        :param dataset_name: [str] Dataset name.
        :param user_fraction: [float | None] User-selection fraction.
        :param type_filter: [str] User-selection strategy.
        :param tw: [Any] Time-window object.
        :param list_ca: [Sequence[Any]] Co-actions in the reference multiplex result.
        :param dict_ca_filter: [dict[str, Any]] Filter configuration by co-action id.
        :param file_prefix: [str] Prefix used for comparison output filenames.
        :param community_size_th: [int | None] Optional minimum community size.
        :return: None.
        """
        self.lm = LogManager("main")
        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter
        self.file_prefix = file_prefix
        self.community_size_th = community_size_th

    def _comparison_worker(self, chm_x: Any | None = None, chm_y: Any | None = None) -> CommunityComparisonWorker:
        """
        Build the internal comparison worker used by specialized operations.

        :param chm_x: [Any | None] First characterization manager.
        :param chm_y: [Any | None] Second characterization manager.
        :return: [CommunityComparisonWorker] Configured internal comparison worker.
        """
        return CommunityComparisonWorker(
            self.dataset_name,
            self.user_fraction,
            self.type_filter,
            self.tw,
            self.list_ca,
            self.dict_ca_filter,
            self.file_prefix,
            chm_x=chm_x,
            chm_y=chm_y,
            community_size_th=self.community_size_th,
        )

    @log_method
    def compute_overlap(
        self,
        chm_x: Any,
        chm_y: Any,
        save_overlapping_tensor: bool = True,
        save_intersections: bool = False,
    ) -> None:
        """
        Compute overlap matrices between two community outputs.

        :param chm_x: [Any] First characterization manager.
        :param chm_y: [Any] Second characterization manager.
        :param save_overlapping_tensor: [bool] Whether to save the overlap tensor.
        :param save_intersections: [bool] Whether to save intersection sets.
        :return: None. Outputs are saved by the internal comparison worker.
        """
        self._comparison_worker(chm_x, chm_y).compute_overlapping(save_overlapping_tensor, save_intersections)

    @log_method
    def compute_single_layer_nmi(self, chm_x: Any, chm_y: Any) -> None:
        """
        Compute NMI between two single-layer community outputs.

        :param chm_x: [Any] First characterization manager.
        :param chm_y: [Any] Second characterization manager.
        :return: None. Outputs are saved by the internal comparison worker.
        """
        self._comparison_worker(chm_x, chm_y).compute_single_layer_NMI()

    @log_method
    def plot_overlap_heatmaps(self, concat_matrix_bool: bool = True, plot_heatmap_list: list[str] | None = None) -> None:
        """
        Plot overlap heatmaps from saved overlap tensors.

        :param concat_matrix_bool: [bool] Whether to concatenate single-layer matrices.
        :param plot_heatmap_list: [list[str] | None] Optional list of generic labels to plot.
        :return: None. Plots are saved by the internal comparison worker.
        """
        self._comparison_worker().plot_heatmap_overlapping_matrix(concat_matrix_bool, plot_heatmap_list)

    @log_method
    def plot_single_layer_nmi_heatmap(self) -> None:
        """
        Plot the single-layer NMI heatmap from saved NMI rows.

        :return: None. Plot is saved by the internal comparison worker.
        """
        self._comparison_worker().plot_heatmap_single_layer_NMI()

    @log_method
    def plot_stacked_flux(
        self,
        type_aggregation: str,
        mid_th: float = 0.5,
        metric: str = "harmonicMean",
        plot_heatmap_list: list[str] | None = None,
    ) -> None:
        """
        Plot stacked flux from saved overlap tensors.

        :param type_aggregation: [str] Aggregation type: communities or users.
        :param mid_th: [float] Threshold separating common from gained/lost communities.
        :param metric: [str] Overlap metric.
        :param plot_heatmap_list: [list[str] | None] Optional list of generic labels to plot.
        :return: None. Outputs are saved by the internal comparison worker.
        """
        self._comparison_worker().plot_stacked_flux(type_aggregation, mid_th, metric, plot_heatmap_list)

    @log_method
    def combine_single_layer_metrics_communities(self, cda) -> None:
        """
        Combine single-layer community metrics used by single-layer comparison plots.

        :param cda: Community-detection algorithm object.
        :return: None. Combined metrics are saved by the internal comparison worker.
        """
        self._comparison_worker().combine_single_layer_metrics_communities(cda)

    @log_method
    def combine_node_metrics(self, cda) -> None:
        """
        Combine node metrics used by gained/lost/common node-metric plots.

        :param cda: Community-detection algorithm object.
        :return: None. Combined metrics are saved by the internal comparison worker.
        """
        self._comparison_worker().combine_node_metrics(cda)

    @log_method
    def plot_single_layer_metrics(
        self,
        visualization: str,
        type_visualization_starplot: str = "single",
        normalized: bool = True,
        features: list[str] | None = None,
        mid_th: float = 0.5,
        metric: str = "harmonicMean",
    ) -> None:
        """
        Plot or compute single-layer metric comparisons.

        :param visualization: [str] Visualization type: t_sne, umap, pca, starplot, or cosine_similarity.
        :param type_visualization_starplot: [str] Starplot layout: single or grid.
        :param normalized: [bool] Whether to normalize metric vectors.
        :param features: [list[str] | None] Optional metric columns to use.
        :param mid_th: [float] Flux threshold encoded in filenames.
        :param metric: [str] Overlap metric encoded in filenames.
        :return: None. Outputs are saved by the internal comparison worker.
        """
        self._comparison_worker().plot_single_layer_metrics(
            visualization,
            type_visualization_starplot=type_visualization_starplot,
            normalized=normalized,
            features=features,
            mid_th=mid_th,
            metric=metric,
        )

    @log_method
    def plot_barchart_cosine_similarity(
        self,
        generic: str = "co-retweet",
        normalized: bool = True,
        metric: str = "harmonicMean",
        mid_th: float = 0.5,
    ) -> None:
        """
        Plot cosine-similarity bars for one generic layer.

        :param generic: [str] Generic layer to plot.
        :param normalized: [bool] Whether the source cosine file used normalized metrics.
        :param metric: [str] Overlap metric encoded in filenames.
        :param mid_th: [float] Flux threshold encoded in filenames.
        :return: None. Plot is saved by the internal comparison worker.
        """
        self._comparison_worker().plot_barchart_cosine_similarity(generic, normalized, metric, mid_th)

    @log_method
    def plot_boxplot_metrics_gained_lost_nodes(self) -> None:
        """
        Plot node-metric boxplots for lost/common/gained user labels.

        :return: None. Outputs are saved by the internal comparison worker.
        """
        self._comparison_worker().plot_boxplot_metrics_gained_lost_nodes()

    @log_method
    def plot_node_metrics_gained_lost(
        self,
        chm_x: Any,
        chm_y: Any,
        metrics_node_to_compute: list[str],
        mid_th: float = 0.5,
        metric: str = "harmonicMean",
        th_size_metrics: int | None = None,
    ) -> None:
        """
        Plot KDE and distribution comparisons for gained/lost communities.

        :param chm_x: [Any] Generic/multiplex characterization manager.
        :param chm_y: [Any] Single-layer characterization manager.
        :param metrics_node_to_compute: [list[str]] Node/community metrics to plot.
        :param mid_th: [float] Flux threshold encoded in filenames.
        :param metric: [str] Overlap metric encoded in filenames.
        :param th_size_metrics: [int | None] Community-size threshold encoded in node-metric filenames.
        :return: None. Outputs are saved by the internal comparison worker.
        """
        self._comparison_worker(chm_x, chm_y).plot_node_metrics_gained_lost(
            metrics_node_to_compute,
            mid_th=mid_th,
            metric=metric,
            th_size_metrics=th_size_metrics,
        )

    @log_method
    def combine_coordination_communities(self, cda) -> None:
        """
        Combine coordination metrics for comparison analyses.

        :param cda: Community-detection algorithm object.
        :return: None. Combined metrics are saved by the internal comparison worker.
        """
        self._comparison_worker().combine_coordination_communities(cda)

    @log_method
    def compute_coordination_by_label(self, mid_th: float = 0.5, metric: str = "harmonicMean") -> None:
        """
        Compute coordination summaries for lost/common/gained overlap labels.

        :param mid_th: [float] Flux threshold encoded in filenames.
        :param metric: [str] Overlap metric encoded in filenames.
        :return: None. Summary is saved by the internal comparison worker.
        """
        self._comparison_worker().compute_coordination_by_label(mid_th, metric)

    @log_method
    def plot_coordination_by_label(
        self,
        cda,
        metric: str = "harmonicMean",
        mid_th: float = 0.5,
        metric_to_plot: str = "avg_weight",
    ) -> None:
        """
        Plot coordination metric distributions by overlap label.

        :param cda: Community-detection algorithm object.
        :param metric: [str] Overlap metric encoded in filenames.
        :param mid_th: [float] Flux threshold encoded in filenames.
        :param metric_to_plot: [str] Coordination metric prefix to plot.
        :return: None. Plots are saved by the internal comparison worker.
        """
        self._comparison_worker().plot_coordination_by_label(cda, metric, mid_th, metric_to_plot)

    @log_method
    def combine_validation_communities(self, cda) -> None:
        """
        Combine validation metrics for comparison analyses.

        :param cda: Community-detection algorithm object.
        :return: None. Combined metrics are saved by the internal comparison worker.
        """
        self._comparison_worker().combine_validation_communities(cda)

    @log_method
    def compute_validation_by_label(self, mid_th: float = 0.5, metric: str = "harmonicMean") -> None:
        """
        Compute validation-category counts for lost/common/gained overlap labels.

        :param mid_th: [float] Flux threshold encoded in filenames.
        :param metric: [str] Overlap metric encoded in filenames.
        :return: None. Summary is saved by the internal comparison worker.
        """
        self._comparison_worker().compute_validation_by_label(mid_th, metric)

    @log_method
    def plot_validation_multimodal(self, cda) -> None:
        """
        Plot validation summaries for multimodal and flattened community assignments.

        :param cda: Community-detection algorithm object.
        :return: None. Plots are saved by the internal comparison worker.
        """
        self._comparison_worker().plot_validation_multimodal(cda)
