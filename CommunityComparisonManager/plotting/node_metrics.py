"""Node-metric plots for gained/lost/common overlap analysis."""

from __future__ import annotations

import os
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from statannotations.Annotator import Annotator

from utils.common_variables import dpi, dtype, flatten_algorithm, palette
from utils.decorator_definition import log_method


file_name = os.path.splitext(os.path.basename(__file__))[0]


class NodeMetricsPlotter:
    """Plot node-level metrics for lost/common/gained overlap groups."""

    def _get_labelled_metrics_df(
        self,
        flux_selected_df: pd.DataFrame,
        node_metrics_generic_df: pd.DataFrame,
        node_metrics_layer_df: pd.DataFrame,
        generic: str,
        layer: str,
    ) -> pd.DataFrame:
        """
        Build a dataframe of node/community metrics labelled as lost or gained.

        :param flux_selected_df: [pd.DataFrame] Flux rows for one generic/layer pair.
        :param node_metrics_generic_df: [pd.DataFrame] Node metrics for the generic partition.
        :param node_metrics_layer_df: [pd.DataFrame] Node metrics for the layer partition.
        :param generic: [str] Generic layer label.
        :param layer: [str] Single-layer label.
        :return: [pd.DataFrame] Metrics dataframe with a label column.
        """
        label_lost_df = flux_selected_df[flux_selected_df["label"] == "lost"][["com_layer"]]
        lost_communities_metrics = node_metrics_layer_df.merge(
            label_lost_df,
            how="inner",
            left_on="community",
            right_on="com_layer",
        )
        lost_communities_metrics.drop(columns=["com_layer"], inplace=True)
        lost_communities_metrics["label"] = "lost"

        label_gained_df = flux_selected_df[flux_selected_df["label"] == "gained"][["com_generic"]]
        if generic == "multimodal":
            node_metrics_generic_df = node_metrics_generic_df[node_metrics_generic_df["layer"] == layer]
        elif generic in flatten_algorithm:
            pass
        gained_communities_metrics = node_metrics_generic_df.merge(
            label_gained_df,
            how="inner",
            left_on="community",
            right_on="com_generic",
        )
        gained_communities_metrics.drop(columns=["com_generic"], inplace=True)
        gained_communities_metrics["label"] = "gained"
        return pd.concat([lost_communities_metrics, gained_communities_metrics])

    def _plot_kde(
        self,
        metrics_node_to_compute: list[str],
        labelled_metrics_df: pd.DataFrame,
        generic: str,
        layer: str,
        metric: str,
    ) -> None:
        """
        Plot pairwise KDEs for labelled node/community metrics.

        :param metrics_node_to_compute: [list[str]] Metrics to plot.
        :param labelled_metrics_df: [pd.DataFrame] Labelled metrics dataframe.
        :param generic: [str] Generic layer label.
        :param layer: [str] Single-layer label.
        :param metric: [str] Overlap metric used in filenames.
        :return: None. KDE plots are saved to disk.
        """
        default_palette = sns.color_palette()
        kde_palette = [default_palette[1], default_palette[2]]
        labels = labelled_metrics_df["label"].unique()

        for col1, col2 in combinations(metrics_node_to_compute, 2):
            if col1 == "role" or col2 == "role":
                continue
            self.lm.printl(f"{file_name}. KDE plot: {generic}/{layer}, {col1} vs {col2}")
            plt.figure(figsize=(8, 6))
            for label, color in zip(labels, kde_palette):
                subset = labelled_metrics_df[labelled_metrics_df["label"] == label]
                self._plot_kde_subset(subset, col1, col2, label, color)
            handles = [Line2D([0], [0], color=color) for color in kde_palette]
            plt.legend(handles, list(labels), title="label")
            plt.title(f"Kernel Density Plot {generic}/{layer}\n{col1} vs. {col2}")
            plt.savefig(
                f"{self.dm.path_overlapping_KDE_plot}{self.file_prefix}_{metric}_"
                f"th_size_{str(self.community_size_th)}_{generic}_{layer}_{col1}_{col2}_KDE.png"
            )
            plt.show()

    def _plot_kde_subset(self, subset: pd.DataFrame, col1: str, col2: str, label: str, color) -> None:
        """
        Plot one KDE subset with fallback settings for difficult distributions.

        :param subset: [pd.DataFrame] Data for one label.
        :param col1: [str] X-axis metric.
        :param col2: [str] Y-axis metric.
        :param label: [str] Flux label.
        :param color: Matplotlib-compatible color.
        :return: None. The current Matplotlib axis is updated.
        """
        try:
            sns.kdeplot(data=subset, x=col1, y=col2, fill=True, alpha=0.5, label=label, color=color)
        except Exception as error:
            self.lm.printl(f"{file_name}. KDE fallback levels=7 for {col1}/{col2}, label={label}: {str(error)}")
            try:
                sns.kdeplot(
                    data=subset,
                    x=col1,
                    y=col2,
                    fill=True,
                    alpha=0.5,
                    label=label,
                    color=color,
                    levels=7,
                    clip=((0, 0.002), (0, 0.002)),
                    bw_adjust=1.2,
                )
            except Exception as second_error:
                self.lm.printl(
                    f"{file_name}. KDE fallback percentile clipping for {col1}/{col2}, "
                    f"label={label}: {str(second_error)}"
                )
                x_min, x_max = np.percentile(subset[col1], (1, 91))
                y_min, y_max = np.percentile(subset[col2], (1, 91))
                sns.kdeplot(
                    data=subset,
                    x=col1,
                    y=col2,
                    fill=True,
                    alpha=0.5,
                    label=label,
                    color=color,
                    levels=7,
                    clip=((x_min, x_max), (y_min, y_max)),
                    bw_adjust=1.2,
                )

    def _plot_distribution(
        self,
        metrics_node_to_compute: list[str],
        labelled_metrics_df: pd.DataFrame,
        generic: str,
        layer: str,
        metric: str,
    ) -> None:
        """
        Plot one-dimensional distributions for labelled node/community metrics.

        :param metrics_node_to_compute: [list[str]] Metrics to plot.
        :param labelled_metrics_df: [pd.DataFrame] Labelled metrics dataframe.
        :param generic: [str] Generic layer label.
        :param layer: [str] Single-layer label.
        :param metric: [str] Overlap metric used in filenames.
        :return: None. Distribution plots are saved to disk.
        """
        for col in metrics_node_to_compute:
            if col == "role":
                continue
            plt.figure(figsize=(8, 6))
            for label in labelled_metrics_df["label"].unique():
                subset = labelled_metrics_df[labelled_metrics_df["label"] == label]
                sns.kdeplot(data=subset, x=col, fill=True, alpha=0.5, label=label)
            plt.title(f"Distribution of {col}")
            plt.legend(title="Label", loc="best")
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.savefig(
                f"{self.dm.path_overlapping_distribution_plot}{self.file_prefix}_{metric}_"
                f"th_size_{str(self.community_size_th)}_{generic}_{layer}__{col}_distribution.png"
            )
            plt.show()

    def _load_gained_lost_metric_inputs(
        self,
        generic: str,
        layer: str,
        metric: str,
        mid_th: float,
        th_size_metrics: int | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load flux and node-metric data used by gained/lost metric plots.

        :param generic: [str] Generic layer label.
        :param layer: [str] Single-layer label.
        :param metric: [str] Overlap metric encoded in the flux filename.
        :param mid_th: [float] Flux threshold encoded in the flux filename.
        :param th_size_metrics: [int | None] Community-size threshold encoded in metric filenames.
        :return: [tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] Flux, generic metrics, and layer metrics.
        """
        flux_df = self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_"
            f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv",
            dtype=dtype,
        )
        file_multimodal = (
            self.dm_x.path_community_analysis
            + f"{self.chm_x.get_cda().get_algorithm_name()}_th_size_{str(th_size_metrics)}_node_metrics_communities.csv"
        )
        file_layer = self.dm_y.path_community_analysis + f"{layer}_th_size_{str(th_size_metrics)}_node_metrics_communities.csv"
        node_metrics_generic_df = self.ch_x.read_dataframe(file_multimodal, dtype=dtype)
        node_metrics_layer_df = self.ch_y.read_dataframe(file_layer, dtype=dtype)
        flux_selected_df = flux_df.loc[(flux_df["generic"] == generic) & (flux_df["layer"] == layer)].copy()
        return flux_selected_df, node_metrics_generic_df, node_metrics_layer_df

    @log_method
    def plot_node_metrics_gained_lost(
        self,
        metrics_node_to_compute: list[str],
        mid_th: float = 0.5,
        metric: str = "harmonicMean",
        th_size_metrics: int | None = None,
    ) -> None:
        """
        Plot KDE and distribution comparisons for gained/lost communities.

        :param metrics_node_to_compute: [list[str]] Node/community metrics to plot.
        :param mid_th: [float] Threshold used in the flux dataframe filename.
        :param metric: [str] Overlap metric used in the flux dataframe filename.
        :param th_size_metrics: [int | None] Threshold used in node-metric filenames.
        :return: None. Plots are saved to disk.
        """
        generic, layer = self.community_data_preparer.get_labels_reordered(
            self.type_algorithm_x,
            self.type_algorithm_y,
            self.list_ca_x,
            self.list_ca_y,
            self.cda_x,
            self.cda_y,
        )
        flux_selected_df, node_metrics_generic_df, node_metrics_layer_df = self._load_gained_lost_metric_inputs(
            generic,
            layer,
            metric,
            mid_th,
            th_size_metrics,
        )
        labelled_metrics_df = self._get_labelled_metrics_df(
            flux_selected_df,
            node_metrics_generic_df,
            node_metrics_layer_df,
            generic,
            layer,
        )
        self._plot_kde(metrics_node_to_compute, labelled_metrics_df, generic, layer, metric)
        self._plot_distribution(metrics_node_to_compute, labelled_metrics_df, generic, layer, metric)

    def _load_node_boxplot_inputs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load node labels and node metrics used by boxplot generation.

        :return: [tuple[pd.DataFrame, pd.DataFrame]] Flux node labels and node metrics.
        """
        flux_df = self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_th_size_{str(self.community_size_th)}_node_labelling.csv",
            dtype=dtype,
        )
        node_metrics_df = self.ch.read_dataframe(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_node_metrics.csv",
            dtype=dtype,
        )
        node_metrics_df = node_metrics_df.rename(columns={"nodeId": "userId"})
        self.lm.printl(f"{file_name}. node metrics shape: {str(node_metrics_df.shape)}")
        self.lm.printl(f"{file_name}. flux labels shape: {str(flux_df.shape)}")
        return flux_df, node_metrics_df

    def _prepare_boxplot_pairs(self, merge_df: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:
        """
        Prepare label order and valid statistical-test pairs for one boxplot dataframe.

        :param merge_df: [pd.DataFrame] Node metrics merged with flux labels.
        :return: [tuple[list[str], list[tuple[str, str]]]] Label order and valid label pairs.
        """
        label_order = ["lost", "common", "gained"]
        candidate_pairs = [("common", "gained"), ("lost", "common"), ("lost", "gained")]
        unique_labels = merge_df["label"].unique().tolist()
        label_order = [label for label in label_order if label in unique_labels]
        pairs = [(label1, label2) for label1, label2 in candidate_pairs if label1 in unique_labels and label2 in unique_labels]
        self.lm.printl(f"{file_name}. labels in boxplot data: {unique_labels}, pairs: {pairs}")
        return label_order, pairs

    def _significance_from_p_value(self, p_value: float) -> str:
        """
        Convert a p-value into a significance label.

        :param p_value: [float] P-value.
        :return: [str] Significance label.
        """
        if p_value <= 0.001:
            return "***"
        if p_value <= 0.01:
            return "**"
        if p_value <= 0.05:
            return "*"
        return "ns"

    def _plot_metric_boxplot(
        self,
        merge_df: pd.DataFrame,
        metric_name: str,
        label_order: list[str],
        pairs: list[tuple[str, str]],
        results_pvalue: list[dict[str, object]],
        generic: str,
        layer: str,
        subplot_index: int,
    ) -> None:
        """
        Plot and annotate one metric boxplot inside a four-panel figure.

        :param merge_df: [pd.DataFrame] Node metrics merged with flux labels.
        :param metric_name: [str] Metric column to plot.
        :param label_order: [list[str]] Ordered labels to show.
        :param pairs: [list[tuple[str, str]]] Valid statistical-test label pairs.
        :param results_pvalue: [list[dict[str, object]]] Output list for statistical-test rows.
        :param generic: [str] Generic layer label.
        :param layer: [str] Single-layer label.
        :param subplot_index: [int] 1-based subplot index.
        :return: None. The current Matplotlib figure is updated.
        """
        scientific_notation_limits = {
            "degree_centrality": (-2, -2),
            "eigenvector_centrality": (-2, -2),
            "local_clustering_coefficient": None,
            "page_rank": (-4, -4),
        }
        pvalue_thresholds = [(1e-3, "***"), (1e-2, "**"), (5e-2, "*"), (1, "ns")]
        ax = plt.subplot(1, 4, subplot_index)
        sns.boxplot(
            x="label",
            y=metric_name,
            data=merge_df,
            palette=palette,
            order=label_order,
            hue="label",
            dodge=False,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1},
        )
        if metric_name != "local_clustering_coefficient":
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="scientific", scilimits=scientific_notation_limits[metric_name])
        ax.tick_params(axis="y", labelsize=14)
        ax.set_xlabel("")
        ax.set_ylabel(metric_name.replace("_", " "), fontsize=14)
        plt.xticks("")

        self.lm.printl(
            f"{file_name}. Running Brunner-Munzel tests for {generic}/{layer}/{metric_name}; "
            "pair order must match the statannotations configuration."
        )
        annotator = Annotator(ax, pairs, data=merge_df, x="label", y=metric_name, order=label_order)
        annotator.configure(test="Brunner-Munzel", text_format="star", pvalue_thresholds=pvalue_thresholds, loc="inside")
        _, test_results = annotator.apply_test().annotate()
        stat_results = [(result.data.stat_value, result.data.pvalue) for result in test_results]

        for (label1, label2), test_result in zip(pairs, stat_results):
            p_value = test_result[1]
            results_pvalue.append(
                {
                    "generic": generic,
                    "layer": layer,
                    "metric": metric_name,
                    "label1": label1,
                    "label2": label2,
                    "p_value": p_value,
                    "significance": self._significance_from_p_value(p_value),
                }
            )

    def _plot_node_metric_boxplot_group(
        self,
        generic_layer_flux_df: pd.DataFrame,
        node_metrics_layer_df: pd.DataFrame,
        generic: str,
        layer: str,
        results_pvalue: list[dict[str, object]],
    ) -> None:
        """
        Plot all node-metric boxplots for one generic/layer pair.

        :param generic_layer_flux_df: [pd.DataFrame] Flux labels for one generic/layer pair.
        :param node_metrics_layer_df: [pd.DataFrame] Node metrics for the layer.
        :param generic: [str] Generic layer label.
        :param layer: [str] Single-layer label.
        :param results_pvalue: [list[dict[str, object]]] Output p-value rows.
        :return: None. The figure is saved to disk.
        """
        merge_df = generic_layer_flux_df.merge(node_metrics_layer_df, on=["userId", "layer"], how="inner")
        merge_df.drop(columns=["com_layer_x", "com_layer_y", "communities", "com_generic"], inplace=True, errors="ignore")
        label_order, pairs = self._prepare_boxplot_pairs(merge_df)
        if not pairs:
            self.lm.printl(
                f"{file_name}. No valid boxplot statistical pairs for generic={generic}, layer={layer}, "
                f"prefix={self.file_prefix}."
            )
            return

        plt.figure(figsize=(7, 3.5))
        for subplot_index, metric_name in enumerate(
            ["degree_centrality", "eigenvector_centrality", "local_clustering_coefficient", "page_rank"],
            start=1,
        ):
            self._plot_metric_boxplot(
                merge_df,
                metric_name,
                label_order,
                pairs,
                results_pvalue,
                generic,
                layer,
                subplot_index,
            )

        plt.subplots_adjust(wspace=1)
        filename = f"{self.dm.path_node_metrics_boxplot}{self.file_prefix}_boxplot_node_metrics_{generic}_{layer}.png"
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        self.lm.printl(f"{file_name}. {filename} saved.")
        plt.show()

    def _save_boxplot_legend(self) -> None:
        """
        Save a standalone horizontal legend for node-metric boxplots.

        :return: None. The legend figure is saved to disk.
        """
        legend_fig = plt.figure(figsize=(6, 2))
        legend_ax = legend_fig.add_subplot(111)
        handles = [plt.Line2D([0], [0], color=palette[key], lw=6) for key in palette.keys()]
        labels = list(palette.keys())
        legend_ax.legend(handles, labels, loc="center", frameon=False, ncol=len(labels))
        legend_ax.axis("off")
        legend_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.tight_layout()
        plt.savefig(
            f"{self.dm.path_node_metrics_boxplot}{self.file_prefix}_legend_horizontal.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.show()

    @log_method
    def plot_boxplot_metrics_gained_lost_nodes(self) -> None:
        """
        Plot node-metric boxplots for lost/common/gained user labels.

        :return: None. Boxplots, legend, and Brunner-Munzel results are saved to disk.
        """
        flux_df, node_metrics_df = self._load_node_boxplot_inputs()
        results_pvalue = []

        for generic in flux_df["generic"].unique():
            generic_flux_df = flux_df[flux_df["generic"] == generic]
            for layer in generic_flux_df["layer"].unique():
                self.lm.printl(f"{file_name}. Plot boxplot for generic={generic}, layer={layer}, prefix={self.file_prefix}.")
                generic_layer_flux_df = generic_flux_df[generic_flux_df["layer"] == layer]
                node_metrics_layer_df = node_metrics_df[node_metrics_df["layer"] == layer]
                self._plot_node_metric_boxplot_group(
                    generic_layer_flux_df,
                    node_metrics_layer_df,
                    generic,
                    layer,
                    results_pvalue,
                )

        self._save_boxplot_legend()
        df_results_pvalue = pd.DataFrame(results_pvalue)
        self.ch.save_dataframe(
            df_results_pvalue,
            f"{self.dm.path_node_metrics_boxplot}{self.file_prefix}_brunner_munzel_results.csv",
        )
